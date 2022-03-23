# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os

os.environ["OMP_NUM_THREADS"] = "1"

from datasets.tts.utils import build_phone_encoder
from tts_utils.indexed_datasets import IndexedDatasetBuilder
import glob
import json
import logging
import sys
import traceback
from multiprocessing.pool import Pool
import numpy as np
import pandas as pd
from tqdm import tqdm
from tts_utils.hparams import hparams, set_hparams
from tts_utils.preprocessor import process_utterance_hfg, get_pitch, get_mel2ph_p

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format=log_format, datefmt='%m/%d %I:%M:%S %p')


def process_item(raw_data_dir, encoder, tg_fn, wav_fn):
    item_name = os.path.basename(wav_fn)[:-4].replace("lj_", "")
    spk_id = 0
    ph_fn = f'{raw_data_dir}/mfa_input_txt/{item_name}.ph'
    # spk_embed_fn = f'{raw_data_dir}/spk_embeds/{item_name}.npy'
    ph = open(ph_fn).readlines()[0].strip()
    ph = "<UNK> " + ph + " <EOS>"
    try:
        phone_encoded = encoder.encode(ph)
        wav_data, mel = process_utterance_hfg(
            wav_fn, fft_size=hparams['n_fft'],
            hop_size=hparams['hop_size'],
            win_length=hparams['win_size'],
            num_mels=hparams['audio_num_mel_bins'],
            fmin=hparams['fmin'],
            fmax=hparams['fmax'],
            sample_rate=hparams['audio_sample_rate'],
            loud_norm=hparams['loud_norm'],
            min_level_db=hparams['min_level_db'],
            return_linear=False, vocoder=hparams['vocoder'])
        mel = mel.T  # [T, 80]
    except:
        traceback.print_exc()
        print("| invalid data", item_name)
        return None

    mel2ph, mel2ph_encoded, dur = get_mel2ph_p(tg_fn, ph, phone_encoded, mel, hparams)
    f0, pitch_coarse = get_pitch(wav_data, mel, hparams)
    return item_name, phone_encoded, mel, mel2ph, mel2ph_encoded, spk_id, pitch_coarse, f0, dur


def process_data(raw_data_dir, encoder, wav_fns, data_dir, prefix):
    data_df = pd.read_csv(os.path.join(raw_data_dir, 'metadata_phone.csv'))
    fn2txt = {k: v for k, v in zip(data_df['wav'], data_df['txt1'])}

    p = Pool(int(os.getenv('N_PROC', os.cpu_count())))
    futures = []

    tg_fn = glob.glob(f"{raw_data_dir}/mfa_outputs/*/*.TextGrid")
    item2tgfn = {os.path.splitext(os.path.basename(v))[0]: v for v in tg_fn}
    for wav_fn in wav_fns:
        item_name = os.path.splitext(os.path.basename(wav_fn))[0].replace("lj_", "")
        # process_item(raw_data_dir, encoder, item2tgfn[item_name], wav_fn)
        futures.append(p.apply_async(process_item, args=(raw_data_dir, encoder, item2tgfn[item_name], wav_fn)))
    p.close()

    builder = IndexedDatasetBuilder(f'{data_dir}/{prefix}')
    all_keys = []
    lengths = []
    f0s = []
    durs = []

    mel2ph_encoded_list = []
    mel_list = []

    for future in tqdm(futures):
        res = future.get()
        if res is None:
            continue
        item_name, phone_encoded, mel, mel2ph, mel2ph_encoded, spk_id, pitch, f0, dur = res
        txt = fn2txt[item_name]
        item_name = f'lj_{item_name}'
        builder.add_item({
            'item_name': item_name,
            'txt': txt,
            'phone': phone_encoded,
            'mel': mel,
            'mel2ph': mel2ph,
            'mel2ph_encoded': mel2ph_encoded,
            'spk_id': spk_id,
            'pitch': pitch,
            'f0': f0,
        })
        lengths.append(mel.shape[0])
        all_keys.append(item_name)
        f0s.append(f0)
        durs.append(dur)

        mel2ph_encoded_list.append(mel2ph_encoded)
        mel_list.append(mel)

    p.join()
    builder.finalize()

    # calculate min and max value of mel feature
    mel_val_max = np.max([np.max(_) for _ in mel_list])
    mel_val_min = np.min([np.min(_) for _ in mel_list])


    # dict that store encoded phoneme with key & matching mel frame as value
    # construct dict, then get statistics
    num_max_phoneme = 100 # arbitrary
    phone_to_mel_frame = []
    phone_to_mean = []
    phone_to_std = []
    phone_to_std_norm = []

    for i in range(num_max_phoneme):
        phone_to_mel_frame.append([]) # empty list that will store frames
        phone_to_mean.append([])
        phone_to_std.append([])
        phone_to_std_norm.append([])

    print("computing phoneme-level mel statistics...")
    for i in tqdm(range(len(mel_list))):
        mel2ph_encoded, mel = mel2ph_encoded_list[i], mel_list[i]
        assert mel2ph_encoded.shape[0] == mel.shape[0]
        for j in range(len(mel2ph_encoded)):
            phone_to_mel_frame[mel2ph_encoded[j]].append(mel[j])
    # compute mean & std
    for i in range(num_max_phoneme):
        if phone_to_mel_frame[i] != []:
            phone_to_mean[i] = np.mean(phone_to_mel_frame[i], axis=0)
            phone_to_std[i] = np.std(phone_to_mel_frame[i], axis=0)

    std_min = 99999
    std_max = -1
    for i in range(num_max_phoneme):
        if phone_to_mel_frame[i] != []:
            std_min = min(std_min, np.min(phone_to_std[i]))
            std_max = max(std_max, np.max(phone_to_std[i]))

    # normalize the std to have maximal value of 1
    for i in range(num_max_phoneme):
        if phone_to_std[i] != []:
            phone_to_std_norm[i] = (phone_to_std[i] - std_min) / (std_max - std_min)

    # fill in "missing" slots, assuming N(0, I), but this will never be used
    for i in range(num_max_phoneme):
        if phone_to_std[i] == []:
            phone_to_std[i] = np.ones((80,), dtype=np.float32)
        if phone_to_mean[i] == []:
            phone_to_mean[i] = np.zeros((80,), dtype=np.float32)
        if phone_to_std_norm[i] == []:
            phone_to_std_norm[i] = np.ones((80,), dtype=np.float32)

    phone_to_mean = np.array(phone_to_mean)
    phone_to_std = np.array(phone_to_std)
    phone_to_std_norm = np.array(phone_to_std_norm)

    np.save(f'{data_dir}/{prefix}_all_keys.npy', all_keys)
    np.save(f'{data_dir}/{prefix}_lengths.npy', lengths)
    np.save(f'{data_dir}/{prefix}_f0s.npy', f0s)
    np.save(f'{data_dir}/{prefix}_durs.npy', durs)
    np.save(f'{data_dir}/{prefix}_phone_to_mean.npy', phone_to_mean)
    np.save(f'{data_dir}/{prefix}_phone_to_std.npy', phone_to_std)
    np.save(f'{data_dir}/{prefix}_phone_to_std_norm.npy', phone_to_std_norm)
    np.save(f'{data_dir}/{prefix}_mel_val_min.npy', mel_val_min)
    np.save(f'{data_dir}/{prefix}_mel_val_max.npy', mel_val_max)

if __name__ == "__main__":
    set_hparams()
    raw_data_dir = hparams['raw_data_dir']
    all_wav_fns = sorted(glob.glob(f'{raw_data_dir}/wavs/*.wav'))
    logging.info("train {}".format(len(all_wav_fns)))

    ph_set = [x.split(' ')[0] for x in open(f'{raw_data_dir}/{hparams["dict_file"]}.txt').readlines()]
    print(ph_set)
    os.makedirs(hparams['data_dir'], exist_ok=True)
    json.dump(ph_set, open(f"{hparams['data_dir']}/phone_set.json", 'w'))
    encoder = build_phone_encoder(hparams['data_dir'])

    # encoder = build_phone_encoder(raw_data_dir)
    os.makedirs(hparams['data_dir'], exist_ok=True)
    process_data(raw_data_dir, encoder, all_wav_fns[:100], hparams['data_dir'], 'valid')
    process_data(raw_data_dir, encoder, all_wav_fns[:100], hparams['data_dir'], 'test')
    process_data(raw_data_dir, encoder, all_wav_fns[100:], hparams['data_dir'], 'train')
