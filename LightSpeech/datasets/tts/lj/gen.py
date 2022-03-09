# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os

os.environ["OMP_NUM_THREADS"] = "1"

from datasets.tts.utils import build_phone_encoder
from utils.indexed_datasets import IndexedDatasetBuilder
import glob
import json
import logging
import sys
import traceback
from multiprocessing.pool import Pool
import numpy as np
import pandas as pd
from tqdm import tqdm
from utils.hparams import hparams, set_hparams
from utils.preprocessor import process_utterance, get_pitch, get_mel2ph

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
        wav_data, mel = process_utterance(
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

    mel2ph, dur = get_mel2ph(tg_fn, ph, mel, hparams)
    f0, pitch_coarse = get_pitch(wav_data, mel, hparams)
    return item_name, phone_encoded, mel, mel2ph, spk_id, pitch_coarse, f0, dur


def process_data(raw_data_dir, encoder, wav_fns, data_dir, prefix):
    data_df = pd.read_csv(os.path.join(raw_data_dir, 'metadata_phone.csv'))
    fn2txt = {k: v for k, v in zip(data_df['wav'], data_df['txt1'])}

    p = Pool(int(os.getenv('N_PROC', os.cpu_count())))
    futures = []

    tg_fn = glob.glob(f"{raw_data_dir}/mfa_outputs/*/*.TextGrid")
    item2tgfn = {os.path.splitext(os.path.basename(v))[0]: v for v in tg_fn}
    for wav_fn in wav_fns:
        item_name = os.path.splitext(os.path.basename(wav_fn))[0].replace("lj_", "")
        futures.append(p.apply_async(process_item, args=(raw_data_dir, encoder, item2tgfn[item_name], wav_fn)))
    p.close()

    builder = IndexedDatasetBuilder(f'{data_dir}/{prefix}')
    all_keys = []
    lengths = []
    f0s = []
    durs = []
    for future in tqdm(futures):
        res = future.get()
        if res is None:
            continue
        item_name, phone_encoded, mel, mel2ph, spk_id, pitch, f0, dur = res
        txt = fn2txt[item_name]
        item_name = f'lj_{item_name}'
        builder.add_item({
            'item_name': item_name,
            'txt': txt,
            'phone': phone_encoded,
            'mel': mel,
            'mel2ph': mel2ph,
            'spk_id': spk_id,
            'pitch': pitch,
            'f0': f0,
        })
        lengths.append(mel.shape[0])
        all_keys.append(item_name)
        f0s.append(f0)
        durs.append(dur)
    p.join()
    builder.finalize()
    np.save(f'{data_dir}/{prefix}_all_keys.npy', all_keys)
    np.save(f'{data_dir}/{prefix}_lengths.npy', lengths)
    np.save(f'{data_dir}/{prefix}_f0s.npy', f0s)
    np.save(f'{data_dir}/{prefix}_durs.npy', durs)

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
    process_data(raw_data_dir, encoder, all_wav_fns[100:200], hparams['data_dir'], 'test')
    process_data(raw_data_dir, encoder, all_wav_fns[200:], hparams['data_dir'], 'train')
