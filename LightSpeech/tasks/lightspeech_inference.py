# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os, glob, re
from tqdm import tqdm
import numpy as np
import torch
import utils
from utils.hparams import hparams, set_hparams
from tasks.lightspeech import LightSpeechDataset, LightSpeechTask

set_hparams()

def get_latest_ckpt(dir):
    # function that returns the latest checkpoint path from the given dir
    ckpt_list = sorted(glob.glob(f'{dir}/model_ckpt_steps_*.ckpt'),
                      key=lambda x: -int(re.findall('.*steps\_(\d+)\.ckpt', x)[0]))
    print("INFO: located checkpoint {}. loading...".format(ckpt_list[0]))
    return ckpt_list[0]

# build LightSpeechTask then the model itself
task = LightSpeechTask()
task.model = task.build_model()

# load the latest checkpoint from work_dir defined in hparams
ckpt = torch.load(get_latest_ckpt(hparams['work_dir']))
task.global_step = ckpt['global_step']
task.load_state_dict(ckpt['state_dict'])

# load the model to gpu
task.model.eval().cuda()

# prepare vocoder
task.prepare_vocoder()

# define LightSpeechDataset. will only use the functions (text_to_phone and phone_to_prior) and not the actual test dataset
dataset = LightSpeechDataset(hparams['data_dir'], task.phone_encoder, None, hparams, shuffle=False, infer_only=True)

# inference requires phoneme input and the corresponding target_mean and target_std
with open(hparams['inference_text'], 'r') as f:
    user_text = f.readlines()

# create sample dir inside work_dir in hparams
gen_dir = os.path.join(hparams['work_dir'], f"inference_{hparams['inference_text']}_{task.global_step}")
os.makedirs(gen_dir, exist_ok=True)
os.makedirs(f'{gen_dir}/wavs', exist_ok=True)
os.makedirs(f'{gen_dir}/spec_plot', exist_ok=True)
os.makedirs(f'{gen_dir}/pitch_plot', exist_ok=True)

# perform text-to-speech then save mel and wav
with torch.no_grad():
    for i, text in enumerate(tqdm(user_text)):
        text = text.strip()
        phone = torch.LongTensor(dataset.text_to_phone(text))
        phone = phone.unsqueeze(0).cuda()
        output = task.model(phone, None, None, None, None, None)
        output['outputs'] = output['mel_out']
        _output = utils.unpack_dict_to_list(output)[0]
        output = {}
        for k, v in _output.items():
            if type(v) is torch.Tensor:
                output[k] = v.cpu().numpy()
        mel_out = task.remove_padding(output['mel_out'])
        noise_outputs = task.remove_padding(output.get("noise_outputs"))
        pitch_pred = task.remove_padding(output.get("pitch"))
        wav_out = task.inv_spec(mel_out, pitch_pred, noise_outputs)
        # save mel and wav
        task.save_result(wav_out, mel_out, f'P', i, text, gen_dir, pitch=pitch_pred)

