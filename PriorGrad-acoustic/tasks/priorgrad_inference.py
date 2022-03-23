# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os, glob, re
from tts_utils.hparams import hparams, set_hparams
from tasks.priorgrad import PriorGradDataset
from tasks.priorgrad import PriorGradTask
import torch
import numpy as np
from tqdm import tqdm

set_hparams()

def get_latest_ckpt(dir):
    # function that returns the latest checkpoint path from the given dir
    ckpt_list = sorted(glob.glob(f'{dir}/model_ckpt_steps_*.ckpt'),
                      key=lambda x: -int(re.findall('.*steps\_(\d+)\.ckpt', x)[0]))
    print("INFO: located checkpoint {}. loading...".format(ckpt_list[0]))
    return ckpt_list[0]

# build PriorGradTask then the model itself
task = PriorGradTask()
task.model = task.build_model()

# load the latest checkpoint from work_dir defined in hparams
ckpt = torch.load(get_latest_ckpt(hparams['work_dir']))
task.global_step = ckpt['global_step']
task.load_state_dict(ckpt['state_dict'])

# load the fast noise schedule saved during test set inference
if hparams['fast']:
    best_schedule_name = 'betas' + str(hparams['fast_iter']) + '_' + hparams['work_dir'].split('/')[-1] + '_' + str(task.global_step) + '.npy'
    best_schedule = np.load(os.path.join(hparams['work_dir'], best_schedule_name))
    task.model.decoder.params.inference_noise_schedule = best_schedule
    print("INFO: saved noise schedule found in {}".format(os.path.join(hparams['work_dir'], best_schedule_name)))
    print("diffusion decoder inference noise schedule is reset to {}".format(best_schedule))

# load the model to gpu
task.model.eval().cuda()

# prepare hifi-gan vocoder
task.prepare_vocoder_hfg()

# define PriorGradDataset. will only use the functions (text_to_phone and phone_to_prior) and not the actual test dataset
dataset = PriorGradDataset(hparams['data_dir'], task.phone_encoder, None, hparams, shuffle=False, infer_only=True)

# inference requires phoneme input and the corresponding target_mean and target_std
with open(hparams['inference_text'], 'r') as f:
    user_text = f.readlines()

# create sample dir inside work_dir in hparams
if hparams['fast']:
    gen_dir = os.path.join(hparams['work_dir'],
                           f'inference_fast{hparams["fast_iter"]}_{task.global_step}')
else:
    gen_dir = os.path.join(hparams['work_dir'],
                           f'inference_{task.global_step}')
os.makedirs(gen_dir, exist_ok=True)
os.makedirs(f'{gen_dir}/text', exist_ok=True)
os.makedirs(f'{gen_dir}/spec', exist_ok=True)
os.makedirs(f'{gen_dir}/spec_plot', exist_ok=True)
os.makedirs(f'{gen_dir}/wavs', exist_ok=True)

# perform text-to-speech then save mel and wav
with torch.no_grad():
    for i, text in enumerate(tqdm(user_text)):
        phone = torch.LongTensor(dataset.text_to_phone(text))
        target_mean, target_std = dataset.phone_to_prior(phone)

        phone = phone.unsqueeze(0).cuda()
        target_mean, target_std = target_mean.unsqueeze(0).cuda(), target_std.unsqueeze(0).cuda()

        outputs = task.model(phone, None, None, None,
                             target_mean, target_std, None, None, None, None,
                             is_training=False, fast_sampling=hparams['fast'])
        mel_out = outputs['mel_out'].permute(0, 2, 1) # [1, num_mels, T]
        wav_out = task.vocoder(mel_out).squeeze().cpu().numpy() # [1, T_wav]
        # save mel and wav
        task.save_result(wav_out, mel_out.cpu()[0].T, f'P', i, text, gen_dir)

