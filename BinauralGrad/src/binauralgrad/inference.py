# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import numpy as np
import os
import torch
import torchaudio
import math
from argparse import ArgumentParser

from binauralgrad.params import AttrDict
import binauralgrad.params as base_params
from binauralgrad.model import BinauralGrad

models = {}

def predict(spectrogram=None, binaural_geowarp=None, tx_view=None, mono=None, binaural_type=None, model_dir=None, params=None, mean_condition=None, device=torch.device('cuda'), fast_sampling=False):
  # Lazy load model.
  if not model_dir in models:
    if os.path.exists(f'{model_dir}/weights.pt'):
      checkpoint = torch.load(f'{model_dir}/weights.pt')
    else:
      checkpoint = torch.load(model_dir)

    model = BinauralGrad(AttrDict(params),  binaural_type=binaural_type).to(device)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    models[model_dir] = model

  model = models[model_dir]
  model.params.override(params)
  with torch.no_grad():
    training_noise_schedule = np.array(model.params.noise_schedule)
    inference_noise_schedule = np.array(model.params.inference_noise_schedule) if fast_sampling else training_noise_schedule

    talpha = 1 - training_noise_schedule
    talpha_cum = np.cumprod(talpha)

    beta = inference_noise_schedule
    alpha = 1 - beta
    alpha_cum = np.cumprod(alpha)

    T = []
    for s in range(len(inference_noise_schedule)):
      for t in range(len(training_noise_schedule) - 1):
        if talpha_cum[t+1] <= alpha_cum[s] <= talpha_cum[t]:
          twiddle = (talpha_cum[t]**0.5 - alpha_cum[s]**0.5) / (talpha_cum[t]**0.5 - talpha_cum[t+1]**0.5)
          T.append(t + twiddle)
          break
    T = np.array(T, dtype=np.float32)


    if not model.params.unconditional:
      if spectrogram is not None:
        if len(spectrogram.shape) == 2:# Expand rank 2 tensors by adding a batch dimension.
          spectrogram = spectrogram.unsqueeze(0)
        spectrogram = spectrogram.to(device)
        audio = torch.randn(spectrogram.shape[0], model.params.hop_samples * spectrogram.shape[-1], device=device)
      else:
        audio = torch.randn(binaural_geowarp.shape[0] - (1 if getattr(model.params, "predict_mean_condition", False) else 0), binaural_geowarp.shape[1], device=device).unsqueeze(0)
        binaural_geowarp = binaural_geowarp.unsqueeze(0).type_as(audio)
        tx_view = tx_view.unsqueeze(0).type_as(audio)
        mono = mono.unsqueeze(0).type_as(audio)
        mean_condition = mean_condition.unsqueeze(0).type_as(audio)
    else:
      audio = torch.randn(1, params.audio_len, device=device)
    noise_scale = torch.from_numpy(alpha_cum**0.5).float().unsqueeze(1).to(device)

    for n in range(len(alpha) - 1, -1, -1):
      c1 = 1 / alpha[n]**0.5
      c2 = beta[n] / (1 - alpha_cum[n])**0.5
      print(audio.shape, binaural_geowarp.shape, tx_view.shape)
      if params.loss_per_layer == 0:
        audio = c1 * (audio - c2 * model(audio, torch.tensor([T[n]], device=audio.device), spectrogram, geowarp=binaural_geowarp, view=tx_view, mono=mono, mean_condition=mean_condition)[0])
      else:
        audio = c1 * (audio - c2 * model(audio, torch.tensor([T[n]], device=audio.device), spectrogram, geowarp=binaural_geowarp, view=tx_view, mono=mono, mean_condition=mean_condition)[0])
      if n > 0:
        noise = torch.randn_like(audio)
        sigma = ((1.0 - alpha_cum[n-1]) / (1.0 - alpha_cum[n]) * beta[n])**0.5
        audio += sigma * noise
      audio = torch.clamp(audio, -1.0, 1.0)
  return audio[0], model.params.sample_rate


def main(args):
  if args.spectrogram_path:
    spectrogram = torch.from_numpy(np.load(args.spectrogram_path))
  else:
    spectrogram = None
  if args.dsp_path:
    mono, _ = torchaudio.load(f"{args.dsp_path}/mono.wav")
    binaural, _ = torchaudio.load(f"{args.dsp_path}/binaural.wav")
    binaural_geowarp, _ = torchaudio.load(f"{args.dsp_path}/binaural_geowarp.wav")
    # receiver is fixed at origin in this dataset, so we only need transmitter view
    tx_view = np.loadtxt(f"{args.dsp_path}/tx_positions.txt").transpose()
    tx_view = torch.from_numpy(np.repeat(tx_view.T, 400, axis=0).T)

    mean_condition_dsp = binaural_geowarp.mean(0, keepdim=True)

    mean_condition_gt = binaural.mean(0, keepdim=True)

    if args.use_gt_mean_condition:
      mean_condition = mean_condition_gt
    elif args.mean_condition_folder:
      mean_condition, _ = torchaudio.load(f"{args.mean_condition_folder}/{args.output.strip('/').split('/')[-1]}")
    else:
      mean_condition = mean_condition_dsp


  else:
    binaural_geowarp = None
    tx_view = None
    mono = None
    mean_condition = None
    mean_condition_dsp = None

  all_audio = []
  clip_len = 2000000 
  for i in range(int(math.ceil(binaural_geowarp.shape[1] / clip_len))):
    audio, sr = predict(spectrogram, binaural_geowarp=binaural_geowarp[ :, clip_len*i: clip_len*(i+1)], tx_view=tx_view[:, clip_len*i: clip_len*(i+1)],  mono=mono[:, clip_len*i: clip_len*(i+1)], 
      binaural_type=getattr(args, "binaural_type", ""), model_dir=args.model_dir, fast_sampling=args.fast, 
      params=getattr(base_params, args.params), mean_condition=mean_condition[:, clip_len*i: clip_len*(i+1)])
    if args.params in []:
      if "premean" in args.params:
        audio = audio.cpu() + binaural_geowarp[ :, clip_len*i: clip_len*(i+1)].mean(0, keepdim=True)
      else:
        audio = audio.cpu() + binaural_geowarp[ :, clip_len*i: clip_len*(i+1)]
    all_audio.append(audio)
  torchaudio.save(args.output, torch.cat(all_audio, axis=-1).cpu(), sample_rate=sr)


if __name__ == '__main__':
  parser = ArgumentParser(description='runs inference')
  parser.add_argument('model_dir',
      help='directory containing a trained model (or full path to weights.pt file)')
  parser.add_argument('--spectrogram_path', '-s',
      help='path to a spectrogram file')
  parser.add_argument('--dsp_path', '-d',
      help='path to dsp folder')
  parser.add_argument('--binaural_type', '-b',
      help='binaural type')  
  parser.add_argument('--output', '-o', default='output.wav',
      help='output file name')
  parser.add_argument('--fast', '-f', action='store_true',
      help='fast sampling procedure')
  parser.add_argument('--params', default="params", type=str,
    help='param set name')  
  parser.add_argument('--use-gt-mean-condition', action='store_true', default=False,
    help='use gt stage 1')
  parser.add_argument('--mean-condition-folder', default='', type=str,
    help='mean condition folder')
  main(parser.parse_args())
