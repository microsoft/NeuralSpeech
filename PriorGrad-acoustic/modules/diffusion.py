# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

# The diffusion acoustic decoder module is based on the DiffWave architecture: https://github.com/lmnt-com/diffwave
# Copyright 2020 LMNT, Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from math import sqrt


Linear = nn.Linear
ConvTranspose2d = nn.ConvTranspose2d


def Conv1d(*args, **kwargs):
  layer = nn.Conv1d(*args, **kwargs)
  nn.init.kaiming_normal_(layer.weight)
  return layer


@torch.jit.script
def silu(x):
  return x * torch.sigmoid(x)


class DiffusionEmbedding(nn.Module):
  def __init__(self, max_steps):
    super().__init__()
    self.register_buffer('embedding', self._build_embedding(max_steps), persistent=False)
    self.projection1 = Linear(128, 512)
    self.projection2 = Linear(512, 512)

  def forward(self, diffusion_step):
    if diffusion_step.dtype in [torch.int32, torch.int64]:
      x = self.embedding[diffusion_step]
    else:
      x = self._lerp_embedding(diffusion_step)
    x = self.projection1(x)
    x = silu(x)
    x = self.projection2(x)
    x = silu(x)
    return x

  def _lerp_embedding(self, t):
    low_idx = torch.floor(t).long()
    high_idx = torch.ceil(t).long()
    low = self.embedding[low_idx]
    high = self.embedding[high_idx]
    return low + (high - low) * (t - low_idx)

  def _build_embedding(self, max_steps):
    steps = torch.arange(max_steps).unsqueeze(1)  # [T,1]
    dims = torch.arange(64).unsqueeze(0)          # [1,64]
    table = steps * 10.0**(dims * 4.0 / 63.0)     # [T,64]
    table = torch.cat([torch.sin(table), torch.cos(table)], dim=1)
    return table


class ResidualBlock(nn.Module):
  def __init__(self, n_mels, residual_channels, conditioner_channels, dilation):
    super().__init__()
    self.dilated_conv = Conv1d(residual_channels, 2 * residual_channels, 3, padding=dilation, dilation=dilation)
    self.diffusion_projection = Linear(512, residual_channels)
    self.conditioner_projection = Conv1d(conditioner_channels, 2 * residual_channels, 1)
    self.output_projection = Conv1d(residual_channels, 2 * residual_channels, 1)

  def forward(self, x, conditioner, diffusion_step):
    diffusion_step = self.diffusion_projection(diffusion_step).unsqueeze(-1)
    conditioner = self.conditioner_projection(conditioner)

    y = x + diffusion_step
    y = self.dilated_conv(y) + conditioner

    gate, filter = torch.chunk(y, 2, dim=1)
    y = torch.sigmoid(gate) * torch.tanh(filter)

    y = self.output_projection(y)
    residual, skip = torch.chunk(y, 2, dim=1)
    return (x + residual) / sqrt(2.0), skip


class DiffDecoder(nn.Module):
  def __init__(self, params):
    super().__init__()
    self.params = params
    self.use_phone_stat = params['use_phone_stat']
    self.condition_phone_stat = params['condition_phone_stat'] if 'condition_phone_stat' in params else False
    if self.use_phone_stat and self.condition_phone_stat:
      self.input_projection = Conv1d(params.n_mels * 3 , params.residual_channels, 1) # to concat target_mean and target_std
    else:
      self.input_projection = Conv1d(params.n_mels, params.residual_channels, 1)
    self.diffusion_embedding = DiffusionEmbedding(len(params.noise_schedule))
    self.residual_layers = nn.ModuleList([
        ResidualBlock(params.n_mels, params.residual_channels, params.conditioner_channels, 2**(i % params.dilation_cycle_length))
        for i in range(params.residual_layers)
    ])
    self.skip_projection = Conv1d(params.residual_channels, params.residual_channels, 1)
    self.output_projection = Conv1d(params.residual_channels, params.n_mels, 1)
    nn.init.zeros_(self.output_projection.weight)

  def forward(self, input, decoder_inp, target_mean, target_std, mel2ph, diffusion_step):
    x = input.permute(0, 2, 1)
    decoder_inp = decoder_inp.permute(0, 2, 1)  # make it [B, 256, frame]
    mask = (mel2ph != 0).float().unsqueeze(1)  # [B, 1, frame]

    if self.use_phone_stat:
      assert target_mean is not None and target_std is not None
      if self.condition_phone_stat:
        target_mean = target_mean.permute(0, 2, 1)
        target_std = target_std.permute(0, 2, 1)
        x = torch.cat([x, target_mean, target_std], dim=1)

    x = self.input_projection(x)
    x = F.relu(x) * mask

    diffusion_step = self.diffusion_embedding(diffusion_step)

    skip = []
    for layer in self.residual_layers:
      x, skip_connection = layer(x, decoder_inp, diffusion_step)
      # apply mask
      x, skip_connection = x * mask, skip_connection * mask
      skip.append(skip_connection)

    x = torch.sum(torch.stack(skip), dim=0) / sqrt(len(self.residual_layers))
    x = self.skip_projection(x)
    x = F.relu(x) * mask
    x = self.output_projection(x) * mask
    x = x.permute(0, 2, 1) # back to [B, frame, 80]
    return x

  def sample(self, decoder_inp, target_mean, target_std, mel2ph, device=torch.device('cuda'), fast_sampling=False, return_all=False):
    with torch.no_grad():
      # Change in notation from the DiffWave paper for fast sampling.
      # DiffWave paper -> Implementation below
      # --------------------------------------
      # alpha -> talpha
      # beta -> training_noise_schedule
      # gamma -> alpha
      # eta -> beta
      training_noise_schedule = np.array(self.params.noise_schedule)
      inference_noise_schedule = np.array(
        self.params.inference_noise_schedule) if fast_sampling else training_noise_schedule

      talpha = 1 - training_noise_schedule
      talpha_cum = np.cumprod(talpha)

      beta = inference_noise_schedule
      alpha = 1 - beta
      alpha_cum = np.cumprod(alpha)

      T = []
      for s in range(len(inference_noise_schedule)):
        for t in range(len(training_noise_schedule) - 1):
          if talpha_cum[t + 1] <= alpha_cum[s] <= talpha_cum[t]:
            twiddle = (talpha_cum[t] ** 0.5 - alpha_cum[s] ** 0.5) / (talpha_cum[t] ** 0.5 - talpha_cum[t + 1] ** 0.5)
            T.append(t + twiddle)
            break
      T = np.array(T, dtype=np.float32)

      # Expand rank 2 tensors by adding a batch dimension.
      if len(decoder_inp.shape) == 2:
        decoder_inp = decoder_inp.unsqueeze(0)
      decoder_inp = decoder_inp.to(device)
      mel_list = []

      if target_mean is not None and target_std is not None:  # start from N(0, sigma)
        mel = torch.randn(decoder_inp.shape[0], decoder_inp.shape[1], self.params.n_mels, device=device) * target_std
      else:  # start from N(0, I)
        mel = torch.randn(decoder_inp.shape[0], decoder_inp.shape[1], self.params.n_mels, device=device)

      mel_list.append(mel.clone() + target_mean if target_mean is not None else mel.clone())

      # return "failed" mel if the target fast inference schedule is "unsupported" by the algorithm
      if len(T) != len(inference_noise_schedule):
        print("WARNING: given fast inference schedule {} is not supported. returning noise as output!".format(inference_noise_schedule))
        if return_all:
          return mel, mel_list
        else:
          return mel, None

      for n in range(len(alpha) - 1, -1, -1):
        c1 = 1 / alpha[n] ** 0.5
        c2 = beta[n] / (1 - alpha_cum[n]) ** 0.5
        if target_mean is not None and target_std is not None:
          mel = c1 * (mel - c2 * self.forward(mel, decoder_inp, target_mean, target_std, mel2ph, torch.tensor([T[n]], device=mel.device))) # mean prediction will be same as the original
        else:
          mel = c1 * (mel - c2 * self.forward(mel, decoder_inp, target_mean, target_std, mel2ph, torch.tensor([T[n]], device=mel.device)))
        if n > 0:
          if target_mean is not None and target_std is not None:
            noise = torch.randn_like(mel) * target_std
          else:
            noise = torch.randn_like(mel)
          sigma = ((1.0 - alpha_cum[n - 1]) / (1.0 - alpha_cum[n]) * beta[n]) ** 0.5
          mel += sigma * noise
        mel_list.append(mel.clone() + target_mean if target_mean is not None else mel.clone())

      if target_mean is not None:
        mel = mel + target_mean # recover mean from the denoised sample
        mel_list.append(mel.clone())

    if return_all:
      return mel, mel_list
    else:
      return mel, None