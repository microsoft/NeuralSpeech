# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy
from scipy.spatial.transform import Rotation as R
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


class BinauralPreNet(nn.Module):
  def __init__(self, n_mels, binaural_type="", addmono=False, use_mean_condition=False, 
      predict_mean_condition=False):
    super().__init__()
    self.conv_view1 = torch.nn.Conv1d(7, 20, 3, padding=1)
    self.conv_view2 = torch.nn.Conv1d(20, 40, 3, padding=1)
    self.addmono = addmono
    self.use_mean_condition = use_mean_condition
    self.predict_mean_condition = predict_mean_condition
    if addmono:
      self.conv_dsp1 = torch.nn.Conv1d(3 + (0 if not use_mean_condition else 1), 20, 3, padding=1)
    else:
      self.conv_dsp1 = torch.nn.Conv1d(2 if not use_mean_condition else 3, 20, 3, padding=1)
    self.conv_dsp2 = torch.nn.Conv1d(20, 40, 3, padding=1)
    self.conv = torch.nn.Conv1d(80, n_mels, 3, padding=1)


  def forward(self, geowarp, view, mono, mean_condition):
    # geowarp = torch.unsqueeze(geowarp, 1)
    if self.addmono:
      if self.use_mean_condition:
        geowarp = torch.cat([geowarp, mono, mean_condition], axis=1)
      else:
        geowarp = torch.cat([geowarp, mono], axis=1)
    geowarp = self.conv_dsp1(geowarp)
    geowarp = F.leaky_relu(geowarp, 0.4)
    geowarp = self.conv_dsp2(geowarp)
    geowarp = F.leaky_relu(geowarp, 0.4)

    view = self.conv_view1(view)
    view = F.leaky_relu(view, 0.4)
    view = self.conv_view2(view)
    view = F.leaky_relu(view, 0.4)

    x = self.conv(torch.cat([geowarp, view], axis=1))
    x = F.leaky_relu(x, 0.4)
    return x


class ResidualBlock(nn.Module):
  def __init__(self, n_mels, residual_channels, dilation, uncond=False):
    '''
    :param n_mels: inplanes of conv1x1 for spectrogram conditional
    :param residual_channels: audio conv
    :param dilation: audio conv dilation
    :param uncond: disable spectrogram conditional
    '''
    super().__init__()
    self.dilated_conv = Conv1d(residual_channels, 2 * residual_channels, 3, padding=dilation, dilation=dilation)
    self.diffusion_projection = Linear(512, residual_channels)
    if not uncond: # conditional model
      self.conditioner_projection = Conv1d(n_mels, 2 * residual_channels, 1)
    else: # unconditional model
      self.conditioner_projection = None

    self.output_projection = Conv1d(residual_channels, 2 * residual_channels, 1)

  def forward(self, x, diffusion_step, conditioner=None):
    assert (conditioner is None and self.conditioner_projection is None) or \
           (conditioner is not None and self.conditioner_projection is not None)

    diffusion_step = self.diffusion_projection(diffusion_step).unsqueeze(-1)
    y = x + diffusion_step
    if self.conditioner_projection is None: # using a unconditional model
      y = self.dilated_conv(y)
    else:
      conditioner = self.conditioner_projection(conditioner)
      y = self.dilated_conv(y) + conditioner

    gate, filter = torch.chunk(y, 2, dim=1)
    y = torch.sigmoid(gate) * torch.tanh(filter)

    y = self.output_projection(y)
    residual, skip = torch.chunk(y, 2, dim=1)
    return (x + residual) / sqrt(2.0), skip


class BinauralGrad(nn.Module):
  def __init__(self, params, binaural_type=""):
    super().__init__()
    self.params = params
    self.binaural_type = binaural_type
    self.loss_per_layer = getattr(params, "loss_per_layer", 0)
    self.use_mean_condition = getattr(params, "use_mean_condition", False)
    self.predict_mean_condition = getattr(params, "predict_mean_condition", False)
    self.warper = None
    if not self.predict_mean_condition:
      self.input_projection = Conv1d(2, params.residual_channels, 1)
      self.output_projection = Conv1d(params.residual_channels, 2, 1)      
    else:
      self.input_projection = Conv1d(1, params.residual_channels, 1)
      self.output_projection = Conv1d(params.residual_channels, 1, 1)     
    self.diffusion_embedding = DiffusionEmbedding(len(params.noise_schedule))
    
    self.binaural_pre_net = BinauralPreNet(params.n_mels, binaural_type=binaural_type, addmono=getattr(params, "use_mono", False), 
      use_mean_condition=self.use_mean_condition, 
      predict_mean_condition=self.predict_mean_condition)
    self.spectrogram_upsampler = None

    self.residual_layers = nn.ModuleList([
        ResidualBlock(params.n_mels, params.residual_channels, 2**(i % params.dilation_cycle_length), uncond=params.unconditional)
        for i in range(params.residual_layers)
    ])
    self.skip_projection = Conv1d(params.residual_channels, params.residual_channels, 1)

    nn.init.zeros_(self.output_projection.weight)

  def forward(self, audio, diffusion_step, spectrogram=None, geowarp=None, view=None, mono=None, mean_condition=None):
    # x = audio.unsqueeze(1)
    x = audio
    x = self.input_projection(x)
    x = F.relu(x)

    diffusion_step = self.diffusion_embedding(diffusion_step)
    spectrogram = self.binaural_pre_net(geowarp, view, mono, mean_condition)

    skip = None
    extra_output = []
    for l_id, layer in enumerate(self.residual_layers):
      x, skip_connection = layer(x, diffusion_step, spectrogram)
      if self.loss_per_layer != 0 and l_id % self.loss_per_layer == self.loss_per_layer - 1:
        extra_output.append(self.output_projection(F.relu(self.skip_projection(skip / sqrt(l_id)))))
      skip = skip_connection if skip is None else skip_connection + skip

    x = skip / sqrt(len(self.residual_layers))
    x = self.skip_projection(x)
    x = F.relu(x)
    x = self.output_projection(x)
    if self.loss_per_layer != 0:
      return x, extra_output, geowarp
    else:
      return x, geowarp
