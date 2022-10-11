# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import numpy as np
import os
import torch
import torch.nn as nn

from torch.nn.parallel import DistributedDataParallel
#from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from binauralgrad.losses import PhaseLoss
from binauralgrad.dataset import from_path
from binauralgrad.model import BinauralGrad
from binauralgrad.params import AttrDict


def _nested_map(struct, map_fn):
  if isinstance(struct, tuple):
    return tuple(_nested_map(x, map_fn) for x in struct)
  if isinstance(struct, list):
    return [_nested_map(x, map_fn) for x in struct]
  if isinstance(struct, dict):
    return { k: _nested_map(v, map_fn) for k, v in struct.items() }
  return map_fn(struct)


class BinauralGradLearner:
  def __init__(self, model_dir, model, dataset, optimizer, params, *args, **kwargs):
    os.makedirs(model_dir, exist_ok=True)
    self.model_dir = model_dir
    self.model = model
    self.dataset = dataset
    self.optimizer = optimizer
    self.params = params
    if params.lambda_phase != 0.0:
      if not getattr(params, "use_mstft", False):
        self.phase_loss = PhaseLoss(sample_rate=self.params.sample_rate)
      else:
        from binauralgrad.mstft_loss import MultiResolutionSTFTLoss
        self.phase_loss = MultiResolutionSTFTLoss(
                                                       sample_rate=self.params.sample_rate,
                                                       w_phs=1.0,
                                                       device="cuda")
    self.autocast = torch.cuda.amp.autocast(enabled=kwargs.get('fp16', False))
    self.scaler = torch.cuda.amp.GradScaler(enabled=kwargs.get('fp16', False))
    self.binaural_type = kwargs.get('binaural_type', "")
    self.step = 0
    self.is_master = True

    beta = np.array(self.params.noise_schedule)
    noise_level = np.cumprod(1 - beta)
    self.noise_level = torch.tensor(noise_level.astype(np.float32))
    if getattr(params, "use_l2_loss", False):
      # if params.use_l2_loss: 
      self.loss_fn = nn.MSELoss()
    else:
      self.loss_fn = nn.L1Loss()
    self.summary_writer = None

  def state_dict(self):
    if hasattr(self.model, 'module') and isinstance(self.model.module, nn.Module):
      model_state = self.model.module.state_dict()
    else:
      model_state = self.model.state_dict()
    return {
        'step': self.step,
        'model': { k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in model_state.items() },
        'optimizer': { k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in self.optimizer.state_dict().items() },
        'params': dict(self.params),
        'scaler': self.scaler.state_dict(),
    }

  def load_state_dict(self, state_dict):
    if hasattr(self.model, 'module') and isinstance(self.model.module, nn.Module):
      self.model.module.load_state_dict(state_dict['model'])
    else:
      self.model.load_state_dict(state_dict['model'])
    self.optimizer.load_state_dict(state_dict['optimizer'])
    self.scaler.load_state_dict(state_dict['scaler'])
    self.step = state_dict['step']

  def save_to_checkpoint(self, filename='weights'):
    save_basename = f'{filename}-{self.step}.pt'
    save_name = f'{self.model_dir}/{save_basename}'
    link_name = f'{self.model_dir}/{filename}.pt'
    torch.save(self.state_dict(), save_name)
    #if os.name == 'nt':
    #  torch.save(self.state_dict(), link_name)
    #else:
    #  if os.path.islink(link_name):
    #    os.unlink(link_name)
    #  os.symlink(save_basename, link_name)

  def restore_from_checkpoint(self, filename='weights'):
    try:
      checkpoint = torch.load(f'{self.model_dir}/{filename}.pt')
      self.load_state_dict(checkpoint)
      return True
    except FileNotFoundError:
      return False

  def train(self, max_steps=None):
    device = next(self.model.parameters()).device
    while True:
      for features in tqdm(self.dataset, desc=f'Epoch {self.step // len(self.dataset)}') if self.is_master else self.dataset:
        if max_steps is not None and self.step >= max_steps:
          return
        features = _nested_map(features, lambda x: x.to(device) if isinstance(x, torch.Tensor) else x)
        loss = self.train_step(features)
        if torch.isnan(loss).any():
          raise RuntimeError(f'Detected NaN loss at step {self.step}.')
        if self.is_master:
          #if self.step % 50 == 0:
          #  self._write_summary(self.step, features, loss)
          if self.step % (len(self.dataset) * 5)== 0: #save ckpt per 10 epoch
            self.save_to_checkpoint()
        self.step += 1

  def train_step(self, features):
    for param in self.model.parameters():
      param.grad = None

    if self.binaural_type:
      audio = features['audio']
      mono = features['mono']
      binaural_geowarp = features['binaural_geowarp']
      view = features['view']
      mean_condition = features['mean_condition']
      if getattr(self.params, "predict_mean_condition", False):
        audio = mean_condition
    else:
      audio = features['audio']
      spectrogram = features['spectrogram']
      mean_condition = None

    N, channel, T = audio.shape
    device = audio.device
    self.noise_level = self.noise_level.to(device)

    with self.autocast:
      t = torch.randint(0, len(self.params.noise_schedule), [N], device=audio.device)
      noise_scale = self.noise_level[t[:, None].repeat(1, channel)].unsqueeze(2)
      noise_scale_sqrt = noise_scale**0.5
      noise = torch.randn_like(audio)
      #print(audio.shape, t.shape, noise_scale.shape)
      noisy_audio = noise_scale_sqrt * audio + (1.0 - noise_scale)**0.5 * noise
      if self.binaural_type:
        if self.params.loss_per_layer != 0:
          predicted, extra_output, _ = self.model(noisy_audio, t, geowarp=binaural_geowarp, view=view, mono=mono, mean_condition=mean_condition)
      else:
        predicted = self.model(noisy_audio, t, spectrogram)
      loss = self.loss_fn(noise, predicted)
      if self.params.loss_per_layer != 0:
        extra_loss = self.loss_fn(torch.cat([noise] * (len(extra_output)), dim=1), torch.cat(extra_output, dim=1))
        if self.params.lambda_phase != 0.0:
          noisy_extra = torch.cat([noise_scale_sqrt * audio] * (len(extra_output)), dim=1) + (1.0 - torch.cat([noise_scale] * (len(extra_output)), dim=1))**0.5 * torch.cat(extra_output, dim=1)
          extra_loss += self.params.lambda_phase * self.phase_loss(torch.cat([noisy_audio] * (len(extra_output)), dim=1), noisy_extra)
        loss += extra_loss

      else:
        if self.params.lambda_phase != 0.0:
          noisy_predict = noise_scale_sqrt * audio + (1.0 - noise_scale)**0.5 * predicted
          loss += self.params.lambda_phase * self.phase_loss(noisy_audio, noisy_predict)




    self.scaler.scale(loss).backward()
    self.scaler.unscale_(self.optimizer)
    self.grad_norm = nn.utils.clip_grad_norm_(self.model.parameters(), self.params.max_grad_norm or 1e9)
    self.scaler.step(self.optimizer)
    self.scaler.update()
    return loss

  def _write_summary(self, step, features, loss):
    writer = self.summary_writer or SummaryWriter(self.model_dir, purge_step=step)
    writer.add_scalar('train/loss', loss, step)
    writer.add_scalar('train/grad_norm', self.grad_norm, step)
    writer.flush()
    self.summary_writer = writer


def _train_impl(replica_id, model, dataset, args, params, binaural_type=""):
  torch.backends.cudnn.benchmark = True
  opt = torch.optim.Adam(model.parameters(), lr=params.learning_rate)

  learner = BinauralGradLearner(args.model_dir, model, dataset, opt, params, fp16=args.fp16, binaural_type=binaural_type)
  learner.is_master = (replica_id == 0)
  learner.restore_from_checkpoint()
  learner.train(max_steps=args.max_steps)


def train(args, params):
  dataset = from_path(args.data_dirs, params, getattr(args, "binaural_type", ""))
  model = BinauralGrad(params, binaural_type=getattr(args, "binaural_type", "")).cuda()
  _train_impl(0, model, dataset, args, params, binaural_type=getattr(args, "binaural_type", ""))


def train_distributed(replica_id, replica_count, port, args, params):
  os.environ['MASTER_ADDR'] = 'localhost'
  os.environ['MASTER_PORT'] = str(port)
  torch.distributed.init_process_group('nccl', rank=replica_id, world_size=replica_count)

  dataset = from_path(args.data_dirs, params, getattr(args, "binaural_type", ""), is_distributed=True)
  device = torch.device('cuda', replica_id)
  torch.cuda.set_device(device)

  model = BinauralGrad(params, binaural_type=getattr(args, "binaural_type", "")).to(device)
  #print(model)
  model = DistributedDataParallel(model, device_ids=[replica_id])
  _train_impl(replica_id, model, dataset, args, params, binaural_type=getattr(args, "binaural_type", ""))
