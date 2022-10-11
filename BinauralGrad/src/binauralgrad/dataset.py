# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import numpy as np
import os
import random
import torch
import torchaudio

from glob import glob
from torch.utils.data.distributed import DistributedSampler
import torch.nn.functional as F

class BinauralConditionalDataset(torch.utils.data.Dataset):
  def __init__(self, paths, binaural_type="", predict_mean_condition=False):
    super().__init__()
    self.mono, self.binaural, self.binaural_geowarp, self.view = [], [], [], []
    self.binaural_type = binaural_type
    self.predict_mean_condition = predict_mean_condition
    for subject_id in range(8):
      mono, _ = torchaudio.load(f"{paths}/subject{subject_id + 1}/mono.wav")
      binaural, _ = torchaudio.load(f"{paths}/subject{subject_id + 1}/binaural.wav")
      binaural_geowarp, _ = torchaudio.load(f"{paths}/subject{subject_id + 1}/binaural_geowarp.wav")
      # receiver is fixed at origin in this dataset, so we only need transmitter view
      tx_view = np.loadtxt(f"{paths}/subject{subject_id + 1}/tx_positions.txt").transpose()
      self.mono.append(mono)
      self.binaural.append(binaural)
      self.binaural_geowarp.append(binaural_geowarp)
      self.view.append(tx_view.astype(np.float32))
    # ensure that chunk_size is a multiple of 400 to match audio (48kHz) and receiver/transmitter positions (120Hz)
    self.chunk_size = 2000 * 48
    if self.chunk_size % 400 > 0:
      self.chunk_size = self.chunk_size + 400 - self.chunk_size % 400
    # compute chunks
    self.chunks = []
    for subject_id in range(8):
      last_chunk_start_frame = self.mono[subject_id].shape[-1] - self.chunk_size + 1
      hop_length = int((1 - 0.5) * self.chunk_size)
      for offset in range(0, last_chunk_start_frame, hop_length):
        self.chunks.append({'subject': subject_id, 'offset': offset})    

  def __len__(self):
    return len(self.chunks)

  def __getitem__(self, idx):
    subject = self.chunks[idx]['subject']
    offset = self.chunks[idx]['offset']
    mono = self.mono[subject][:, offset:offset+self.chunk_size]
    view = self.view[subject][:, offset//400:(offset+self.chunk_size)//400]    

    binaural = self.binaural[subject][0:2, offset:offset+self.chunk_size]
    binaural_geowarp = self.binaural_geowarp[subject][0:2, offset:offset+self.chunk_size]

    mean_condition = self.binaural[subject][0:2, offset:offset+self.chunk_size].mean(0, keepdim=True)

    return {
        'mono': mono,
        'binaural': binaural,
        'binaural_geowarp': binaural_geowarp,
        'view': view,
        'mean_condition': mean_condition,     
    }




class Collator:
  def __init__(self, params):
    self.params = params
  
  def collate_binaural(self, minibatch):

    clip_length = self.params.clip_length
    for record in minibatch:

      start_view = random.randint(0, record['mono'].shape[1] // 400 - clip_length // 400)
      start = start_view * 400
      end_view = start_view + clip_length // 400
      end = end_view * 400
      record['mono'] = record['mono'][:, start:end]
      record['mean_condition'] = record['mean_condition'][:, start:end]
      record['binaural'] = record['binaural'][:, start:end]
      record['binaural_geowarp'] = record['binaural_geowarp'][:, start:end]
      record['view'] = record['view'][:, start_view:end_view].T
      record['view'] = np.repeat(record['view'], 400, axis=0).T

    mono = np.stack([record['mono'] for record in minibatch if 'mono' in record])
    mean_condition = np.stack([record['mean_condition'] for record in minibatch if 'mean_condition' in record])
    binaural = np.stack([record['binaural'] for record in minibatch if 'binaural' in record])
    binaural_geowarp = np.stack([record['binaural_geowarp'] for record in minibatch if 'binaural_geowarp' in record])
    view = np.stack([record['view'] for record in minibatch if 'view' in record])

    assert binaural_geowarp.shape[0] == view.shape[0]

    return {
        'mono': torch.from_numpy(mono),
        'mean_condition': torch.from_numpy(mean_condition),
        'audio': torch.from_numpy(binaural),
        'binaural_geowarp': torch.from_numpy(binaural_geowarp),
        'view': torch.from_numpy(view),
    }

def from_path(data_dirs, params, binaural_type="", is_distributed=False):
  if binaural_type:
    dataset = BinauralConditionalDataset(data_dirs[0], binaural_type, 
      predict_mean_condition=getattr(params, "predict_mean_condition", False))
  else:
    raise ValueError("Unsupported binaural_type")
  return torch.utils.data.DataLoader(
      dataset,
      batch_size=params.batch_size,
      collate_fn=Collator(params).collate_binaural,
      shuffle=not is_distributed,
      num_workers=os.cpu_count(),
      sampler=DistributedSampler(dataset) if is_distributed else None,
      pin_memory=True,
      drop_last=True)
