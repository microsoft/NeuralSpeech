# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import sys
import librosa
import numpy as np
import scipy.linalg
from scipy.spatial.transform import Rotation as R
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import soundfile

from binauralgrad.warping import GeometricTimeWarper, MonotoneTimeWarper



mono_fn = sys.argv[1]
position_fn = sys.argv[2]
binaural_fn = sys.argv[3]

def load_position(position_fn):
    position_list = []
    with open(position_fn, 'r', encoding='utf-8') as infile:
        for line in infile.readlines():
            line = line.strip().split()
            assert len(line) == 7
            position_list.append([float(i) for i in line])
    return np.array(position_list)

mono_audio, sr = librosa.load(mono_fn, mono=True, sr=None)
position_array = load_position(position_fn=position_fn)

assert len(position_array) * 400 == len(mono_audio)
assert sr == 48000

class GeometricWarper(nn.Module):
    def __init__(self, sampling_rate=48000):
        super().__init__()
        self.warper = GeometricTimeWarper(sampling_rate=sampling_rate)

    def _transmitter_mouth(self, view):
        # offset between tracking markers and real mouth position in the dataset
        mouth_offset = np.array([0.09, 0, -0.20])
        quat = view[:, 3:, :].transpose(2, 1).contiguous().detach().cpu().view(-1, 4).numpy()
        # make sure zero-padded values are set to non-zero values (else scipy raises an exception)
        norms = scipy.linalg.norm(quat, axis=1)
        eps_val = (norms == 0).astype(np.float32)
        quat = quat + eps_val[:, None]
        transmitter_rot_mat = R.from_quat(quat)
        transmitter_mouth = transmitter_rot_mat.apply(mouth_offset, inverse=True)
        transmitter_mouth = th.Tensor(transmitter_mouth).view(view.shape[0], -1, 3).transpose(2, 1).contiguous()
        if view.is_cuda:
            transmitter_mouth = transmitter_mouth.cuda()
        return transmitter_mouth

    def _3d_displacements(self, view):
        transmitter_mouth = self._transmitter_mouth(view)
        # offset between tracking markers and ears in the dataset
        left_ear_offset = th.Tensor([0, -0.08, -0.22]).cuda() if view.is_cuda else th.Tensor([0, -0.08, -0.22])
        right_ear_offset = th.Tensor([0, 0.08, -0.22]).cuda() if view.is_cuda else th.Tensor([0, 0.08, -0.22])
        # compute displacements between transmitter mouth and receiver left/right ear
        displacement_left = view[:, 0:3, :] + transmitter_mouth - left_ear_offset[None, :, None]
        displacement_right = view[:, 0:3, :] + transmitter_mouth - right_ear_offset[None, :, None]
        displacement = th.stack([displacement_left, displacement_right], dim=1)
        return displacement

    def _warpfield(self, view, seq_length):
        return self.warper.displacements2warpfield(self._3d_displacements(view), seq_length)

    def forward(self, mono, view):
        '''
        :param mono: input signal as tensor of shape B x 1 x T
        :param view: rx/tx position/orientation as tensor of shape B x 7 x K (K = T / 400)
        :return: warped: warped left/right ear signal as tensor of shape B x 2 x T
        '''
        return self.warper(th.cat([mono, mono], dim=1), self._3d_displacements(view))

geometric_warper = GeometricWarper()
dsp_result = geometric_warper(th.Tensor(mono_audio[None, None, :]), th.Tensor(position_array.transpose(1,0))[None, :, :])

soundfile.write(binaural_fn, dsp_result[0].numpy().transpose(1,0), 48000, 'PCM_16')




