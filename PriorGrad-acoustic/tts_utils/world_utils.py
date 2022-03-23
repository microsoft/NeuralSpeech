# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

##########
# world
##########
import numpy as np
import pysptk
import copy

import torch

gamma = 0
mcepInput = 3  # 0 for dB, 3 for magnitude
alpha = 0.45
en_floor = 10 ** (-80 / 20)
FFT_SIZE = 2048


def code_harmonic(sp, order):
    # get mcep
    mceps = np.apply_along_axis(pysptk.mcep, 1, sp, order - 1, alpha, itype=mcepInput, threshold=en_floor)

    # do fft and take real
    scale_mceps = copy.copy(mceps)
    scale_mceps[:, 0] *= 2
    scale_mceps[:, -1] *= 2
    mirror = np.hstack([scale_mceps[:, :-1], scale_mceps[:, -1:0:-1]])
    mfsc = np.fft.rfft(mirror).real

    return mfsc


def decode_harmonic(mfsc, fftlen=FFT_SIZE):
    # get mcep back
    mceps_mirror = np.fft.irfft(mfsc)
    mceps_back = mceps_mirror[:, :60]
    mceps_back[:, 0] /= 2
    mceps_back[:, -1] /= 2

    # get sp
    spSm = np.exp(np.apply_along_axis(pysptk.mgc2sp, 1, mceps_back, alpha, gamma, fftlen=fftlen).real)

    return spSm


f0_bin = 256
f0_max = 1100.0
f0_min = 50.0


def f0_to_coarse(f0):
    f0_mel = 1127 * np.log(1 + f0 / 700)
    f0_mel_min = 1127 * np.log(1 + f0_min / 700)
    f0_mel_max = 1127 * np.log(1 + f0_max / 700)
    # f0_mel[f0_mel == 0] = 0
    # 大于0的分为255个箱
    f0_mel[f0_mel > 0] = (f0_mel[f0_mel > 0] - f0_mel_min) * (f0_bin - 2) / (f0_mel_max - f0_mel_min) + 1

    f0_mel[f0_mel < 0] = 1
    f0_mel[f0_mel > f0_bin - 1] = f0_bin - 1
    f0_coarse = np.rint(f0_mel).astype(np.int)
    # print('Max f0', np.max(f0_coarse), ' ||Min f0', np.min(f0_coarse))
    assert (np.max(f0_coarse) <= 256 and np.min(f0_coarse) >= 0)
    return f0_coarse


def f0_to_coarse_torch(f0):
    f0_mel = 1127 * (1 + f0 / 700).log()
    f0_mel_min = 1127 * np.log(1 + f0_min / 700)
    f0_mel_max = 1127 * np.log(1 + f0_max / 700)
    # f0_mel[f0_mel == 0] = 0
    # 大于0的分为255个箱
    f0_mel[f0_mel > 0] = (f0_mel[f0_mel > 0] - f0_mel_min) * (f0_bin - 2) / (f0_mel_max - f0_mel_min) + 1

    f0_mel[f0_mel < 0] = 1
    f0_mel[f0_mel > f0_bin - 1] = f0_bin - 1
    f0_coarse = f0_mel.long()
    # print('Max f0', np.max(f0_coarse), ' ||Min f0', np.min(f0_coarse))
    assert (f0_coarse.max() <= 256 and f0_coarse.min() >= 0)
    return f0_coarse


def process_f0(f0, hparams):
    f0_ = (f0 - hparams['f0_mean']) / hparams['f0_std']
    f0_[f0 == 0] = np.interp(np.where(f0 == 0)[0], np.where(f0 > 0)[0], f0_[f0 > 0])
    uv = (torch.FloatTensor(f0) == 0).float()
    f0 = f0_
    f0 = torch.FloatTensor(f0)
    return f0, uv


def restore_pitch(pitch, uv, hparams, pitch_padding=None, min=None, max=None):
    if pitch_padding is None:
        pitch_padding = pitch == -200
    pitch = pitch * hparams['f0_std'] + hparams['f0_mean']
    if min is not None:
        pitch = pitch.clamp(min=min)
    if max is not None:
        pitch = pitch.clamp(max=max)
    if uv is not None:
        pitch[uv > 0] = 1
    pitch[pitch_padding] = 0
    return pitch
