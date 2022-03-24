# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

##########
# world
##########
import numpy as np
import pysptk
import copy
import math

import torch
import torch.nn as nn

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
f0_mel_min = 1127 * np.log(1 + f0_min / 700)
f0_mel_max = 1127 * np.log(1 + f0_max / 700)


def f0_to_coarse(f0):
    f0_mel = 1127 * np.log(1 + f0 / 700)
    # f0_mel[f0_mel == 0] = 0
    # split those greater than 0 to 255 bins
    f0_mel[f0_mel > 0] = (f0_mel[f0_mel > 0] - f0_mel_min) * (f0_bin - 2) / (f0_mel_max - f0_mel_min) + 1

    f0_mel[f0_mel < 0] = 1
    f0_mel[f0_mel > f0_bin - 1] = f0_bin - 1
    f0_coarse = np.rint(f0_mel).astype(np.int)
    # print('Max f0', np.max(f0_coarse), ' ||Min f0', np.min(f0_coarse))
    assert (np.max(f0_coarse) <= 256 and np.min(f0_coarse) >= 0), (f0_coarse.max(), f0_coarse.min())
    return f0_coarse


def f0_to_coarse_torch(f0):
    f0_mel = 1127 * (1 + f0 / 700).log()
    # f0_mel[f0_mel == 0] = 0
    # split those greater than 0 to 255 bins
    f0_mel[f0_mel > 0] = (f0_mel[f0_mel > 0] - f0_mel_min) * (f0_bin - 2) / (f0_mel_max - f0_mel_min) + 1
    f0_mel[f0_mel < 1] = 1
    f0_mel[f0_mel > f0_bin - 1] = f0_bin - 1
    f0_coarse = (f0_mel + 0.5).long()
    #print('Max f0', np.max(f0_coarse), ' ||Min f0', np.min(f0_coarse))
    assert (f0_coarse.max() <= 255 and f0_coarse.min() >= 1), (f0_coarse.max(), f0_coarse.min())
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


def gelu_accurate(x):
    if not hasattr(gelu_accurate, "_a"):
        gelu_accurate._a = math.sqrt(2 / math.pi)
    return 0.5 * x * (1 + torch.tanh(gelu_accurate._a * (x + 0.044715 * torch.pow(x, 3))))


def gelu(x):
    if hasattr(torch.nn.functional, 'gelu'):
        return torch.nn.functional.gelu(x.float()).type_as(x)
    else:
        return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


class GeLU(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        return gelu(x)

class GeLUAcc(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        return gelu_accurate(x)


def build_activation(act_func, inplace=True):
    if act_func == 'relu':
        return nn.ReLU(inplace=inplace)
    elif act_func == 'relu6':
        return nn.ReLU6(inplace=inplace)
    elif act_func == 'gelu':
        return GeLU()
    elif act_func == 'gelu_accurate':
        return GeLUAcc()
    elif act_func == 'tanh':
        return nn.Tanh()
    elif act_func == 'sigmoid':
        return nn.Sigmoid()
    elif act_func is None:
        return None
    else:
        raise ValueError('do not support: %s' % act_func)


def fix_dp_return_type(result, device):
    if isinstance(result, torch.Tensor):
        return result.to(device)
    if isinstance(result, dict):
        return {k:fix_dp_return_type(v, device) for k,v in result.items()}
    if isinstance(result, tuple):
        return tuple([fix_dp_return_type(v, device) for v in result])
    if isinstance(result, list):
        return [fix_dp_return_type(v, device) for v in result]
    # Must be a number then
    return torch.Tensor([result]).to(device)


def move_id2last(arch):
    ID_OPID = 27
    n = len(arch)
    p1 = 0
    while p1 < n and arch[p1] != ID_OPID:
        p1 += 1
    p2 = p1 + 1
    while p2 < n and arch[p2] == ID_OPID:
        p2 += 1
    while p1 < n and p2 < n:
        if arch[p1] == 27:
            arch[p1], arch[p2] = arch[p2], arch[p1]
            while p2 < n and arch[p2] == 27:
                p2 += 1
        p1 += 1            
    return arch


def generate_arch(n, layers, candidate_ops, adjust_id=True):
    num_ops = len(candidate_ops)
    if num_ops ** layers <= n:
        return generate_all_arch(layers, candidate_ops)
    enc_layer = dec_layers = layers // 2
    def _get_arch():
        arch = [candidate_ops[np.random.randint(num_ops)] for _ in range(layers)]
        arch = move_id2last(arch[:enc_layer]) + move_id2last(arch[enc_layer:])
        return arch
    archs = [_get_arch() for i in range(n)]
    return archs


def sample_arch(arch_pool, prob=None):
    N = len(arch_pool)
    indices = [i for i in range(N)]
    if prob is not None:
        prob = np.array(prob, dtype=np.float32)
        prob = prob / prob.sum()
        index = np.random.choice(indices, p=prob)
    else:
        index = np.random.choice(indices)
    arch = arch_pool[index]
    return arch


def generate_all_arch(layers, candidate_ops):
    res = []
    num_ops = len(candidate_ops)
    def dfs(cur_arch, layer_id):
        if layer_id == layers:
            res.append(cur_arch)
            return
        for op in range(num_ops):
            dfs(cur_arch+[candidate_ops[op]], layer_id+1)
    dfs([], 0)
    return res


def convert_to_features(arch, candidate_ops):
    res = []
    num_ops = len(candidate_ops)
    op2id = {candidate_ops[i]:i for i in range(num_ops)}
    for op in arch:
        tmp = [0 for _ in range(num_ops)]
        tmp[op2id[op]] = 1
        res += tmp
    return res
