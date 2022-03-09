# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import librosa
import librosa.filters
import numpy as np
from scipy import signal
from scipy.io import wavfile


def save_wav(wav, path, sr, norm=False):
    if norm:
        wav = wav / np.abs(wav).max()
    wav *= 32767
    # proposed by @dsmiller
    wavfile.write(path, sr, wav.astype(np.int16))


def get_hop_size(hparams):
    hop_size = hparams['hop_size']
    if hop_size is None:
        assert hparams['frame_shift_ms'] is not None
        hop_size = int(hparams['frame_shift_ms'] / 1000 * hparams['audio_sample_rate'])
    return hop_size


###########################################################################################
def griffin_lim(S, hparams):
    angles = np.exp(2j * np.pi * np.random.rand(*S.shape))
    S_complex = np.abs(S).astype(np.complex)
    y = _istft(S_complex * angles, hparams)
    for i in range(hparams['griffin_lim_iters']):
        angles = np.exp(1j * np.angle(_stft(y, hparams)))
        y = _istft(S_complex * angles, hparams)
    return y


def preemphasis(wav, k, preemphasize=True):
    if preemphasize:
        return signal.lfilter([1, -k], [1], wav)
    return wav


def inv_preemphasis(wav, k, inv_preemphasize=True):
    if inv_preemphasize:
        return signal.lfilter([1], [1, -k], wav)
    return wav


def _stft(y, hparams):
    return librosa.stft(y=y, n_fft=hparams['n_fft'], hop_length=get_hop_size(hparams),
                        win_length=hparams['win_size'], pad_mode='constant')


def _istft(y, hparams):
    return librosa.istft(y, hop_length=get_hop_size(hparams), win_length=hparams['win_size'])


##########################################################
# Those are only correct when using lws!!! (This was messing with Wavenet quality for a long time!)
def num_frames(length, fsize, fshift):
    """Compute number of time frames of spectrogram
        """
    pad = (fsize - fshift)
    if length % fshift == 0:
        M = (length + pad * 2 - fsize) // fshift + 1
    else:
        M = (length + pad * 2 - fsize) // fshift + 2
    return M


def pad_lr(x, fsize, fshift):
    """Compute left and right padding
    """
    M = num_frames(len(x), fsize, fshift)
    pad = (fsize - fshift)
    T = len(x) + 2 * pad
    r = (M - 1) * fshift + fsize - T
    return pad, pad + r


##########################################################

def librosa_pad_lr(x, fsize, fshift, pad_sides=1):
    '''compute right padding (final frame) or both sides padding (first and final frames)
    '''
    assert pad_sides in (1, 2)
    # return int(fsize // 2)
    pad = (x.shape[0] // fshift + 1) * fshift - x.shape[0]
    if pad_sides == 1:
        return 0, pad
    else:
        return pad // 2, pad // 2 + pad % 2


# Conversions
_mel_basis = None
_inv_mel_basis = None


def _linear_to_mel(spectogram, hparams):
    global _mel_basis
    if _mel_basis is None:
        _mel_basis = _build_mel_basis(hparams)
    return np.dot(_mel_basis, spectogram)


def _mel_to_linear(mel_spectrogram, hparams):
    global _inv_mel_basis
    if _inv_mel_basis is None:
        _inv_mel_basis = np.linalg.pinv(_build_mel_basis(hparams))
    return np.maximum(1e-10, np.dot(_inv_mel_basis, mel_spectrogram))


def _build_mel_basis(hparams):
    assert hparams['fmax'] <= hparams['audio_sample_rate'] // 2
    return librosa.filters.mel(hparams['audio_sample_rate'], hparams['n_fft'], n_mels=hparams['audio_num_mel_bins'],
                               fmin=hparams['fmin'], fmax=hparams['fmax'])


def amp_to_db(x):
    return 20 * np.log10(np.maximum(1e-5, x))


def db_to_amp(x):
    return np.power(10.0, (x) * 0.05)


def normalize(S, hparams):
    return (S - hparams['min_level_db']) / -hparams['min_level_db']


def denormalize(D, hparams):
    return (D * -hparams['min_level_db']) + hparams['min_level_db']


def plot_spec(spec, path, info=None):
    fig = plt.figure(figsize=(14, 7))
    heatmap = plt.pcolor(spec)
    fig.colorbar(heatmap)

    xlabel = 'Time'
    if info is not None:
        xlabel += '\n\n' + info
    plt.xlabel(xlabel)
    plt.ylabel('Mel filterbank')
    plt.tight_layout()
    plt.savefig(path, format='png')
    plt.close(fig)


def plot_curve(x, path, ymin=None, ymax=None):
    fig = plt.figure(figsize=(14, 7))
    if isinstance(x, list):
        for x_ in x:
            plt.plot(x_)
    else:
        plt.plot(x)
    if ymin is not None:
        plt.ylim(ymin, ymax)
    plt.tight_layout()
    plt.savefig(path, format='png')
    plt.close(fig)


# Compute the mel scale spectrogram from the wav
def dynamic_range_compression(x, C=1, clip_val=1e-5):
    """
    PARAMS
    ------
    C: compression factor
    """
    return np.log(np.clip(x, a_min=clip_val, a_max=None) * C)


def dynamic_range_decompression(x, C=1):
    """
    PARAMS
    ------
    C: compression factor used to compress
    """
    return np.exp(x) / C
