# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import json
import warnings

import torch
from skimage.transform import resize

from tts_utils.world_utils import f0_to_coarse

warnings.filterwarnings("ignore")

import struct
import webrtcvad
from scipy.ndimage.morphology import binary_dilation
import pyworld as pw

import librosa
import numpy as np
from tts_utils import audio
import pyloudnorm as pyln
from tts_utils.parse_textgrid import remove_empty_lines, TextGrid

from librosa.util import normalize
from scipy.io.wavfile import read
from librosa.filters import mel as librosa_mel_fn

from matplotlib import pyplot as plt

int16_max = (2 ** 15) - 1

def trim_long_silences(path, sr, return_raw_wav=False, norm=True):
    """
    Ensures that segments without voice in the waveform remain no longer than a
    threshold determined by the VAD parameters in params.py.
    :param wav: the raw waveform as a numpy array of floats
    :return: the same waveform with silences trimmed away (length <= original wav length)
    """

    ## Voice Activation Detection
    # Window size of the VAD. Must be either 10, 20 or 30 milliseconds.
    # This sets the granularity of the VAD. Should not need to be changed.
    sampling_rate = 16000
    wav_raw, sr = librosa.core.load(path, sr=sr)

    if norm:
        meter = pyln.Meter(sr)  # create BS.1770 meter
        loudness = meter.integrated_loudness(wav_raw)
        wav_raw = pyln.normalize.loudness(wav_raw, loudness, -20.0)
        if np.abs(wav_raw).max() > 1.0:
            wav_raw = wav_raw / np.abs(wav_raw).max()

    wav = librosa.resample(wav_raw, sr, sampling_rate, res_type='kaiser_best')

    vad_window_length = 30  # In milliseconds
    # Number of frames to average together when performing the moving average smoothing.
    # The larger this value, the larger the VAD variations must be to not get smoothed out.
    vad_moving_average_width = 8
    # Maximum number of consecutive silent frames a segment can have.
    vad_max_silence_length = 12

    # Compute the voice detection window size
    samples_per_window = (vad_window_length * sampling_rate) // 1000

    # Trim the end of the audio to have a multiple of the window size
    wav = wav[:len(wav) - (len(wav) % samples_per_window)]

    # Convert the float waveform to 16-bit mono PCM
    pcm_wave = struct.pack("%dh" % len(wav), *(np.round(wav * int16_max)).astype(np.int16))

    # Perform voice activation detection
    voice_flags = []
    vad = webrtcvad.Vad(mode=3)
    for window_start in range(0, len(wav), samples_per_window):
        window_end = window_start + samples_per_window
        voice_flags.append(vad.is_speech(pcm_wave[window_start * 2:window_end * 2],
                                         sample_rate=sampling_rate))
    voice_flags = np.array(voice_flags)

    # Smooth the voice detection with a moving average
    def moving_average(array, width):
        array_padded = np.concatenate((np.zeros((width - 1) // 2), array, np.zeros(width // 2)))
        ret = np.cumsum(array_padded, dtype=float)
        ret[width:] = ret[width:] - ret[:-width]
        return ret[width - 1:] / width

    audio_mask = moving_average(voice_flags, vad_moving_average_width)
    audio_mask = np.round(audio_mask).astype(np.bool)

    # Dilate the voiced regions
    audio_mask = binary_dilation(audio_mask, np.ones(vad_max_silence_length + 1))
    audio_mask = np.repeat(audio_mask, samples_per_window)
    audio_mask = resize(audio_mask, (len(wav_raw),)) > 0
    if return_raw_wav:
        return wav_raw, audio_mask
    return wav_raw[audio_mask], audio_mask

# hifi-gan-compatible mel processing
# borrowed from HiFi-GAN: https://github.com/jik876/hifi-gan/blob/master/meldataset.py
# MIT License
#
# Copyright (c) 2020 Jungil Kong
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

MAX_WAV_VALUE = 32768.0

def load_wav(full_path):
    sampling_rate, data = read(full_path)
    return data, sampling_rate


def dynamic_range_compression(x, C=1, clip_val=1e-5):
    return np.log(np.clip(x, a_min=clip_val, a_max=None) * C)


def dynamic_range_decompression(x, C=1):
    return np.exp(x) / C


def dynamic_range_compression_torch(x, C=1, clip_val=1e-5):
    return torch.log(torch.clamp(x, min=clip_val) * C)


def dynamic_range_decompression_torch(x, C=1):
    return torch.exp(x) / C


def spectral_normalize_torch(magnitudes):
    output = dynamic_range_compression_torch(magnitudes)
    return output


def spectral_de_normalize_torch(magnitudes):
    output = dynamic_range_decompression_torch(magnitudes)
    return output

mel_basis = {}
hann_window = {}

def mel_spectrogram(y, n_fft, num_mels, sampling_rate, hop_size, win_size, fmin, fmax, center=False):
    if torch.min(y) < -1.:
        print('min value is ', torch.min(y))
    if torch.max(y) > 1.:
        print('max value is ', torch.max(y))

    global mel_basis, hann_window
    if fmax not in mel_basis:
        mel = librosa_mel_fn(sampling_rate, n_fft, num_mels, fmin, fmax)
        mel_basis[str(fmax)+'_'+str(y.device)] = torch.from_numpy(mel).float().to(y.device)
        hann_window[str(y.device)] = torch.hann_window(win_size).to(y.device)

    y = torch.nn.functional.pad(y.unsqueeze(1), (int((n_fft-hop_size)/2), int((n_fft-hop_size)/2)), mode='reflect')
    y = y.squeeze(1)

    # complex tensor as default, then use view_as_real for future pytorch compatibility
    spec = torch.stft(y, n_fft, hop_length=hop_size, win_length=win_size, window=hann_window[str(y.device)],
                      center=center, pad_mode='reflect', normalized=False, onesided=True, return_complex=True)
    spec = torch.view_as_real(spec)
    spec = torch.sqrt(spec.pow(2).sum(-1)+(1e-9))

    spec = torch.matmul(mel_basis[str(fmax)+'_'+str(y.device)], spec)
    spec = spectral_normalize_torch(spec)

    return spec

def process_utterance_hfg(wav_path,
                      fft_size=1024,
                      hop_size=256,
                      win_length=1024,
                      window="hann",
                      num_mels=80,
                      fmin=80,
                      fmax=7600,
                      eps=1e-10,
                      sample_rate=22050,
                      loud_norm=False,
                      min_level_db=-100,
                      return_linear=False,
                      trim_long_sil=False,
                      vocoder='hfg'):

    audio, sampling_rate = load_wav(wav_path)
    audio = audio / MAX_WAV_VALUE
    audio = normalize(audio) * 0.95

    wav = audio

    audio = torch.FloatTensor(audio)
    audio = audio.unsqueeze(0)

    mel = mel_spectrogram(audio, fft_size, num_mels,
                          sample_rate, hop_size, win_length, fmin, fmax,
                          center=False)
    mel = mel[0].numpy()
    assert wav.shape[0] >= mel.shape[1] * hop_size, "size mismatch"
    wav = wav[:mel.shape[1]*hop_size]

    if not return_linear:
        return wav, mel
    else:
        raise NotImplementedError

# pitch calculation
def get_pitch(wav_data, mel, hparams):
    """
    :param wav_data: [T]
    :param mel: [T, 80]
    :param hparams:
    :return:
    """
    _f0, t = pw.dio(wav_data.astype(np.double), hparams['audio_sample_rate'],
                    frame_period=hparams['hop_size'] / hparams['audio_sample_rate'] * 1000)
    f0 = pw.stonemask(wav_data.astype(np.double), _f0, t, hparams['audio_sample_rate'])  # pitch refinement
    delta_l = len(mel) - len(f0)
    assert np.abs(delta_l) <= 2
    if delta_l > 0:
        f0 = np.concatenate([f0] + [f0[-1]] * delta_l)
    f0 = f0[:len(mel)]
    pitch_coarse = f0_to_coarse(f0) + 1
    return f0, pitch_coarse

# mel2ph calculation
def get_mel2ph_p(tg_fn, ph, phone_encoded, mel, hparams):
    ph_list = ph.split(" ")
    with open(tg_fn, "r") as f:
        tg = f.readlines()
    tg = remove_empty_lines(tg)
    tg = TextGrid(tg)
    tg = json.loads(tg.toJson())
    split = np.zeros(len(ph_list) + 1, np.int)
    split[0] = 0
    split[-1] = len(mel)
    tg_idx = 0
    ph_idx = 1
    tg_align = [x for x in tg['tiers'][0]['items']]

    while tg_idx < len(tg_align):
        ph = ph_list[ph_idx]
        x = tg_align[tg_idx]
        if x['text'] == '':
            tg_idx += 1
            continue
        if x['text'] not in ['punc', 'sep']:
            assert x['text'] == ph_list[ph_idx].lower(), (x['text'], ph_list[ph_idx])
        if x['text'] == 'sep':
            assert ph == '|', (ph, '|')
        split[ph_idx] = int(float(x['xmin']) * hparams['audio_sample_rate'] / hparams['hop_size'])
        ph_idx += 1
        tg_idx += 1

    assert tg_idx == len(tg_align), ph_idx == len(ph_list)
    split[ph_idx] = int(float(x['xmax']) * hparams['audio_sample_rate'] / hparams['hop_size'])
    mel2ph = np.zeros([mel.shape[0]], np.int)
    mel2ph_encoded = np.zeros([mel.shape[0]], np.int) - 1
    assert len(ph_list) == len(phone_encoded)
    for ph_idx in range(len(ph_list)):
        mel2ph[split[ph_idx]:split[ph_idx + 1]] = ph_idx + 1
        mel2ph_encoded[split[ph_idx]:split[ph_idx+1]] = phone_encoded[ph_idx]
    assert np.all(mel2ph_encoded != -1)

    mel2ph_torch = torch.from_numpy(mel2ph)
    T_t = len(ph_list)
    dur = mel2ph_torch.new_zeros([T_t + 1]).scatter_add(0, mel2ph_torch, torch.ones_like(mel2ph_torch))
    dur = dur[1:].numpy()
    return mel2ph, mel2ph_encoded, dur