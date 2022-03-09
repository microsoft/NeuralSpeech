# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import json
import warnings
import struct

import webrtcvad
from skimage.transform import resize
from scipy.ndimage.morphology import binary_dilation
import pyworld as pw
import numpy as np
import torch
import librosa
import pyloudnorm as pyln

from utils import audio
from utils.world_utils import f0_to_coarse
from utils.parse_textgrid import remove_empty_lines, TextGrid

warnings.filterwarnings("ignore")

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


def process_utterance(wav_path,
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
                      trim_long_sil=False, vocoder='pwg'):
    if isinstance(wav_path, str):
        if trim_long_sil:
            wav, _ = trim_long_silences(wav_path, sample_rate)
        else:
            wav, _ = librosa.core.load(wav_path, sr=sample_rate)
    else:
        wav = wav_path

    if loud_norm:
        meter = pyln.Meter(sample_rate)  # create BS.1770 meter
        loudness = meter.integrated_loudness(wav)
        wav = pyln.normalize.loudness(wav, loudness, -22.0)
        if np.abs(wav).max() > 1:
            wav = wav / np.abs(wav).max()

    # get amplitude spectrogram
    x_stft = librosa.stft(wav, n_fft=fft_size, hop_length=hop_size,
                          win_length=win_length, window=window, pad_mode="constant")
    spc = np.abs(x_stft)  # (n_bins, T)

    # get mel basis
    fmin = 0 if fmin is -1 else fmin
    fmax = sample_rate / 2 if fmax is -1 else fmax
    mel_basis = librosa.filters.mel(sample_rate, fft_size, num_mels, fmin, fmax)
    mel = mel_basis @ spc

    if vocoder == 'pwg':
        mel = np.log10(np.maximum(eps, mel))  # (n_mel_bins, T)
    else:
        assert False, f'"{vocoder}" is not in ["pwg"].'

    l_pad, r_pad = audio.librosa_pad_lr(wav, fft_size, hop_size, 1)
    wav = np.pad(wav, (l_pad, r_pad), mode='constant', constant_values=0.0)
    wav = wav[:mel.shape[1] * hop_size]

    if not return_linear:
        return wav, mel
    else:
        spc = audio.amp_to_db(spc)
        spc = audio.normalize(spc, {'min_level_db': min_level_db})
        return wav, mel, spc


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


def get_mel2ph(tg_fn, ph, mel, hparams):
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
        # if ph != '|' and x['text'] == '':
        #     tg_idx += 1
        #     continue
        # elif ph == '|' and x['text'] != '':
        #     split[ph_idx] = split[ph_idx - 1]
        #     ph_idx += 1
        # else:  # ph == '|' and x['text'] == '' or ph != '|' and x['text'] != ''
        #     if x['text'] not in ['punc', '']:
        #         assert x['text'] == ph_list[ph_idx].lower(), (x['text'], ph_list[ph_idx])
        #     if x['text'] == '':
        #         assert ph == '|', (ph, '|')
        #     split[ph_idx] = int(float(x['xmin']) * hparams['audio_sample_rate'] / hparams['hop_size'])
        #     ph_idx += 1
        #     tg_idx += 1

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
    for ph_idx in range(len(ph_list)):
        mel2ph[split[ph_idx]:split[ph_idx + 1]] = ph_idx + 1

    mel2ph_torch = torch.from_numpy(mel2ph)
    T_t = len(ph_list)
    dur = mel2ph_torch.new_zeros([T_t + 1]).scatter_add(0, mel2ph_torch, torch.ones_like(mel2ph_torch))
    dur = dur[1:].numpy()
    return mel2ph, dur
