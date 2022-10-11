# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import argparse
import numpy as np
import torch as th
import torchaudio as ta

from src.binauralgrad.losses import L2Loss, AmplitudeLoss, PhaseLoss
import auraloss
import speechmetrics

import sys

result_folder = sys.argv[1]
ref_folder = sys.argv[2]


window_length = 5 # seconds
scoresmetrics = speechmetrics.load(['pesq', 'stoi'], window_length)
mrstft_function = auraloss.freq.MultiResolutionSTFTLoss()


def compute_metrics(binauralized, reference, path_pt, path_gt):
    '''
    compute l2 error, amplitude error, and angular phase error for the given binaural and reference singal
    :param binauralized: 2 x T tensor containing predicted binaural signal
    :param reference: 2 x T tensor containing reference binaural signal
    :return: errors as a scalar value for each metric and the number of samples in the sequence
    '''
    binauralized, reference = binauralized.unsqueeze(0), reference.unsqueeze(0)

    # compute error metrics
    l2_error = L2Loss()(binauralized, reference)
    amplitude_error = AmplitudeLoss(sample_rate=48000)(binauralized, reference)
    phase_error = PhaseLoss(sample_rate=48000, ignore_below=0.2)(binauralized, reference)
    mrstft_error = mrstft_function(binauralized, reference)

    scores = scoresmetrics(path_pt, path_gt)
    pesq = scores['pesq'] if len(scores['pesq']) == 1 else scores['pesq'][0]

    return{
        "l2": l2_error,
        "amplitude": amplitude_error,
        "phase": phase_error,
        "mrstft": mrstft_error,
        "pesq_score": pesq,
        "samples": binauralized.shape[-1]

    }


# binauralized and evaluate test sequence for the eight subjects and the validation sequence
test_sequences = [f"subject{i+1}" for i in range(8)] + ["validation_sequence"]

errors = []
for test_sequence in test_sequences:
    print(f"Cal {test_sequence}...")

    # binauralize and save output
    binaural, _ = ta.load(f"{result_folder}/{test_sequence}.wav")

    # compute error metrics
    reference, sr = ta.load(f"{ref_folder}/{test_sequence}/binaural.wav")
    errors.append(compute_metrics(binaural, reference, f"{result_folder}/{test_sequence}.wav", f"{ref_folder}/{test_sequence}/binaural.wav"))
    print(errors[-1])


# accumulate errors
sequence_weights = np.array([err["samples"] for err in errors])
sequence_weights = sequence_weights / np.sum(sequence_weights)
l2_error = sum([err["l2"] * sequence_weights[i] for i, err in enumerate(errors)])
amplitude_error = sum([err["amplitude"] * sequence_weights[i] for i, err in enumerate(errors)])
phase_error = sum([err["phase"] * sequence_weights[i] for i, err in enumerate(errors)])
mrstft_error = sum([err["mrstft"] * sequence_weights[i] for i, err in enumerate(errors)])
pesq = sum([err["pesq_score"] * sequence_weights[i] for i, err in enumerate(errors)])

# print accumulated errors on testset
print(f"l2 (x10^3):     {l2_error * 1000:.3f}")
print(f"amplitude:      {amplitude_error:.3f}")
print(f"phase:          {phase_error:.3f}")
print(f"mrstft:          {mrstft_error:.3f}")
print(f"pesq:           {pesq:.3f}")
