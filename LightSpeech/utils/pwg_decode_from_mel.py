# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import logging
import yaml

import numpy as np
from sklearn.preprocessing import StandardScaler
import torch
from torch import nn

import utils
from parallel_wavegan.models import ParallelWaveGANGenerator
from parallel_wavegan.utils import read_hdf5


def load_pwg_model(config_path, checkpoint_path, stats_path):
    # load config
    with open(config_path) as f:
        config = yaml.load(f, Loader=yaml.Loader)

    # setup
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    model = ParallelWaveGANGenerator(**config["generator_params"])

    ckpt_dict = torch.load(checkpoint_path, map_location="cpu")
    if 'state_dict' not in ckpt_dict:  # official vocoder
        model.load_state_dict(torch.load(checkpoint_path, map_location="cpu")["model"]["generator"])
        scaler = StandardScaler()
        if config["format"] == "hdf5":
            scaler.mean_ = read_hdf5(stats_path, "mean")
            scaler.scale_ = read_hdf5(stats_path, "scale")
        elif config["format"] == "npy":
            scaler.mean_ = np.load(stats_path)[0]
            scaler.scale_ = np.load(stats_path)[1]
        else:
            raise ValueError("support only hdf5 or npy format.")
    else:  # custom PWG vocoder
        fake_task = nn.Module()
        fake_task.model_gen = model
        fake_task.load_state_dict(torch.load(checkpoint_path, map_location="cpu")["state_dict"], strict=False)
        scaler = None

    model.remove_weight_norm()
    model = model.eval().to(device)
    logging.info(f"loaded model parameters from {checkpoint_path}.")
    return model, scaler, config, device


def generate_wavegan(c, model, scaler, config, device, profile=False):
    # start generation
    pad_size = (config["generator_params"]["aux_context_window"],
                config["generator_params"]["aux_context_window"])

    if scaler is not None:
        c = scaler.transform(c)

    with torch.no_grad():
        with utils.Timer('vocoder', print_time=profile):
            # generate each utterance
            z = torch.randn(1, 1, c.shape[0] * config["hop_size"]).to(device)
            c = np.pad(c, (pad_size, (0, 0)), "edge")
            c = torch.FloatTensor(c).unsqueeze(0).transpose(2, 1).to(device)
            y = model(z, c).view(-1)
    return y
