# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import matplotlib.pyplot as plt
import numpy as np

def spec_numpy_to_figure(spec):
    fig = plt.figure(figsize=(8, 4))
    plt.imshow(spec.T, aspect='auto', origin='lower')
    return fig


def numpy_to_figure(numpy_data):
    fig = plt.figure()
    pos = plt.imshow(numpy_data, aspect='auto', origin='lower')
    fig.colorbar(pos)
    return fig


def plot_to_figure(numpy_data):
    fig = plt.figure()
    if isinstance(numpy_data, dict):
        for k, d in numpy_data.items():
            plt.plot(d, label=k)
    else:
        plt.plot(numpy_data, aspect='auto', origin='lower')
    plt.legend()
    return fig


def weight_to_figure(w, mask=None, plots=None):
    '''
    author @zcguan
    :param w: 2D weight matrix
    :return: pyplot figure.
    '''
    fig = plt.figure(figsize=(8, 6))
    plt.imshow(w, aspect='auto', origin='lower')

    if mask is not None:
        import copy
        my_cmap = copy.copy(plt.cm.get_cmap('Reds'))
        my_cmap.set_bad(alpha=0)
        mask = mask.copy()
        mask[mask < 0.5] = np.nan
        plt.imshow(mask, aspect='auto', origin='lower',
                   interpolation='none', alpha=0.5, cmap=my_cmap)
    if plots is not None:
        for p in plots:
            plt.plot(p)
    plt.tight_layout()
    return fig
