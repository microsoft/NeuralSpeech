# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import argparse
import os

import yaml

hparams = {}


class Args:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            self.__setattr__(k, v)


def set_hparams(use_cmd=True, config='', exp_name='', hparams_str=''):
    if use_cmd:
        parser = argparse.ArgumentParser(description='neural music')
        parser.add_argument('--config', type=str, default='',
                            help='location of the data corpus')
        parser.add_argument('--exp_name', type=str, default='', help='exp_name')
        parser.add_argument('--hparams', type=str, default='',
                            help='location of the data corpus')
        parser.add_argument('--infer', action='store_true', help='infer')
        parser.add_argument('--fast', action='store_true', help='fast_sampling for diffusion')
        parser.add_argument('--fast_iter', type=int, default=12, help='number of fast sampling iter')
        parser.add_argument('--inference_text', type=str, default=None,
                            help="path of text file containing user-given text for TTS inference.")
        parser.add_argument('--validate', action='store_true', help='validate')
        parser.add_argument('--reset', action='store_true', help='reset hparams')
        parser.add_argument('--debug', action='store_true', help='debug')
        args, unknown = parser.parse_known_args()
    else:
        args = Args(config=config, exp_name=exp_name, hparams=hparams_str,
                    infer=False, fast=False, validate=False, reset=False, debug=False)
    args_work_dir = ''
    if args.exp_name != '':
        args.work_dir = args.exp_name
        args_work_dir = f'checkpoints/{args.work_dir}'

    def load_config(config_fn):
        with open(config_fn) as f:
            hparams_ = yaml.safe_load(f)
        if 'base_config' in hparams_:
            ret_hparams = load_config(hparams_['base_config'])
            ret_hparams.update(hparams_)
        else:
            ret_hparams = hparams_
        return ret_hparams

    global hparams
    assert args.config != '' or args_work_dir != ''
    saved_hparams = {}
    if args_work_dir != 'checkpoints/':
        ckpt_config_path = f'{args_work_dir}/config.yaml'
        if os.path.exists(ckpt_config_path):
            try:
                with open(ckpt_config_path) as f:
                    saved_hparams.update(yaml.safe_load(f))
            except:
                pass
        if args.config == '':
            args.config = ckpt_config_path

    hparams.update(load_config(args.config))
    hparams['work_dir'] = args_work_dir

    if not args.reset:
        hparams.update(saved_hparams)

    if args.hparams != "":
        for new_hparam in args.hparams.split(","):
            k, v = new_hparam.split("=")
            if v in ['True', 'False'] or type(hparams[k]) == bool:
                hparams[k] = eval(v)
            else:
                hparams[k] = type(hparams[k])(v)

    if args_work_dir != '' and (not os.path.exists(ckpt_config_path) or args.reset):
        os.makedirs(hparams['work_dir'], exist_ok=True)
        with open(ckpt_config_path, 'w') as f:
            yaml.safe_dump(hparams, f)

    hparams['infer'] = args.infer
    hparams['fast'] = args.fast
    hparams['fast_iter'] = args.fast_iter
    hparams['inference_text'] = args.inference_text
    hparams['debug'] = args.debug
    hparams['validate'] = args.validate
