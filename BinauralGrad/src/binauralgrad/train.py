# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from argparse import ArgumentParser
from torch.cuda import device_count
from torch.multiprocessing import spawn

from binauralgrad.learner import train, train_distributed
import binauralgrad.params as params_all


def _get_free_port():
  import socketserver
  with socketserver.TCPServer(('localhost', 0), None) as s:
    return s.server_address[1]


def main(args):
  replica_count = device_count()
  params = getattr(params_all, args.params)
  if replica_count > 1:
    if params.batch_size % replica_count != 0:
      raise ValueError(f'Batch size {params.batch_size} is not evenly divisble by # GPUs {replica_count}.')
    params.batch_size = params.batch_size // replica_count
    port = _get_free_port()
    spawn(train_distributed, args=(replica_count, port, args, params), nprocs=replica_count, join=True)
  else:
    train(args, params)


if __name__ == '__main__':
  parser = ArgumentParser(description='train (or resume training) a BinauralGrad model')
  parser.add_argument('model_dir',
      help='directory in which to store model checkpoints and training logs')
  parser.add_argument('data_dirs', nargs='+',
      help='space separated list of directories from which to read .wav files for training')
  parser.add_argument('--max_steps', default=None, type=int,
      help='maximum number of training steps')
  parser.add_argument('--fp16', action='store_true', default=False,
      help='use 16-bit floating point operations for training')
  parser.add_argument('--binaural-type', default="", type=str,
      help='binaural type')  
  parser.add_argument('--params', default="params", type=str,
    help='param set name')  
  main(parser.parse_args())
