# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import glob
import logging
import re
import time
from collections import defaultdict
import os
import sys
import shutil
import types
import numpy as np
import torch
import torch.nn.functional as F
import torch.distributed as dist


def reduce_tensors(metrics):
    new_metrics = {}
    for k, v in metrics.items():
        if isinstance(v, torch.Tensor):
            dist.all_reduce(v)
            v = v / dist.get_world_size()
        if type(v) is dict:
            v = reduce_tensors(v)
        new_metrics[k] = v
    return new_metrics


def tensors_to_scalars(metrics):
    new_metrics = {}
    for k, v in metrics.items():
        if isinstance(v, torch.Tensor):
            v = v.item()
        if type(v) is dict:
            v = tensors_to_scalars(v)
        new_metrics[k] = v
    return new_metrics


def move_to_cpu(tensors):
    ret = {}
    for k, v in tensors.items():
        if isinstance(v, torch.Tensor):
            v = v.cpu()
        if type(v) is dict:
            v = move_to_cpu(v)
        ret[k] = v
    return ret



def move_to_cuda(tensor):
    if torch.cuda.is_available():
        return tensor.cuda(non_blocking=True)
    return tensor


def count_parameters(model):
    return np.sum(np.prod(v.size()) for name, v in model.named_parameters())


class AvgrageMeter(object):

    def __init__(self):
        self.reset()

    def reset(self):
        self.avg = 0
        self.sum = 0
        self.cnt = 0

    def update(self, val, n=1):
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt


def collate_1d(values, pad_idx=0, left_pad=False, max_len=None):
    """Convert a list of 1d tensors into a padded 2d tensor."""
    size = max(v.size(0) for v in values) if max_len is None else max_len
    res = values[0].new(len(values), size).fill_(pad_idx)

    def copy_tensor(src, dst):
        assert dst.numel() == src.numel()
        dst.copy_(src)

    for i, v in enumerate(values):
        copy_tensor(v, res[i][size - len(v):] if left_pad else res[i][:len(v)])
    return res


def collate_2d(values, pad_idx=0, left_pad=False, shift_right=False, max_len=None):
    """Convert a list of 2d tensors into a padded 3d tensor."""
    size = max(v.size(0) for v in values) if max_len is None else max_len
    res = values[0].new(len(values), size, values[0].shape[1]).fill_(pad_idx)

    def copy_tensor(src, dst):
        assert dst.numel() == src.numel()
        if shift_right:
            dst[1:] = src[:-1]
        else:
            dst.copy_(src)

    for i, v in enumerate(values):
        copy_tensor(v, res[i][size - len(v):] if left_pad else res[i][:len(v)])
    return res


def _is_batch_full(batch, num_tokens, max_tokens, max_sentences):
    if len(batch) == 0:
        return 0
    if len(batch) == max_sentences:
        return 1
    if num_tokens > max_tokens:
        return 1
    return 0


def batch_by_size(
        indices, num_tokens_fn, max_tokens=None, max_sentences=None,
        required_batch_size_multiple=1, distributed=False
):
    """
    Yield mini-batches of indices bucketed by size. Batches may contain
    sequences of different lengths.

    Args:
        indices (List[int]): ordered list of dataset indices
        num_tokens_fn (callable): function that returns the number of tokens at
            a given index
        max_tokens (int, optional): max number of tokens in each batch
            (default: None).
        max_sentences (int, optional): max number of sentences in each
            batch (default: None).
        required_batch_size_multiple (int, optional): require batch size to
            be a multiple of N (default: 1).
    """
    max_tokens = max_tokens if max_tokens is not None else sys.maxsize
    max_sentences = max_sentences if max_sentences is not None else sys.maxsize
    bsz_mult = required_batch_size_multiple

    if isinstance(indices, types.GeneratorType):
        indices = np.fromiter(indices, dtype=np.int64, count=-1)

    sample_len = 0
    sample_lens = []
    batch = []
    batches = []
    for i in range(len(indices)):
        idx = indices[i]
        num_tokens = num_tokens_fn(idx)
        sample_lens.append(num_tokens)
        sample_len = max(sample_len, num_tokens)

        assert sample_len <= max_tokens, (
            "sentence at index {} of size {} exceeds max_tokens "
            "limit of {}!".format(idx, sample_len, max_tokens)
        )
        num_tokens = (len(batch) + 1) * sample_len

        if _is_batch_full(batch, num_tokens, max_tokens, max_sentences):
            mod_len = max(
                bsz_mult * (len(batch) // bsz_mult),
                len(batch) % bsz_mult,
            )
            batches.append(batch[:mod_len])
            batch = batch[mod_len:]
            sample_lens = sample_lens[mod_len:]
            sample_len = max(sample_lens) if len(sample_lens) > 0 else 0
        batch.append(idx)
    if len(batch) > 0:
        batches.append(batch)
    return batches


def make_positions(tensor, padding_idx):
    """Replace non-padding symbols with their position numbers.

    Position numbers begin at padding_idx+1. Padding symbols are ignored.
    """
    # The series of casts and type-conversions here are carefully
    # balanced to both work with ONNX export and XLA. In particular XLA
    # prefers ints, cumsum defaults to output longs, and ONNX doesn't know
    # how to handle the dtype kwarg in cumsum.
    mask = tensor.ne(padding_idx).int()
    return (
                   torch.cumsum(mask, dim=1).type_as(mask) * mask
           ).long() + padding_idx


def softmax(x, dim):
    return F.softmax(x, dim=dim, dtype=torch.float32)


def sequence_mask(lengths, maxlen, dtype=torch.bool):
    if maxlen is None:
        maxlen = lengths.max()
    mask = ~(torch.ones((len(lengths), maxlen)).to(lengths.device).cumsum(dim=1).t() > lengths).t()
    mask.type(dtype)
    return mask


INCREMENTAL_STATE_INSTANCE_ID = defaultdict(lambda: 0)


def _get_full_incremental_state_key(module_instance, key):
    module_name = module_instance.__class__.__name__

    # assign a unique ID to each module instance, so that incremental state is
    # not shared across module instances
    if not hasattr(module_instance, '_instance_id'):
        INCREMENTAL_STATE_INSTANCE_ID[module_name] += 1
        module_instance._instance_id = INCREMENTAL_STATE_INSTANCE_ID[module_name]

    return '{}.{}.{}'.format(module_name, module_instance._instance_id, key)


def get_incremental_state(module, incremental_state, key):
    """Helper for getting incremental state for an nn.Module."""
    full_key = _get_full_incremental_state_key(module, key)
    if incremental_state is None or full_key not in incremental_state:
        return None
    return incremental_state[full_key]


def set_incremental_state(module, incremental_state, key, value):
    """Helper for setting incremental state for an nn.Module."""
    if incremental_state is not None:
        full_key = _get_full_incremental_state_key(module, key)
        incremental_state[full_key] = value


def fill_with_neg_inf(t):
    """FP16-compatible function that fills a tensor with -inf."""
    return t.float().fill_(float('-inf')).type_as(t)


def fill_with_neg_inf2(t):
    """FP16-compatible function that fills a tensor with -inf."""
    return t.float().fill_(-1e9).type_as(t)


def save_checkpoint(state, is_best, save):
    filename = os.path.join(save, 'checkpoint.pth.tar')
    torch.save(state, filename)
    if is_best:
        best_filename = os.path.join(save, 'model_best.pth.tar')
        shutil.copyfile(filename, best_filename)


def get_all_ckpts(checkpoint_name):
    all_ckpts = glob.glob(checkpoint_name)
    all_ckpts = [x for x in all_ckpts if len(re.findall(r'.*/checkpoint(\d+).pt', x)) > 0]
    return sorted(all_ckpts, key=lambda x: -int(re.findall(r'.*/checkpoint(\d+).pt', x)[0]))


def save(model_path, model, epoch, step, optimizer, best_valid_loss=None, is_best=True):
    if isinstance(optimizer, dict):
        optimizer_states = {k: x.state_dict() for k, x in optimizer.items()}
    else:
        optimizer_states = optimizer.state_dict()
    if isinstance(model, dict):
        model_states = {k: (x.state_dict() if not hasattr(x, 'module') else x.module.state_dict())
                        for k, x in model.items()}
    else:
        model_states = model.state_dict() if not hasattr(model, 'module') else model.module.state_dict()
    state_dict = {
        'model': model_states,
        'optimizer': optimizer_states,
        'epoch': epoch,
        'step': step,
        'best_valid_loss': best_valid_loss,
    }
    filename = os.path.join(model_path, 'checkpoint{}.pt'.format(epoch))
    all_ckpts = get_all_ckpts(os.path.join(model_path, 'checkpoint*.pt'))
    for c in all_ckpts[5:]:
        logging.info(f"Remove ckpt: {c}")
        os.remove(c)

    torch.save(state_dict, filename)
    newest_filename = os.path.join(model_path, 'checkpoint_latest.pt')
    shutil.copyfile(filename, newest_filename)
    logging.info(f'Save ckpt: {filename}.')

    if is_best:
        best_filename = os.path.join(model_path, 'checkpoint_best.pt')
        shutil.copyfile(filename, best_filename)
        logging.info(f'Find best ckpt.')


def load(model_path):
    if os.path.isdir(model_path):
        newest_filename = os.path.join(model_path, 'checkpoint_latest.pt')
    else:
        assert os.path.isfile(model_path), model_path
        newest_filename = model_path
    if not os.path.exists(newest_filename):
        return None, 0, 0, None, float('inf')
    state_dict = torch.load(newest_filename, map_location="cpu")
    model_state_dict = state_dict['model']
    epoch = state_dict['epoch']
    step = state_dict['step']
    optimizer_state_dict = state_dict['optimizer']
    best_valid_loss = state_dict['best_valid_loss'] if state_dict['best_valid_loss'] is not None else float('inf')
    return model_state_dict, epoch, step, optimizer_state_dict, best_valid_loss


def create_exp_dir(path, scripts_to_save=None):
    if not os.path.exists(path):
        os.makedirs(path)
        print('Experiment dir : {}'.format(path))

    if scripts_to_save is not None:
        os.makedirs(os.path.join(path, 'scripts'), exist_ok=True)
        for script in scripts_to_save:
            dst_file = os.path.join(path, 'scripts', os.path.basename(script))
            shutil.copyfile(script, dst_file)


def debug_log(fname, *args):
    with open(fname, 'a+') as f:
        for c in args:
            f.write('{}\n'.format(c))


def unpack_dict_to_list(samples):
    samples_ = []
    bsz = samples.get('outputs').size(0)
    for i in range(bsz):
        res = {}
        for k, v in samples.items():
            try:
                res[k] = v[i]
            except:
                pass
        samples_.append(res)
    return samples_


def get_focus_rate(attn, src_padding_mask=None, tgt_padding_mask=None):
    ''' 
    attn: bs x L_t x L_s
    '''
    if src_padding_mask is not None:
        attn = attn * (1 - src_padding_mask.float())[:, None, :]

    if tgt_padding_mask is not None:
        attn = attn * (1 - tgt_padding_mask.float())[:, :, None]

    focus_rate = attn.max(-1).values.sum(-1)
    focus_rate = focus_rate / attn.sum(-1).sum(-1)
    return focus_rate


def get_word_coverage_rate(attn, src_padding_mask=None, src_seg_mask=None, tgt_padding_mask=None):
    ''' 
    attn: bs x L_t x L_s
    '''

    return


def get_phone_coverage_rate(attn, src_padding_mask=None, src_seg_mask=None, tgt_padding_mask=None):
    ''' 
    attn: bs x L_t x L_s
    '''
    src_mask = attn.new(attn.size(0), attn.size(-1)).bool().fill_(False)
    if src_padding_mask is not None:
        src_mask |= src_padding_mask
    if src_seg_mask is not None:
        src_mask |= src_seg_mask

    attn = attn * (1 - src_mask.float())[:, None, :]
    if tgt_padding_mask is not None:
        attn = attn * (1 - tgt_padding_mask.float())[:, :, None]

    phone_coverage_rate = attn.max(1).values.sum(-1)
    # phone_coverage_rate = phone_coverage_rate / attn.sum(-1).sum(-1)
    phone_coverage_rate = phone_coverage_rate / (1 - src_mask.float()).sum(-1)
    return phone_coverage_rate


def get_diagonal_focus_rate(attn, attn_ks, target_len, src_padding_mask=None, tgt_padding_mask=None,
                            band_mask_factor=5, band_width=50):
    ''' 
    attn: bx x L_t x L_s
    attn_ks: shape: tensor with shape [batch_size], input_lens/output_lens
    
    diagonal: y=k*x (k=attn_ks, x:output, y:input)
    1 0 0
    0 1 0
    0 0 1
    y>=k*(x-width) and y<=k*(x+width):1
    else:0
    '''
    # width = min(target_len/band_mask_factor, 50)
    width1 = target_len / band_mask_factor
    width2 = target_len.new(target_len.size()).fill_(band_width)
    width = torch.where(width1 < width2, width1, width2).float()
    base = torch.ones(attn.size()).to(attn.device)
    zero = torch.zeros(attn.size()).to(attn.device)
    x = torch.arange(0, attn.size(1)).to(attn.device)[None, :, None].float() * base
    y = torch.arange(0, attn.size(2)).to(attn.device)[None, None, :].float() * base
    cond = (y - attn_ks[:, None, None] * x)
    cond1 = cond + attn_ks[:, None, None] * width[:, None, None]
    cond2 = cond - attn_ks[:, None, None] * width[:, None, None]
    mask1 = torch.where(cond1 < 0, zero, base)
    mask2 = torch.where(cond2 > 0, zero, base)
    mask = mask1 * mask2

    if src_padding_mask is not None:
        attn = attn * (1 - src_padding_mask.float())[:, None, :]
    if tgt_padding_mask is not None:
        attn = attn * (1 - tgt_padding_mask.float())[:, :, None]

    diagonal_attn = attn * mask
    diagonal_focus_rate = diagonal_attn.sum(-1).sum(-1) / attn.sum(-1).sum(-1)
    return diagonal_focus_rate, mask


def generate_arch(n, layers, num_ops=10):
    def _get_arch():
        arch = [np.random.randint(1, num_ops + 1) for _ in range(layers)]
        return arch

    archs = [_get_arch() for i in range(n)]
    return archs


def parse_arch_to_seq(arch):
    seq = [op for op in arch]
    return seq


def parse_seq_to_arch(seq):
    arch = [idx for idx in seq]
    return arch


def pairwise_accuracy(la, lb):
    n = len(la)
    assert n == len(lb)
    total = 0
    count = 0
    for i in range(n):
        for j in range(i + 1, n):
            if la[i] >= la[j] and lb[i] >= lb[j]:
                count += 1
            if la[i] < la[j] and lb[i] < lb[j]:
                count += 1
            total += 1
    return float(count) / total


def hamming_distance(la, lb):
    N = len(la)
    assert N == len(lb)

    def _hamming_distance(s1, s2):
        n = len(s1)
        assert n == len(s2)
        c = 0
        for i, j in zip(s1, s2):
            if i != j:
                c += 1
        return c

    dis = 0
    for i in range(N):
        line1 = la[i]
        line2 = lb[i]
        dis += _hamming_distance(line1, line2)
    return dis / N


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


def select_attn(attn_logits, type='best'):
    """

    :param attn_logits: [n_layers, B, n_head, T_sp, T_txt]
    :return:
    """
    encdec_attn = torch.stack(attn_logits, 0).transpose(1, 2)
    # [n_layers * n_head, B, T_sp, T_txt]
    encdec_attn = (encdec_attn.reshape([-1, *encdec_attn.shape[2:]])).softmax(-1)
    if type == 'best':
        indices = encdec_attn.max(-1).values.sum(-1).argmax(0)
        encdec_attn = encdec_attn.gather(
            0, indices[None, :, None, None].repeat(1, 1, encdec_attn.size(-2), encdec_attn.size(-1)))[0]
        return encdec_attn
    elif type == 'mean':
        return encdec_attn.mean(0)



def get_num_heads(arch):
    num_heads = []
    for i in range(len(arch)):
        op = arch[i]
        if op <= 7 or op == 11:
            num_heads.append(1)
        elif op == 8:
            num_heads.append(2)
        elif op == 9:
            num_heads.append(4)
        elif op == 10:
            num_heads.append(8)
    return num_heads


def remove_padding(x, padding_idx=0):
    if x is None:
        return None
    assert len(x.shape) in [1, 2]
    if len(x.shape) == 2:  # [T, H]
        return x[np.abs(x).sum(-1) != padding_idx]
    elif len(x.shape) == 1:  # [T]
        return x[x != padding_idx]


class Timer:
    timer_map = {}

    def __init__(self, name, print_time=False):
        if name not in Timer.timer_map:
            Timer.timer_map[name] = 0
        self.name = name
        self.print_time = print_time

    def __enter__(self):
        self.t = time.time()

    def __exit__(self, exc_type, exc_val, exc_tb):
        Timer.timer_map[self.name] += time.time() - self.t
        if self.print_time:
            print(self.name, Timer.timer_map[self.name])
