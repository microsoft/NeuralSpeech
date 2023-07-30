# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import math

import torch
import torch.nn as nn
import torch.onnx.operators

from fairseq import utils


class SinusoidalLengthControlPositionalEmbedding(nn.Module):
    """This module produces sinusoidal positional embeddings of any length.

    Padding symbols are ignored, but it is necessary to specify whether padding
    is added on the left side (left_pad=True) or right side (left_pad=False).
    """

    def __init__(self, embedding_dim, padding_idx, left_pad,init_size=1024,quant_N=1):
        super().__init__()
        self.quant_N=quant_N
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        self.left_pad = left_pad
        self.weights = SinusoidalLengthControlPositionalEmbedding.get_embedding(
            init_size,
            embedding_dim,
            padding_idx,
        )
        self.onnx_trace = False
        self.register_buffer('_float_tensor', torch.FloatTensor(1))

    def prepare_for_onnx_export_(self):
        self.onnx_trace = True

    @staticmethod
    def get_embedding(num_embeddings, embedding_dim, padding_idx=None, length=None):
        """Build sinusoidal embeddings.

        This matches the implementation in tensor2tensor, but differs slightly
        from the description in Section 3.5 of "Attention Is All You Need".
        """
        half_dim = embedding_dim // 2
        if length is None:
            #default
            emb = math.log(10000) / (half_dim - 1)
            emb = torch.exp(torch.arange(half_dim, dtype=torch.float) * -emb)
            emb = torch.arange(num_embeddings, dtype=torch.float).unsqueeze(1) * emb.unsqueeze(0)
            emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1).view(num_embeddings, -1)
        else:
            #represent length by sinusoidal pos
            emb = length.float().log() / (half_dim - 1) #batch
            emb = torch.exp(torch.arange(half_dim, dtype=torch.float, device=emb.device).unsqueeze(0) * -emb.unsqueeze(1)) #batch * dim
            wave = torch.arange(num_embeddings, dtype=torch.float, device=emb.device).unsqueeze(0).expand(emb.size(0), num_embeddings)
            emb = wave.unsqueeze(2) * emb.unsqueeze(1) #batch * len * dim
            emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=2).view(emb.size(0), num_embeddings, -1)
        if embedding_dim % 2 == 1:
            # zero pad
            if length is None:
                emb = torch.cat([emb, torch.zeros(num_embeddings, 1)], dim=1)
            else:
                emb = torch.cat([emb, torch.zeros(num_embeddings, 1, device=emb.device)], dim=2)
        if padding_idx is not None:
            if length is None:
                emb[padding_idx, :] = 0
            else:
                emb[:, padding_idx, :] = 0
        return emb

    def forward(self, input,tgt_subwd_lengths, incremental_state=None, length=None, timestep=None, sinpostype=None):
        """Input is expected to be of size [bsz x seqlen]."""
        bsz, seq_len = torch.onnx.operators.shape_as_tensor(input)
        max_pos = self.padding_idx + 1 + seq_len
        if length is not None and sinpostype == 'ratio':
            length4getemb = length
        else:
            length4getemb = None
        if self.weights is None or length4getemb is not None or max_pos > self.weights.size(0):
            # recompute/expand embeddings if needed
            self.weights = SinusoidalLengthControlPositionalEmbedding.get_embedding(
                max_pos,
                self.embedding_dim,
                self.padding_idx,
                length4getemb,
            )
        self.weights = self.weights.type_as(self._float_tensor)


        if incremental_state is not None:
            # positions is the same for every token when decoding a single step
            pos = (timestep.int() + 1).long() if timestep is not None else seq_len
            if length4getemb is None and sinpostype == None:
                if self.onnx_trace:  # False
                    return self.weights[self.padding_idx + pos, :].unsqueeze(1).repeat(bsz, 1, 1)
                return self.weights[self.padding_idx + pos, :].expand(bsz, 1, -1)
            
            elif sinpostype == 'relative':
                pos=tgt_subwd_lengths.to(length.device)
                divpos = pos / length.unsqueeze(1)  # pos/len                
                quant_divpos=self.quantize_func(divpos).clamp(min=0,max=1023)
                return self.weights.index_select(0, quant_divpos.view(-1)).view(bsz, 1, -1)
            elif sinpostype == 'duration':
                durpos=tgt_subwd_lengths.to(length.device).clamp(min=0,max=1023)
                return self.weights.index_select(0, durpos.view(-1)).view(bsz, 1, -1)

            else:
                return self.weights[:, self.padding_idx + pos, :]
        if sinpostype == 'duration' or sinpostype == 'relative' or sinpostype == 'absolute':
            positions = self.make_subwd_positions(tgt_subwd_lengths, self.padding_idx, self.onnx_trace)  # replace by tgt_subwd_length (B, seq_len)
        elif sinpostype == None:
            positions = utils.make_positions(input, self.padding_idx, onnx_trace=self.onnx_trace)
        
        if length4getemb is None and sinpostype == None:
            if self.onnx_trace:
                flat_embeddings = self.weights.detach().index_select(0, positions.view(-1))
                embedding_shape = torch.cat((bsz.view(1), seq_len.view(1), torch.LongTensor([-1])))
                embeddings = torch.onnx.operators.reshape_from_tensor_shape(flat_embeddings, embedding_shape)
                return embeddings
            return self.weights.index_select(0, positions.view(-1)).view(bsz, seq_len, -1).detach()
        elif sinpostype == 'absolute':
            #add 3 to set range value with positions (if no value addition, cause error due to index -1)
            #correspondence to padding_idx (and left_pad?)
            minuspos = (length.view(-1, 1) + 3).expand(bsz, seq_len) - positions.view(bsz, seq_len)  # len-pos
            return self.weights.index_select(0, minuspos.view(-1)).view(bsz, seq_len, -1).detach()
        elif sinpostype == 'relative':
            divpos = positions.view(bsz, seq_len) / ((length.view(-1, 1) + 1).expand(bsz, seq_len))   # pos/len
            quant_divpos=self.quantize_func(divpos)
            return self.weights.index_select(0, quant_divpos.view(-1)).view(bsz, seq_len, -1).detach()
        elif sinpostype == 'duration':
            durpos = positions.view(bsz, seq_len)   # duration pos
            return self.weights.index_select(0, durpos.view(-1)).view(bsz, seq_len, -1).detach()
        else:  # sinpostype == 'ratio'
            return self.weights.index_select(1, positions[0]).view(bsz, seq_len, -1).detach()

    def max_positions(self):
        """Maximum number of supported positions."""
        return int(1e5)  # an arbitrary large number

    def quantize_func(self,divpos):
        quant_N=self.quant_N
        quant_div_pos=(divpos*quant_N).floor().long()
        return quant_div_pos

    def make_subwd_positions(self, input, padding_idx: int, onnx_trace: bool = False): 
        """Replace non-padding symbols with their position numbers.
            rewrite from fairseq.utils make_positions()
        Position numbers begin at padding_idx+1. Padding symbols are ignored.
        """
        # The series of casts and type-conversions here are carefully
        # balanced to both work with ONNX export and XLA. In particular XLA
        # prefers ints, cumsum defaults to output longs, and ONNX doesn't know
        # how to handle the dtype kwarg in cumsum.
        mask = input.ne(padding_idx).int()
        input = input * mask
        return (torch.cumsum(input, dim=1).type_as(mask) * mask).long() + padding_idx # remove + padding_idx