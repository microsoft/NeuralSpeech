# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from fairseq import utils
from torch import Tensor


class LearnedLengthControlPositionalEmbedding(nn.Embedding):
    """
    This module learns positional embeddings up to a fixed maximum size.
    Padding ids are ignored by either offsetting based on padding_idx
    or by setting padding_idx to None and ensuring that the appropriate
    position ids are passed to the forward function.
    """

    def __init__(self, num_embeddings: int, embedding_dim: int, padding_idx: int):
        super().__init__(num_embeddings, embedding_dim, padding_idx)
        self.onnx_trace = False
        self.quant_N = 5
        if self.padding_idx is not None:
            self.max_positions = self.num_embeddings - self.padding_idx - 1
        else:
            self.max_positions = self.num_embeddings

    def forward(
        self,
        input: Tensor,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        length = None,
        positions: Optional[Tensor] = None,
        postype=None
    ):
        """Input is expected to be of size [bsz x seqlen]."""
        bsz, seq_len = torch.onnx.operators.shape_as_tensor(input)
      
        if positions is None:
            if incremental_state is not None:
                # positions is the same for every token when decoding a single step
                # Without the int() cast, it doesn't work in some cases when exporting to ONNX
                positions = torch.zeros(
                    (1, 1), device=input.device, dtype=input.dtype
                ).fill_(int(self.padding_idx + input.size(1)))
            else:
                positions = utils.make_positions(
                    input, self.padding_idx, onnx_trace=self.onnx_trace
                )
        elif positions is not None:
            if incremental_state is not None:
                if postype == 'relative':
                    pos=positions.to(length.device)
                    divpos = pos / length.unsqueeze(1)  # pos/len 
                    positions=self.quantize_func(divpos)

                elif postype == 'duration':
                    positions=positions
            else:
                positions = self.make_subwd_positions(positions, self.padding_idx, self.onnx_trace)  # replace by tgt_subwd_length (B, seq_len)
                if length is not None and postype=="relative":
                    positions = positions.view(bsz, seq_len) / ((length.view(-1, 1) + 1).expand(bsz, seq_len))   # pos/len
                    positions=self.quantize_func(positions)
                elif length is None and postype=="duration":
                    positions = positions.view(bsz, seq_len)   # duration pos
            positions=positions.clamp(min=0,max=1499)

        return F.embedding(
            positions,
            self.weight,
            self.padding_idx,
            self.max_norm,
            self.norm_type,
            self.scale_grad_by_freq,
            self.sparse,
        )
    
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
