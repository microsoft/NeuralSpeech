# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from einops import rearrange, repeat #, reduce, pack, unpack

from module import SinusoidalPositionalEmbedding, TransformerEncoderLayer, LayerNorm
from utils import AttrDict, SinusoidalPosEmb, Mish

DEFAULT_MAX_SOURCE_POSITIONS=2048

class TransformerEncoder(nn.Module):
    def __init__(self, num_layers, hidden_size, last_ln=True, use_cln=False):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        embed_dim = self.hidden_size
        self.padding_idx = 0
        self.pos_embed_alpha = nn.Parameter(torch.Tensor([1]))
        self.dropout = 0.2
        self.embed_scale = math.sqrt(embed_dim)
        self.max_source_positions = DEFAULT_MAX_SOURCE_POSITIONS
        self.embed_positions = SinusoidalPositionalEmbedding(
            embed_dim, self.padding_idx,
            init_size=self.max_source_positions + self.padding_idx + 1,
        )
        self.layers = nn.ModuleList([])
        self.layers.extend([
            TransformerEncoderLayer(i, self.hidden_size, self.dropout, use_cln=use_cln)
            for i in range(self.num_layers)
        ])
        self.last_ln = last_ln
        if last_ln:
            self.layer_norm = LayerNorm(embed_dim, use_cln=use_cln)

    def forward(self, text_vector, style_vector, learned_queries, diffusion_step):
        """
        :param text_vector: [B, num_tquery, C]
        :param (noised) style_vector: [B, num_lquery, C]
        :param learned_queries: [B, num_lquery, C]
        :param diffusion_step: [B, 1, C]
        :return: predicted results (noise or x0)
        """
        x = torch.cat([text_vector, diffusion_step, style_vector, learned_queries], dim=-2)
        encoder_padding_mask = torch.zeros_like(x[..., 0]).bool()  # all non-mask
        x = x.transpose(0, 1) # B x T x C -> T x B x C

        # encoder layers
        for layer in self.layers:
            x = layer(x, encoder_padding_mask=encoder_padding_mask, condition=diffusion_step.squeeze(1))
            
        if self.last_ln:
            x = self.layer_norm(x, condition=diffusion_step.squeeze(1))
            x = x * (1 - encoder_padding_mask.to(x.dtype)).transpose(0, 1)[..., None]

        return x

class TransformerEstimator(nn.Module):
    def __init__(self, num_layers, hidden_size, query_tokennum):
        # in our case the tokennum of TTS control tokens is the same as that of prompt tokens
        # and all representations (TTS control, text prompt) has the same dimension
        # but these can be different.
        super().__init__()
        self.decoder = TransformerEncoder(hidden_size=hidden_size, num_layers=num_layers)
        
        self.params = params = AttrDict(
            # Model params
            transformer_hidden=hidden_size,
            latent_dim=hidden_size
        )

        self.query_tokennum = query_tokennum
        
        dim = params.latent_dim
        
        self.diffusion_embedding = SinusoidalPosEmb(dim)
        
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            Mish(),
            nn.Linear(dim * 4, params.transformer_hidden)
        )
        
        self.text_linear = nn.Linear(dim, params.transformer_hidden)
        self.style_linear = nn.Linear(dim, params.transformer_hidden)

        self.learned_query = nn.Parameter(torch.randn(self.query_tokennum, params.transformer_hidden))

        self.output_linear = nn.Linear(params.transformer_hidden, dim)
    
    def forward(self, text_vector, style_vector, diffusion_step):
        """
        :param text_vector (prompt representation): [B, num_tquery, C]
        :param (noised) style_vector (TTS control representation): [B, num_lquery, C]
        :param diffusion_step: [B]
        :return: predicted results (noise or x0)
        """
        diffusion_step = self.diffusion_embedding(diffusion_step)
        diffusion_step = self.mlp(diffusion_step)

        diffusion_step = diffusion_step.unsqueeze(1)  # B x 1 x C


        text_vector = self.text_linear(text_vector)  # B x num_tquery x C
        style_vector = self.style_linear(style_vector)  # B x num_lquery x C
        learned_queries = self.learned_query[None, :, :].repeat(style_vector.shape[0], 1, 1)

        x = self.decoder(text_vector, style_vector, learned_queries, diffusion_step=diffusion_step)
        x = self.output_linear(x)
        
        x = x.transpose(0, 1)  # B x T x C
        
        return x[:, -self.query_tokennum:, :]  # query relative to output
