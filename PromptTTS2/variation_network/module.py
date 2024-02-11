# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

# Reference: https://github.com/NATSpeech/NATSpeech/blob/main/modules/commons/transformer.py

import math
import torch
from torch import nn
from torch.nn import Parameter
import torch.nn.functional as F


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

class SinusoidalPositionalEmbedding(nn.Module):
    """This module produces sinusoidal positional embeddings of any length.

    Padding symbols are ignored.
    """

    def __init__(self, embedding_dim, padding_idx, init_size=1024):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        self.weights = SinusoidalPositionalEmbedding.get_embedding(
            init_size,
            embedding_dim,
            padding_idx,
        )
        self.register_buffer('_float_tensor', torch.FloatTensor(1))

    @staticmethod
    def get_embedding(num_embeddings, embedding_dim, padding_idx=None):
        """Build sinusoidal embeddings.

        This matches the implementation in tensor2tensor, but differs slightly
        from the description in Section 3.5 of "Attention Is All You Need".
        """
        half_dim = embedding_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float) * -emb)
        emb = torch.arange(num_embeddings, dtype=torch.float).unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1).view(num_embeddings, -1)
        if embedding_dim % 2 == 1:
            # zero pad
            emb = torch.cat([emb, torch.zeros(num_embeddings, 1)], dim=1)
        if padding_idx is not None:
            emb[padding_idx, :] = 0
        return emb

    def forward(self, input, incremental_state=None, timestep=None, **kwargs):
        """Input is expected to be of size [bsz x seqlen]."""
        bsz, seq_len = input.shape[:2]
        max_pos = self.padding_idx + 1 + seq_len
        if self.weights is None or max_pos > self.weights.size(0):
            # recompute/expand embeddings if needed
            self.weights = SinusoidalPositionalEmbedding.get_embedding(
                max_pos,
                self.embedding_dim,
                self.padding_idx,
            )
        self.weights = self.weights.to(self._float_tensor)

        if incremental_state is not None:
            # positions is the same for every token when decoding a single step
            pos = timestep.view(-1)[0] + 1 if timestep is not None else seq_len
            return self.weights[self.padding_idx + pos, :].expand(bsz, 1, -1)

        positions = make_positions(input, self.padding_idx)
        return self.weights.index_select(0, positions.view(-1)).view(bsz, seq_len, -1).detach()

    def max_positions(self):
        """Maximum number of supported positions."""
        return int(1e5)  # an arbitrary large number

class AffineLinear(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(AffineLinear, self).__init__()
        affine = nn.Linear(in_dim, out_dim)
        self.affine = affine

    def forward(self, input):
        return self.affine(input)

def Linear(in_features, out_features, bias=True):
    m = nn.Linear(in_features, out_features, bias)
    nn.init.xavier_uniform_(m.weight)
    if bias:
        nn.init.constant_(m.bias, 0.)
    return m

def softmax(x, dim):
    return F.softmax(x, dim=dim, dtype=torch.float32)

class MultiheadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, kdim=None, vdim=None, dropout=0., bias=True,
                 add_bias_kv=False, add_zero_attn=False, self_attention=False,
                 encoder_decoder_attention=False):
        super().__init__()
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self.qkv_same_dim = self.kdim == embed_dim and self.vdim == embed_dim

        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"
        self.scaling = self.head_dim ** -0.5

        self.self_attention = self_attention
        self.encoder_decoder_attention = encoder_decoder_attention

        assert not self.self_attention or self.qkv_same_dim, 'Self-attention requires query, key and ' \
                                                             'value to be of the same size'

        if self.qkv_same_dim:
            self.in_proj_weight = Parameter(torch.Tensor(3 * embed_dim, embed_dim))
        else:
            self.k_proj_weight = Parameter(torch.Tensor(embed_dim, self.kdim))
            self.v_proj_weight = Parameter(torch.Tensor(embed_dim, self.vdim))
            self.q_proj_weight = Parameter(torch.Tensor(embed_dim, embed_dim))

        if bias:
            self.in_proj_bias = Parameter(torch.Tensor(3 * embed_dim))
        else:
            self.register_parameter('in_proj_bias', None)

        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        if add_bias_kv:
            self.bias_k = Parameter(torch.Tensor(1, 1, embed_dim))
            self.bias_v = Parameter(torch.Tensor(1, 1, embed_dim))
        else:
            self.bias_k = self.bias_v = None

        self.add_zero_attn = add_zero_attn

        self.reset_parameters()

        self.enable_torch_version = False
        if hasattr(F, "multi_head_attention_forward"):
            self.enable_torch_version = True
        else:
            self.enable_torch_version = False

    def reset_parameters(self):
        if self.qkv_same_dim:
            nn.init.xavier_uniform_(self.in_proj_weight)
        else:
            nn.init.xavier_uniform_(self.k_proj_weight)
            nn.init.xavier_uniform_(self.v_proj_weight)
            nn.init.xavier_uniform_(self.q_proj_weight)

        nn.init.xavier_uniform_(self.out_proj.weight)
        if self.in_proj_bias is not None:
            nn.init.constant_(self.in_proj_bias, 0.)
            nn.init.constant_(self.out_proj.bias, 0.)
        if self.bias_k is not None:
            nn.init.xavier_normal_(self.bias_k)
        if self.bias_v is not None:
            nn.init.xavier_normal_(self.bias_v)

    def forward(
            self,
            query, key, value,
            key_padding_mask=None,
            incremental_state=None,
            need_weights=True,
            static_kv=False,
            attn_mask=None,
            before_softmax=False,
            need_head_weights=False,
            enc_dec_attn_constraint_mask=None
    ):
        """Input shape: Time x Batch x Channel

        Args:
            key_padding_mask (ByteTensor, optional): mask to exclude
                keys that are pads, of shape `(batch, src_len)`, where
                padding elements are indicated by 1s.
            need_weights (bool, optional): return the attention weights,
                averaged over heads (default: False).
            attn_mask (ByteTensor, optional): typically used to
                implement causal attention, where the mask prevents the
                attention from looking forward in time (default: None).
            before_softmax (bool, optional): return the raw attention
                weights and values before the attention softmax.
            need_head_weights (bool, optional): return the attention
                weights for each head. Implies *need_weights*. Default:
                return the average attention weights over all heads.
        """
        if need_head_weights:
            need_weights = True
        tgt_len, bsz, embed_dim = query.size()
        assert embed_dim == self.embed_dim
        assert list(query.size()) == [tgt_len, bsz, embed_dim]

        if self.enable_torch_version and incremental_state is None and not static_kv:
            if self.qkv_same_dim:
                return F.multi_head_attention_forward(query, key, value,
                                                      self.embed_dim, self.num_heads,
                                                      self.in_proj_weight,
                                                      self.in_proj_bias, self.bias_k, self.bias_v,
                                                      self.add_zero_attn, self.dropout,
                                                      self.out_proj.weight, self.out_proj.bias,
                                                      self.training, key_padding_mask, need_weights,
                                                      attn_mask)
            else:
                return F.multi_head_attention_forward(query, key, value,
                                                      self.embed_dim, self.num_heads,
                                                      torch.empty([0]),
                                                      self.in_proj_bias, self.bias_k, self.bias_v,
                                                      self.add_zero_attn, self.dropout,
                                                      self.out_proj.weight, self.out_proj.bias,
                                                      self.training, key_padding_mask, need_weights,
                                                      attn_mask, use_separate_proj_weight=True,
                                                      q_proj_weight=self.q_proj_weight,
                                                      k_proj_weight=self.k_proj_weight,
                                                      v_proj_weight=self.v_proj_weight)

        saved_state = None

        if self.self_attention:
            # self-attention
            q, k, v = self.in_proj_qkv(query)
        elif self.encoder_decoder_attention:
            # encoder-decoder attention
            q = self.in_proj_q(query)
            if key is None:
                assert value is None
                k = v = None
            else:
                k = self.in_proj_k(key)
                v = self.in_proj_v(key)

        else:
            q = self.in_proj_q(query)
            k = self.in_proj_k(key)
            v = self.in_proj_v(value)
        q *= self.scaling

        if self.bias_k is not None:
            assert self.bias_v is not None
            k = torch.cat([k, self.bias_k.repeat(1, bsz, 1)])
            v = torch.cat([v, self.bias_v.repeat(1, bsz, 1)])
            if attn_mask is not None:
                attn_mask = torch.cat([attn_mask, attn_mask.new_zeros(attn_mask.size(0), 1)], dim=1)
            if key_padding_mask is not None:
                key_padding_mask = torch.cat(
                    [key_padding_mask, key_padding_mask.new_zeros(key_padding_mask.size(0), 1)], dim=1)

        q = q.contiguous().view(tgt_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        if k is not None:
            k = k.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        if v is not None:
            v = v.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)

        src_len = k.size(1)

        # This is part of a workaround to get around fork/join parallelism
        # not supporting Optional types.
        if key_padding_mask is not None and key_padding_mask.shape == torch.Size([]):
            key_padding_mask = None

        if key_padding_mask is not None:
            assert key_padding_mask.size(0) == bsz
            assert key_padding_mask.size(1) == src_len

        if self.add_zero_attn:
            src_len += 1
            k = torch.cat([k, k.new_zeros((k.size(0), 1) + k.size()[2:])], dim=1)
            v = torch.cat([v, v.new_zeros((v.size(0), 1) + v.size()[2:])], dim=1)
            if attn_mask is not None:
                attn_mask = torch.cat([attn_mask, attn_mask.new_zeros(attn_mask.size(0), 1)], dim=1)
            if key_padding_mask is not None:
                key_padding_mask = torch.cat(
                    [key_padding_mask, torch.zeros(key_padding_mask.size(0), 1).type_as(key_padding_mask)], dim=1)

        attn_weights = torch.bmm(q, k.transpose(1, 2))
        attn_weights = self.apply_sparse_mask(attn_weights, tgt_len, src_len, bsz)

        assert list(attn_weights.size()) == [bsz * self.num_heads, tgt_len, src_len]

        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(0)
            attn_weights += attn_mask

        if enc_dec_attn_constraint_mask is not None:  # bs x head x L_kv
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights.masked_fill(
                enc_dec_attn_constraint_mask.unsqueeze(2).bool(),
                float('-inf'),
            )
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        if key_padding_mask is not None:
            # don't attend to padding symbols
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2),
                float('-inf'),
            )
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        attn_logits = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)

        if before_softmax:
            return attn_weights, v

        attn_weights_float = utils.softmax(attn_weights, dim=-1)
        attn_weights = attn_weights_float.type_as(attn_weights)
        attn_probs = F.dropout(attn_weights_float.type_as(attn_weights), p=self.dropout, training=self.training)

        attn = torch.bmm(attn_probs, v)
        assert list(attn.size()) == [bsz * self.num_heads, tgt_len, self.head_dim]
        attn = attn.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
        attn = self.out_proj(attn)

        if need_weights:
            attn_weights = attn_weights_float.view(bsz, self.num_heads, tgt_len, src_len).transpose(1, 0)
            if not need_head_weights:
                # average attention weights over heads
                attn_weights = attn_weights.mean(dim=0)
        else:
            attn_weights = None

        return attn, (attn_weights, attn_logits)

    def in_proj_qkv(self, query):
        return self._in_proj(query).chunk(3, dim=-1)

    def in_proj_q(self, query):
        if self.qkv_same_dim:
            return self._in_proj(query, end=self.embed_dim)
        else:
            bias = self.in_proj_bias
            if bias is not None:
                bias = bias[:self.embed_dim]
            return F.linear(query, self.q_proj_weight, bias)

    def in_proj_k(self, key):
        if self.qkv_same_dim:
            return self._in_proj(key, start=self.embed_dim, end=2 * self.embed_dim)
        else:
            weight = self.k_proj_weight
            bias = self.in_proj_bias
            if bias is not None:
                bias = bias[self.embed_dim:2 * self.embed_dim]
            return F.linear(key, weight, bias)

    def in_proj_v(self, value):
        if self.qkv_same_dim:
            return self._in_proj(value, start=2 * self.embed_dim)
        else:
            weight = self.v_proj_weight
            bias = self.in_proj_bias
            if bias is not None:
                bias = bias[2 * self.embed_dim:]
            return F.linear(value, weight, bias)

    def _in_proj(self, input, start=0, end=None):
        weight = self.in_proj_weight
        bias = self.in_proj_bias
        weight = weight[start:end, :]
        if bias is not None:
            bias = bias[start:end]
        return F.linear(input, weight, bias)


    def apply_sparse_mask(self, attn_weights, tgt_len, src_len, bsz):
        return attn_weights




class StyleAdaptiveLayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=True):
        super(StyleAdaptiveLayerNorm, self).__init__()
        self.in_channel = normalized_shape
        self.norm = nn.LayerNorm(self.in_channel, elementwise_affine=False, eps=eps)

        self.style = AffineLinear(self.in_channel, self.in_channel * 2)
        self.style.affine.bias.data[:self.in_channel] = 1
        self.style.affine.bias.data[self.in_channel:] = 0

    def forward(self, input, condition):

        style = self.style(condition).unsqueeze(0)

        gamma, beta = style.chunk(2, -1)
        
        out = self.norm(input)

        
        out = gamma * out + beta
        return out

class ClassicLayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=True):
        super(ClassicLayerNorm, self).__init__()
        self.norm = nn.LayerNorm(normalized_shape=normalized_shape, eps=eps, elementwise_affine=elementwise_affine)

    def forward(self, input, condition=None):
        out = self.norm(input)
        return out

def LayerNorm(normalized_shape, eps=1e-5, elementwise_affine=True, export=False, use_cln=False):
    
    if use_cln:
        return StyleAdaptiveLayerNorm(normalized_shape=normalized_shape, eps=eps, elementwise_affine=elementwise_affine)
    else:
        return ClassicLayerNorm(normalized_shape=normalized_shape, eps=eps, elementwise_affine=elementwise_affine)
    

class TransformerFFNLayer(nn.Module):
    def __init__(self, hidden_size, filter_size, padding="SAME", kernel_size=1, dropout=0.):
        super().__init__()
        self.kernel_size = kernel_size
        self.dropout = dropout
        if padding == 'SAME':
            self.ffn_1 = nn.Conv1d(hidden_size, filter_size, kernel_size, padding=kernel_size // 2)
        elif padding == 'LEFT':
            self.ffn_1 = nn.Sequential(
                nn.ConstantPad1d((kernel_size - 1, 0), 0.0),
                nn.Conv1d(hidden_size, filter_size, kernel_size)
            )
        self.ffn_2 = Linear(filter_size, hidden_size)

    def forward(self, x):

        x = self.ffn_1(x.permute(1, 2, 0)).permute(2, 0, 1)
        x = x * self.kernel_size ** -0.5

        x = F.relu(x)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.ffn_2(x)
        return x




class EncSALayer(nn.Module):
    def __init__(self, c, num_heads, dropout, attention_dropout=0.1, relu_dropout=0.1, use_cln=False,
                 kernel_size=9, padding='SAME'):
        super().__init__()
        self.c = c
        self.dropout = dropout
        self.layer_norm1 = LayerNorm(c, use_cln=use_cln)

        self.self_attn = MultiheadAttention(
            self.c, num_heads, self_attention=True, dropout=attention_dropout, bias=False,
        )
        self.layer_norm2 = LayerNorm(c, use_cln=use_cln)
        self.ffn = TransformerFFNLayer(c, 4 * c, kernel_size=kernel_size, dropout=relu_dropout, padding=padding)

    def forward(self, x, encoder_padding_mask=None, condition=None, **kwargs):
        layer_norm_training = kwargs.get('layer_norm_training', None)
        if layer_norm_training is not None:
            self.layer_norm1.training = layer_norm_training
            self.layer_norm2.training = layer_norm_training
        residual = x

        x = self.layer_norm1(x, condition=condition)
        x, _, = self.self_attn(
            query=x,
            key=x,
            value=x,
            key_padding_mask=encoder_padding_mask
        )
        x = F.dropout(x, self.dropout, training=self.training)
        x = residual + x
        x = x * (1 - encoder_padding_mask.to(x.dtype)).transpose(0, 1)[..., None]

        residual = x
        x = self.layer_norm2(x, condition=condition)
        x = self.ffn(x)
        x = F.dropout(x, self.dropout, training=self.training)
        x = residual + x
        x = x * (1 - encoder_padding_mask.to(x.dtype)).transpose(0, 1)[..., None]
        return x

class TransformerEncoderLayer(nn.Module):
    def __init__(self, layer, hidden_size, dropout, use_cln):
        super().__init__()
        self.layer = layer
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.op = EncSALayer(hidden_size, 4, use_cln=use_cln, dropout=dropout,
                                     attention_dropout=0.0, relu_dropout=dropout)

    def forward(self, x, **kwargs):
        return self.op(x, **kwargs)