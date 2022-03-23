# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import math
import torch
from torch import nn
from torch.nn import Parameter
import torch.onnx.operators
import torch.nn.functional as F
import tts_utils
from tts_utils.hparams import hparams


class SelfAttention(nn.Module):
    def __init__(self, hid_dim, n_heads, dropout=0.1, gaussian_bias=False, gaussian_tao=None,
                 gaus_init_l=3000):
        super().__init__()

        self.hid_dim = hid_dim
        self.n_heads = n_heads
        assert hid_dim % n_heads == 0

        self.w_q = Linear(hid_dim, hid_dim)
        self.w_k = Linear(hid_dim, hid_dim)
        self.w_v = Linear(hid_dim, hid_dim)
        self.gaussian_bias = gaussian_bias
        if gaussian_bias:
            self.tao = nn.Parameter(torch.FloatTensor(n_heads))
            nn.init.constant_(self.tao, gaussian_tao)  # sigma = tao^2
            # pre construct a gaussian matrix without dividing sigma^2
            self.bias_matrix = torch.Tensor([[-abs(i - j) ** 2 / 2.
                                              for i in range(gaus_init_l)] for j in range(gaus_init_l)])
        self.fc = Linear(hid_dim, hid_dim)
        self.dropout = nn.Dropout(dropout)

        self.sqrt_d = (hid_dim // n_heads) ** -0.5

    def forward(self, query, key, value, mask=None, require_w=False):
        # Q,K,V计算与变形： input is L B C
        for m in [query, key, value]:
            m.transpose_(0, 1)  # convert to B L C

        bsz, length, emb_dim = query.shape

        Q = self.w_q(query)  # B, L, hid_emb
        K = self.w_k(key)
        V = self.w_v(value)

        # -1 means the dim should be inferred
        # B, L, n, C//n，把embedding分成n份, n为num_head
        Q = Q.view(bsz, -1, self.n_heads, self.hid_dim // self.n_heads)
        Q = Q.permute(0, 2, 1, 3)  # B, n, L, C//n
        K = K.view(bsz, -1, self.n_heads, self.hid_dim // self.n_heads).permute(0, 2, 1, 3)
        V = V.view(bsz, -1, self.n_heads, self.hid_dim // self.n_heads).permute(0, 2, 1, 3)

        QK = torch.matmul(Q, K.transpose(2, 3)) * self.sqrt_d  # Transpose last two dim, out = B, n, L, L

        if self.gaussian_bias:
            # gaussian distribution: L*L
            L = QK.size(-1)
            # get matrix of  -(i-j)^2 / 2: i, j \in [0, L-1]
            if L <= self.bias_matrix.size(0):
                gaussian_mask = self.bias_matrix[:L, :L].repeat(self.n_heads, 1, 1).to(QK.device)  # n. L. L.
            else:  # 样本太长, 预构建的数值矩阵不够大
                gaussian_mask = torch.tensor([[-abs(i - j) ** 2 / 2.
                                               for i in range(L)] for j in range(L)]
                                             ).repeat(self.n_heads, 1, 1).to(QK.device)
                print("Tensor is too long, size:", L)

            # divide gaussian matrix by tao^4 (multiply by tao^(-4))
            # tao: nn.Parameters, size [n_head]
            gaussian_mask = torch.mul(gaussian_mask, torch.pow(self.tao, -4)[:, None, None])  # out shape n,L,L
            QK += gaussian_mask.repeat(bsz, 1, 1, 1).to(QK.device)  # expand mask n,L,L to B, n, L, L

        if mask is not None:
            '''
            attn weight size: b*n_h, L, L
            mask size: b, L -> b, 1, 1, L + 
            attn_weight(b,n_head,L,L).masked_fill(mask)
            '''
            QK = QK.masked_fill(mask[:, None, None, :], float('-inf'))

        # 然后对Q,K相乘的结果 计算softmax 计算weight
        attn_weight = torch.softmax(QK, dim=-1)
        attention = self.dropout(attn_weight)

        # 第三步，attention结果与V相乘
        x = torch.matmul(attention, V)  # B, n, L, C//n

        # 最后将多头排列好，就是multi-head attention的结果了
        x = x.permute(0, 2, 1, 3).contiguous()  # B, L, n, C//n
        x = x.view(bsz, -1, self.n_heads * (self.hid_dim // self.n_heads))  # B, L, C

        x = self.fc(x)

        x.transpose_(0, 1)  # return L B C
        # return to operations.py: EncGausSALayer.forward()
        return (x, attn_weight[:1, :1, ...]) if require_w else (x, None)  # remember to add bracket


class EncGausSALayer(nn.Module):
    def __init__(self, hid_dim, num_heads, dropout, attention_dropout=0.1, relu_dropout=0.1, gaus_bias=False,
                 gaus_tao=10):
        super().__init__()
        self.dropout = dropout
        self.layer_norm1 = LayerNorm(hid_dim)
        # hid_dim, n_heads, dropout=0.1, gaussian_bias=False, gaussian_tao=None
        self.self_attn_gaus_bias = SelfAttention(hid_dim, num_heads, attention_dropout, gaus_bias, gaus_tao)
        self.layer_norm2 = LayerNorm(hid_dim)
        self.ffn = TransformerFFNLayer(hid_dim, 4 * hid_dim, kernel_size=9, dropout=relu_dropout)

    def forward(self, x, encoder_padding_mask=None, require_w=False, **kwargs):
        layer_norm_training = kwargs.get('layer_norm_training', None)
        # print("EncGausSA, kwargs", str(kwargs))
        # require_w = kwargs.get('require_w', False)
        # print("EncGausSA, require_w", require_w)
        if layer_norm_training is not None:
            self.layer_norm1.training = layer_norm_training
            self.layer_norm2.training = layer_norm_training
        residual = x
        x = self.layer_norm1(x)
        # x = self.self_attn_gaus_bias(query=x, key=x, value=x, mask=encoder_padding_mask)
        x, attn_w = self.self_attn_gaus_bias(query=x, key=x, value=x, mask=encoder_padding_mask, require_w=require_w)
        # print("self_attn return:", type(x))
        x = F.dropout(x, self.dropout, training=self.training)
        x = residual + x

        residual = x
        x = self.layer_norm2(x)
        x = self.ffn(x)
        x = F.dropout(x, self.dropout, training=self.training)
        x = residual + x
        return (x, attn_w) if require_w else x


class CyclicalPositionEmb(nn.Module):
    def __init__(self, K, emb_size):
        super(CyclicalPositionEmb, self).__init__()

        self.fc = Linear(K, emb_size)

    def forward(self, x):
        '''
        :param x: B * T * 1
        :return: x
        '''
        pass  # todo


def LayerNorm(normalized_shape, eps=1e-5, elementwise_affine=True, export=False):
    if not export and torch.cuda.is_available():
        try:
            from apex.normalization import FusedLayerNorm
            return FusedLayerNorm(normalized_shape, eps, elementwise_affine)
        except ImportError:
            pass
    return torch.nn.LayerNorm(normalized_shape, eps, elementwise_affine)


def Linear(in_features, out_features, bias=True):
    m = nn.Linear(in_features, out_features, bias)
    nn.init.xavier_uniform_(m.weight)
    if bias:
        nn.init.constant_(m.bias, 0.)
    return m


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

        positions = tts_utils.make_positions(input, self.padding_idx)
        return self.weights.index_select(0, positions.view(-1)).view(bsz, seq_len, -1).detach()

    def max_positions(self):
        """Maximum number of supported positions."""
        return int(1e5)  # an arbitrary large number


class ConvTBC(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding=0):
        super(ConvTBC, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = padding

        self.weight = torch.nn.Parameter(torch.Tensor(
            self.kernel_size, in_channels, out_channels))
        self.bias = torch.nn.Parameter(torch.Tensor(out_channels))

    def forward(self, input):
        return torch.conv_tbc(input.contiguous(), self.weight, self.bias, self.padding)


class EncConvLayer(nn.Module):
    def __init__(self, c, kernel_size, dropout):
        super().__init__()
        self.layer_norm = LayerNorm(c)
        conv = ConvTBC(c, c, kernel_size, padding=kernel_size // 2)
        std = math.sqrt((4 * (1.0 - dropout)) / (kernel_size * c))
        nn.init.normal_(conv.weight, mean=0, std=std)
        nn.init.constant_(conv.bias, 0)
        self.conv = nn.utils.weight_norm(conv, dim=2)
        self.dropout = dropout

    def forward(self, x, encoder_padding_mask=None, **kwargs):
        layer_norm_training = kwargs.get('layer_norm_training', None)
        if layer_norm_training is not None:
            self.layer_norm.training = layer_norm_training
        residual = x
        if encoder_padding_mask is not None:
            x = x.masked_fill(encoder_padding_mask.t().unsqueeze(-1), 0)
        x = self.layer_norm(x)
        x = self.conv(x)
        x = F.relu(x)
        x = F.dropout(x, self.dropout, self.training)
        x = x + residual
        return x


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

        if incremental_state is not None:
            saved_state = self._get_input_buffer(incremental_state)
            if 'prev_key' in saved_state:
                # previous time steps are cached - no need to recompute
                # key and value if they are static
                if static_kv:
                    assert self.encoder_decoder_attention and not self.self_attention
                    key = value = None
        else:
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

        if saved_state is not None:
            # saved states are stored with shape (bsz, num_heads, seq_len, head_dim)
            if 'prev_key' in saved_state:
                prev_key = saved_state['prev_key'].view(bsz * self.num_heads, -1, self.head_dim)
                if static_kv:
                    k = prev_key
                else:
                    k = torch.cat((prev_key, k), dim=1)
            if 'prev_value' in saved_state:
                prev_value = saved_state['prev_value'].view(bsz * self.num_heads, -1, self.head_dim)
                if static_kv:
                    v = prev_value
                else:
                    v = torch.cat((prev_value, v), dim=1)
            if 'prev_key_padding_mask' in saved_state and saved_state['prev_key_padding_mask'] is not None:
                prev_key_padding_mask = saved_state['prev_key_padding_mask']
                if static_kv:
                    key_padding_mask = prev_key_padding_mask
                else:
                    key_padding_mask = torch.cat((prev_key_padding_mask, key_padding_mask), dim=1)

            saved_state['prev_key'] = k.view(bsz, self.num_heads, -1, self.head_dim)
            saved_state['prev_value'] = v.view(bsz, self.num_heads, -1, self.head_dim)
            saved_state['prev_key_padding_mask'] = key_padding_mask

            self._set_input_buffer(incremental_state, saved_state)

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

        attn_weights_float = tts_utils.softmax(attn_weights, dim=-1)
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

    def _get_input_buffer(self, incremental_state):
        return tts_utils.get_incremental_state(
            self,
            incremental_state,
            'attn_state',
        ) or {}

    def _set_input_buffer(self, incremental_state, buffer):
        tts_utils.set_incremental_state(
            self,
            incremental_state,
            'attn_state',
            buffer,
        )

    def apply_sparse_mask(self, attn_weights, tgt_len, src_len, bsz):
        return attn_weights

    def clear_buffer(self, incremental_state=None):
        if incremental_state is not None:
            saved_state = self._get_input_buffer(incremental_state)
            if 'prev_key' in saved_state:
                del saved_state['prev_key']
            if 'prev_value' in saved_state:
                del saved_state['prev_value']
            self._set_input_buffer(incremental_state, saved_state)


class TransformerFFNLayer(nn.Module):
    def __init__(self, hidden_size, filter_size, padding="SAME", kernel_size=1, dropout=0.):
        super().__init__()
        self.kernel_size = kernel_size
        self.dropout = dropout
        if kernel_size == 1:
            self.ffn_1 = Linear(hidden_size, filter_size)
        else:
            if padding == 'SAME':
                assert kernel_size % 2 == 1
                self.first_offset = -((kernel_size - 1) // 2)
            else:
                assert padding == 'LEFT'
                self.first_offset = -(kernel_size - 1)
            self.last_offset = self.first_offset + kernel_size - 1
            self.ffn_1 = nn.ModuleList()
            for i in range(kernel_size):
                self.ffn_1.append(Linear(hidden_size, filter_size, bias=(i == 0)))
        self.ffn_2 = Linear(filter_size, hidden_size)

    def forward(self, x, incremental_state=None):
        # x: T x B x C
        if incremental_state is not None:
            saved_state = self._get_input_buffer(incremental_state)
            if 'prev_input' in saved_state:
                prev_input = saved_state['prev_input']
                x = torch.cat((prev_input, x), dim=0)
            x = x[-self.kernel_size:]
            saved_state['prev_input'] = x
            self._set_input_buffer(incremental_state, saved_state)

        if self.kernel_size == 1:
            x = self.ffn_1(x)
        else:
            padded = F.pad(x, (0, 0, 0, 0, -self.first_offset, self.last_offset))
            results = []
            for i in range(self.kernel_size):
                shifted = padded[i:x.size(0) + i] if i else x
                results.append(self.ffn_1[i](shifted))
            res = sum(results)
            x = res * self.kernel_size ** -0.5

        if incremental_state is not None:
            x = x[-1:]

        x = F.relu(x)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.ffn_2(x)
        return x

    def _get_input_buffer(self, incremental_state):
        return tts_utils.get_incremental_state(
            self,
            incremental_state,
            'f',
        ) or {}

    def _set_input_buffer(self, incremental_state, buffer):
        tts_utils.set_incremental_state(
            self,
            incremental_state,
            'f',
            buffer,
        )

    def clear_buffer(self, incremental_state):
        if incremental_state is not None:
            saved_state = self._get_input_buffer(incremental_state)
            if 'prev_input' in saved_state:
                del saved_state['prev_input']
            self._set_input_buffer(incremental_state, saved_state)


class NewTransformerFFNLayer(nn.Module):
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

    def forward(self, x, incremental_state=None):
        # x: T x B x C
        if incremental_state is not None:
            saved_state = self._get_input_buffer(incremental_state)
            if 'prev_input' in saved_state:
                prev_input = saved_state['prev_input']
                x = torch.cat((prev_input, x), dim=0)
            x = x[-self.kernel_size:]
            saved_state['prev_input'] = x
            self._set_input_buffer(incremental_state, saved_state)

        x = self.ffn_1(x.permute(1, 2, 0)).permute(2, 0, 1)
        x = x * self.kernel_size ** -0.5

        if incremental_state is not None:
            x = x[-1:]

        x = F.relu(x)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.ffn_2(x)
        return x

    def _get_input_buffer(self, incremental_state):
        return tts_utils.get_incremental_state(
            self,
            incremental_state,
            'f',
        ) or {}

    def _set_input_buffer(self, incremental_state, buffer):
        tts_utils.set_incremental_state(
            self,
            incremental_state,
            'f',
            buffer,
        )

    def clear_buffer(self, incremental_state):
        if incremental_state is not None:
            saved_state = self._get_input_buffer(incremental_state)
            if 'prev_input' in saved_state:
                del saved_state['prev_input']
            self._set_input_buffer(incremental_state, saved_state)


class EncSALayer(nn.Module):
    def __init__(self, c, num_heads, dropout, attention_dropout=0.1, relu_dropout=0.1,
                 kernel_size=9, padding='SAME'):
        super().__init__()
        self.c = c
        self.dropout = dropout
        self.layer_norm1 = LayerNorm(c)

        self.self_attn = MultiheadAttention(
            self.c, num_heads, self_attention=True, dropout=attention_dropout, bias=False,
        )
        self.layer_norm2 = LayerNorm(c)
        if hparams['use_new_ffn']:
            self.ffn = NewTransformerFFNLayer(c, 4 * c, kernel_size=kernel_size, dropout=relu_dropout, padding=padding)
        else:
            self.ffn = TransformerFFNLayer(c, 4 * c, kernel_size=kernel_size, dropout=relu_dropout, padding=padding)

    def forward(self, x, encoder_padding_mask=None, **kwargs):
        layer_norm_training = kwargs.get('layer_norm_training', None)
        if layer_norm_training is not None:
            self.layer_norm1.training = layer_norm_training
            self.layer_norm2.training = layer_norm_training
        residual = x
        x = self.layer_norm1(x)
        x, _, = self.self_attn(
            query=x,
            key=x,
            value=x,
            key_padding_mask=encoder_padding_mask
        )
        x = F.dropout(x, self.dropout, training=self.training)
        x = residual + x
        x = x * (1 - encoder_padding_mask.float()).transpose(0, 1)[..., None]

        residual = x
        x = self.layer_norm2(x)
        x = self.ffn(x)
        x = F.dropout(x, self.dropout, training=self.training)
        x = residual + x
        x = x * (1 - encoder_padding_mask.float()).transpose(0, 1)[..., None]
        return x


class EncLocalSALayer(nn.Module):
    def __init__(self, c, num_heads, dropout, attention_dropout=0.1, relu_dropout=0.1):
        super().__init__()
        self.c = c
        self.dropout = dropout
        self.layer_norm1 = LayerNorm(c)

        self.self_attn = MultiheadAttention(
            self.c, num_heads, self_attention=True, dropout=attention_dropout, bias=False,
        )
        self.layer_norm2 = LayerNorm(c)
        self.ffn = TransformerFFNLayer(c, 4 * c, kernel_size=9, dropout=relu_dropout)
        self.chunk_size = 101

    def forward(self, x, encoder_padding_mask=None, **kwargs):
        layer_norm_training = kwargs.get('layer_norm_training', None)
        if layer_norm_training is not None:
            self.layer_norm1.training = layer_norm_training
            self.layer_norm2.training = layer_norm_training
        residual = x
        x = self.layer_norm1(x)
        states = []
        T = x.shape[0]
        all_neg_inf = tts_utils.fill_with_neg_inf2(x.new(T, T))
        half_chunk_size = self.chunk_size // 2
        attn_mask = torch.triu(all_neg_inf, half_chunk_size + 1) \
                    + torch.tril(all_neg_inf, -half_chunk_size - 1)
        encoder_padding_mask = encoder_padding_mask.data
        for i in range(0, x.shape[0], half_chunk_size + 1):
            k_start = max(0, i - half_chunk_size)
            k_end = min(x.shape[0], i + self.chunk_size)
            kv = x[k_start:k_end]
            q = x[i:i + half_chunk_size + 1]

            q_nonpadding = (1 - encoder_padding_mask[:, i:i + half_chunk_size + 1].float()).data
            encoder_padding_mask_ = encoder_padding_mask[:, k_start:k_end].data
            encoder_padding_mask_[q_nonpadding.sum(-1) == 0, :] = 0
            x_, _, = self.self_attn(
                query=q,
                key=kv,
                value=kv,
                key_padding_mask=encoder_padding_mask_,
                attn_mask=attn_mask[i:i + half_chunk_size + 1, k_start:k_end]
            )
            x_ = x_ * (1 - q_nonpadding.T[:, :, None])
            states.append(x_)
        x = torch.cat(states)
        x = F.dropout(x, self.dropout, training=self.training)
        x = residual + x

        residual = x
        x = self.layer_norm2(x)
        x = self.ffn(x)
        x = F.dropout(x, self.dropout, training=self.training)
        x = residual + x
        return x


class EncLSTMLayer(nn.Module):
    def __init__(self, c, dropout):
        super().__init__()
        self.c = c
        self.layer_norm = LayerNorm(c)
        self.lstm = nn.LSTM(c, c, 1, bidirectional=True)
        self.out_proj = Linear(2 * c, c)
        self.dropout = dropout

    def forward(self, x, **kwargs):
        layer_norm_training = kwargs.get('layer_norm_training', None)
        if layer_norm_training is not None:
            self.layer_norm.training = layer_norm_training
        self.lstm.flatten_parameters()
        residual = x
        x = self.layer_norm(x)
        x, _ = self.lstm(x)
        x = self.out_proj(x)
        x = F.dropout(x, self.dropout, training=self.training)
        x = residual + x
        return x


class ConvAttentionLayer(nn.Module):
    def __init__(self, c, hidden_size, dropout=0.):
        super().__init__()
        self.in_projection = Linear(c, hidden_size)
        self.out_projection = Linear(hidden_size, c)
        self.dropout = dropout

    def forward(self, x, key, value, encoder_padding_mask=None, enc_dec_attn_constraint_mask=None):
        # x, key, value : T x B x C
        # attention
        query = self.in_projection(x)
        attn_weights = torch.bmm(query.transpose(0, 1), key.transpose(0, 1).transpose(1, 2))

        # don't attend over padding
        if encoder_padding_mask is not None:
            attn_weights = attn_weights.masked_fill(
                encoder_padding_mask.unsqueeze(1),
                float('-inf')
            ).type_as(attn_weights)  # FP16 support: cast to float and back

        if enc_dec_attn_constraint_mask is not None:
            attn_weights = attn_weights.masked_fill(
                enc_dec_attn_constraint_mask.bool(),
                float('-inf'),
            ).type_as(attn_weights)

        attn_logits = attn_weights
        # softmax over last dim
        sz = attn_weights.size()
        attn_scores = F.softmax(attn_weights.view(sz[0] * sz[1], sz[2]), dim=1)
        attn_scores = attn_scores.view(sz)
        attn_scores = F.dropout(attn_scores, p=self.dropout, training=self.training)

        attn = torch.bmm(attn_scores, value.transpose(0, 1)).transpose(0, 1)

        # scale attention output (respecting potentially different lengths)
        s = value.size(0)
        if encoder_padding_mask is None:
            attn = attn * (s * math.sqrt(1.0 / s))
        else:
            s = s - encoder_padding_mask.type_as(attn).sum(dim=1, keepdim=True)  # exclude padding
            s = s.transpose(0, 1).unsqueeze(-1)
            attn = attn * (s * s.rsqrt())

        # project back
        attn = self.out_projection(attn)
        return attn, attn_scores, attn_logits


class LinearizedConvolution(ConvTBC):
    """An optimized version of nn.Conv1d.

    At training time, this module uses ConvTBC, which is an optimized version
    of Conv1d. At inference time, it optimizes incremental generation (i.e.,
    one time step at a time) by replacing the convolutions with linear layers.
    Note that the input order changes from training to inference.
    """

    def __init__(self, in_channels, out_channels, kernel_size, **kwargs):
        super().__init__(in_channels, out_channels, kernel_size, **kwargs)
        self._linearized_weight = None
        self.register_backward_hook(self._clear_linearized_weight)

    def forward(self, input, incremental_state=None):
        """
        Args:
            incremental_state: Used to buffer signal; if not None, then input is
                expected to contain a single frame. If the input order changes
                between time steps, call reorder_incremental_state.
        Input:
            Time x Batch x Channel
        """
        if incremental_state is None:
            output = super().forward(input)
            if self.kernel_size > 1 and self.padding > 0:
                # remove future timesteps added by padding
                output = output[:-self.padding, :, :]
            return output

        # reshape weight
        weight = self._get_linearized_weight()
        kw = self.kernel_size

        input = input.transpose(0, 1)
        bsz = input.size(0)  # input: bsz x len x dim
        if kw > 1:
            input = input.data
            input_buffer = self._get_input_buffer(incremental_state)
            if input_buffer is None:
                input_buffer = input.new(bsz, kw, input.size(2)).zero_()
                self._set_input_buffer(incremental_state, input_buffer)
            else:
                # shift buffer
                input_buffer[:, :-1, :] = input_buffer[:, 1:, :].clone()
            # append next input
            input_buffer[:, -1, :] = input[:, -1, :]
            input = input_buffer
        with torch.no_grad():
            output = F.linear(input.view(bsz, -1), weight, self.bias)
        return output.view(bsz, 1, -1).transpose(0, 1)

    def _get_input_buffer(self, incremental_state):
        return tts_utils.get_incremental_state(self, incremental_state, 'input_buffer')

    def _set_input_buffer(self, incremental_state, new_buffer):
        return tts_utils.set_incremental_state(self, incremental_state, 'input_buffer', new_buffer)

    def _get_linearized_weight(self):
        if self._linearized_weight is None:
            kw = self.kernel_size
            weight = self.weight.transpose(2, 1).transpose(1, 0).contiguous()
            assert weight.size() == (self.out_channels, kw, self.in_channels)
            self._linearized_weight = weight.view(self.out_channels, -1)
        return self._linearized_weight

    def _clear_linearized_weight(self, *args):
        self._linearized_weight = None

    def clear_buffer(self, input, incremental_state=None):
        if incremental_state is not None:
            self._set_input_buffer(incremental_state, None)


class DecConvLayer(nn.Module):
    def __init__(self, c, kernel_size, dropout, attention_dropout=0.1):
        super().__init__()
        self.layer_norm1 = LayerNorm(c)
        conv = LinearizedConvolution(c, c, kernel_size, padding=kernel_size - 1)
        std = math.sqrt((4 * (1.0 - dropout)) / (kernel_size * c))
        nn.init.normal_(conv.weight, mean=0, std=std)
        nn.init.constant_(conv.bias, 0)
        self.conv = nn.utils.weight_norm(conv, dim=2)
        self.layer_norm2 = LayerNorm(c)
        # self.attention = ConvAttentionLayer(c, c, dropout=attention_dropout)
        self.attention = MultiheadAttention(c, 1, dropout=attention_dropout, encoder_decoder_attention=True, bias=False)
        self.dropout = dropout

    def forward(self, x, encoder_out=None, encoder_padding_mask=None, incremental_state=None, **kwargs):
        layer_norm_training = kwargs.get('layer_norm_training', None)
        if layer_norm_training is not None:
            self.layer_norm1.training = layer_norm_training
            self.layer_norm2.training = layer_norm_training
        residual = x
        x = self.layer_norm1(x)
        x = self.conv(x, incremental_state=incremental_state)
        x = F.relu(x)
        x = F.dropout(x, self.dropout, training=self.training)
        x = residual + x
        x = self.layer_norm2(x)
        x, attn = self.attention(
            query=x,
            key=encoder_out,
            value=encoder_out,
            key_padding_mask=encoder_padding_mask,
            incremental_state=incremental_state,
            static_kv=True,
            enc_dec_attn_constraint_mask=tts_utils.get_incremental_state(self, incremental_state,
                                                                     'enc_dec_attn_constraint_mask')
        )
        x = F.dropout(x, self.dropout, training=self.training)
        x = residual + x
        attn_logits = attn[1]
        # if len(attn_logits.size()) > 3:
        #    attn_logits = attn_logits[:, 0]
        return x, attn_logits

    def clear_buffer(self, input, encoder_out=None, encoder_padding_mask=None, incremental_state=None):
        self.conv.clear_buffer(input, incremental_state)

    def set_buffer(self, name, tensor, incremental_state):
        return tts_utils.set_incremental_state(self, incremental_state, name, tensor)


class DecSALayer(nn.Module):
    def __init__(self, c, num_heads, dropout, attention_dropout=0.1, relu_dropout=0.1, kernel_size=9):
        super().__init__()
        self.c = c
        self.dropout = dropout
        self.layer_norm1 = LayerNorm(c)
        self.self_attn = MultiheadAttention(
            c, num_heads, self_attention=True, dropout=attention_dropout, bias=False
        )
        self.layer_norm2 = LayerNorm(c)
        self.encoder_attn = MultiheadAttention(
            c, num_heads, encoder_decoder_attention=True, dropout=attention_dropout, bias=False,
        )
        self.layer_norm3 = LayerNorm(c)
        if hparams['use_new_ffn']:
            self.ffn = NewTransformerFFNLayer(c, 4 * c, padding='LEFT', kernel_size=kernel_size, dropout=relu_dropout)
        else:
            self.ffn = TransformerFFNLayer(c, 4 * c, padding='LEFT', kernel_size=kernel_size, dropout=relu_dropout)

    def forward(
            self,
            x,
            encoder_out=None,
            encoder_padding_mask=None,
            incremental_state=None,
            self_attn_mask=None,
            self_attn_padding_mask=None,
            **kwargs,
    ):
        layer_norm_training = kwargs.get('layer_norm_training', None)
        if layer_norm_training is not None:
            self.layer_norm1.training = layer_norm_training
            self.layer_norm2.training = layer_norm_training
            self.layer_norm3.training = layer_norm_training
        residual = x
        x = self.layer_norm1(x)
        x, _ = self.self_attn(
            query=x,
            key=x,
            value=x,
            key_padding_mask=self_attn_padding_mask,
            incremental_state=incremental_state,
            attn_mask=self_attn_mask
        )
        x = F.dropout(x, self.dropout, training=self.training)
        x = residual + x

        residual = x
        x = self.layer_norm2(x)
        x, attn = self.encoder_attn(
            query=x,
            key=encoder_out,
            value=encoder_out,
            key_padding_mask=encoder_padding_mask,
            incremental_state=incremental_state,
            static_kv=True,
            enc_dec_attn_constraint_mask=tts_utils.get_incremental_state(self, incremental_state,
                                                                     'enc_dec_attn_constraint_mask')
        )
        x = F.dropout(x, self.dropout, training=self.training)
        x = residual + x

        residual = x
        x = self.layer_norm3(x)
        x = self.ffn(x, incremental_state=incremental_state)
        x = F.dropout(x, self.dropout, training=self.training)
        x = residual + x
        attn_logits = attn[1]
        # if len(attn_logits.size()) > 3:
        #    indices = attn_logits.softmax(-1).max(-1).values.sum(-1).argmax(-1)
        #    attn_logits = attn_logits.gather(1, 
        #        indices[:, None, None, None].repeat(1, 1, attn_logits.size(-2), attn_logits.size(-1))).squeeze(1)
        return x, attn_logits

    def clear_buffer(self, input, encoder_out=None, encoder_padding_mask=None, incremental_state=None):
        self.encoder_attn.clear_buffer(incremental_state)
        self.ffn.clear_buffer(incremental_state)

    def set_buffer(self, name, tensor, incremental_state):
        return tts_utils.set_incremental_state(self, incremental_state, name, tensor)


class LSTMAttentionLayer(nn.Module):
    def __init__(self, input_embed_dim, source_embed_dim, output_embed_dim, bias=False, dropout=0.):
        super().__init__()

        self.input_proj = Linear(input_embed_dim, source_embed_dim, bias=bias)
        self.output_proj = Linear(input_embed_dim + source_embed_dim, output_embed_dim, bias=bias)
        self.dropout = dropout

    def forward(self, input, source_hids, encoder_padding_mask=None, enc_dec_attn_constraint_mask=None):
        # input: tgtlen x bsz x input_embed_dim
        # source_hids: srclen x bsz x source_embed_dim

        # x: tgtlen x bsz x source_embed_dim
        x = self.input_proj(input)

        # compute attention
        attn_weights = torch.bmm(x.transpose(0, 1), source_hids.transpose(0, 1).transpose(1, 2))

        # don't attend over padding
        if encoder_padding_mask is not None:
            attn_weights = attn_weights.float().masked_fill_(
                encoder_padding_mask.unsqueeze(1),
                float('-inf')
            ).type_as(attn_weights)  # FP16 support: cast to float and back

        if enc_dec_attn_constraint_mask is not None:
            attn_weights = attn_weights.float().masked_fill_(
                enc_dec_attn_constraint_mask.bool(),
                float('-inf')
            ).type_as(attn_weights)

        attn_logits = attn_weights
        sz = attn_weights.size()
        attn_scores = F.softmax(attn_weights.view(sz[0] * sz[1], sz[2]), dim=1)
        attn_scores = attn_scores.view(sz)
        attn_scores = F.dropout(attn_scores, p=self.dropout, training=self.training)

        # sum weighted sources
        attn = torch.bmm(attn_scores, source_hids.transpose(0, 1)).transpose(0, 1)

        x = torch.tanh(self.output_proj(torch.cat((attn, input), dim=-1)))
        return x, attn_scores, attn_logits


class DecLSTMLayer(nn.Module):
    def __init__(self, c, dropout, attention_dropout=0.1):
        super().__init__()
        self.c = c
        self.layer_norm1 = LayerNorm(c)
        self.lstm = nn.LSTM(c, c, 1, dropout=dropout)
        self.layer_norm2 = LayerNorm(c)
        # self.attention = LSTMAttentionLayer(c, c, c, dropout=attention_dropout)
        self.attention = MultiheadAttention(c, 1, dropout=attention_dropout, encoder_decoder_attention=True, bias=False)
        self.dropout = dropout

    def forward(self, x, encoder_out=None, encoder_padding_mask=None, incremental_state=None, **kwargs):
        layer_norm_training = kwargs.get('layer_norm_training', None)
        if layer_norm_training is not None:
            self.layer_norm1.training = layer_norm_training
            self.layer_norm2.training = layer_norm_training
        self.lstm.flatten_parameters()
        if incremental_state is not None:
            x = x[-1:, :, :]
        cached_state = tts_utils.get_incremental_state(self, incremental_state, 'cached_state')
        if cached_state is not None:
            prev_hiddens, prev_cells = cached_state
        else:
            prev_hiddens = encoder_out.mean(dim=0, keepdim=True)
            prev_cells = encoder_out.mean(dim=0, keepdim=True)

        residual = x
        x = self.layer_norm1(x)
        x, hidden = self.lstm(x, (prev_hiddens, prev_cells))
        hiddens, cells = hidden
        x = residual + x

        x = self.layer_norm2(x)
        x, attn = self.attention(
            query=x,
            key=encoder_out,
            value=encoder_out,
            key_padding_mask=encoder_padding_mask,
            incremental_state=incremental_state,
            static_kv=True,
            enc_dec_attn_constraint_mask=tts_utils.get_incremental_state(self, incremental_state,
                                                                     'enc_dec_attn_constraint_mask')
        )
        x = F.dropout(x, self.dropout, training=self.training)

        if incremental_state is not None:
            # prev_hiddens = torch.cat((prev_hiddens, hiddens), dim=0)
            # prev_cells = torch.cat((prev_cells, cells), dim=0)
            prev_hiddens = hiddens
            prev_cells = cells
            tts_utils.set_incremental_state(
                self, incremental_state, 'cached_state',
                (prev_hiddens, prev_cells),
            )

        x = residual + x
        attn_logits = attn[1]
        # if len(attn_logits.size()) > 3:
        #    attn_logits = attn_logits[:, 0]
        return x, attn_logits

    def clear_buffer(self, input, encoder_out=None, encoder_padding_mask=None, incremental_state=None):
        if incremental_state is not None:
            prev_hiddens = encoder_out.mean(dim=0, keepdim=True)
            prev_cells = encoder_out.mean(dim=0, keepdim=True)
            tts_utils.set_incremental_state(
                self, incremental_state, 'cached_state',
                (prev_hiddens, prev_cells)
            )

    def set_buffer(self, name, tensor, incremental_state):
        return tts_utils.set_incremental_state(self, incremental_state, name, tensor)


OPERATIONS_ENCODER = {  # c = hidden size
    1: lambda c, dropout: EncConvLayer(c, 1, dropout),  # h, num_heads, dropout
    2: lambda c, dropout: EncConvLayer(c, 5, dropout),
    3: lambda c, dropout: EncConvLayer(c, 9, dropout),
    4: lambda c, dropout: EncConvLayer(c, 13, dropout),
    5: lambda c, dropout: EncConvLayer(c, 17, dropout),
    6: lambda c, dropout: EncConvLayer(c, 21, dropout),
    7: lambda c, dropout: EncConvLayer(c, 25, dropout),
    8: lambda c, dropout: EncSALayer(c, 2, dropout=dropout,
                                     attention_dropout=0.0, relu_dropout=dropout,
                                     kernel_size=hparams['enc_ffn_kernel_size'],
                                     padding=hparams['ffn_padding']),
    9: lambda c, dropout: EncSALayer(c, 4, dropout),
    10: lambda c, dropout: EncSALayer(c, 8, dropout),
    11: lambda c, dropout: EncLocalSALayer(c, 2, dropout),
    12: lambda c, dropout: EncLSTMLayer(c, dropout),
    13: lambda c, dropout, g_bias, tao: EncGausSALayer(c, 1, dropout, gaus_bias=g_bias, gaus_tao=tao),
    14: lambda c, dropout: EncSALayer(c, 2, dropout, kernel_size=1),
    15: lambda c, dropout: EncSALayer(c, 2, dropout, kernel_size=15),
}

OPERATIONS_DECODER = {
    1: lambda c, dropout: DecConvLayer(c, 1, dropout),
    2: lambda c, dropout: DecConvLayer(c, 5, dropout),
    3: lambda c, dropout: DecConvLayer(c, 9, dropout),
    4: lambda c, dropout: DecConvLayer(c, 13, dropout),
    5: lambda c, dropout: DecConvLayer(c, 17, dropout),
    6: lambda c, dropout: DecConvLayer(c, 21, dropout),
    7: lambda c, dropout: DecConvLayer(c, 25, dropout),
    8: lambda c, dropout: DecSALayer(c, 2, dropout=dropout,
                                     attention_dropout=0.0, relu_dropout=dropout,
                                     kernel_size=hparams['dec_ffn_kernel_size']),
    9: lambda c, dropout: DecSALayer(c, 4, dropout),
    10: lambda c, dropout: DecSALayer(c, 8, dropout),
    11: lambda c, dropout: DecLSTMLayer(c, dropout),
    12: lambda c, dropout: DecSALayer(c, 2, dropout, kernel_size=1),
}
