# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
import logging

import sys

import torch
import torch.nn.functional as F
from torch import Tensor
from argparse import Namespace
from typing import Any, Dict, List, Optional, Tuple
from fairseq import utils
from fairseq.models.transformer import (
    TransformerDecoder,
    TransformerEncoder,
    TransformerModel,
)
from fairseq.dataclass.utils import (
    convert_namespace_to_omegaconf,
    gen_parser_from_dataclass,
)
from omegaconf import DictConfig
from fairseq.models import register_model, register_model_architecture
from fairseq.models.transformer import Embedding
from fairseq.modules.transformer_sentence_encoder import init_bert_params
from fairseq.modules import  LearnedLengthControlPositionalEmbedding, SinusoidalLengthControlPositionalEmbedding
from fairseq.modules.tts_modules import DurationPredictor

import torch.nn as nn

logger = logging.getLogger(__name__)

def _mean_pooling(enc_feats, src_masks):
    # enc_feats: T x B x C
    # src_masks: B x T or None
    if src_masks is None:
        enc_feats = enc_feats.mean(0)
    else:
        src_masks = (~src_masks).transpose(0, 1).type_as(enc_feats)
        enc_feats = (
            (enc_feats / src_masks.sum(0)[None, :, None]) * src_masks[:, :, None]
        ).sum(0)
    return enc_feats

def normal_(data):
    # with FSDP, module params will be on CUDA, so we cast them back to CPU
    # so that the RNG is consistent with and without FSDP
    data.copy_(data.cpu().normal_(mean=0.0, std=0.02).to(data.device))

@register_model("transformer_speech_length")
class TransformerSpeechLengthModel(TransformerModel):
    """
    Abstract class for all nonautoregressive-based models
    """

    def __init__(self, args, encoder, decoder):
        super().__init__(args, encoder, decoder)
        self.tgt_dict = decoder.dictionary
        self.bos = decoder.dictionary.bos()
        self.eos = decoder.dictionary.eos()
        self.pad = decoder.dictionary.pad()
        self.unk = decoder.dictionary.unk()

    @staticmethod
    def add_args(parser):
        TransformerModel.add_args(parser)

        # duration prediction
        parser.add_argument(
            "--use-dur-predictor",
            action="store_true",
            help="predicting the subword's duration difference between the target and gt sentences",
        )
        # duration condition
        parser.add_argument(
            "--use-dur-condition",
            action="store_true",
            help="predicting the subword's duration difference between the target and gt sentences",
        )
        
        # length prediction
        parser.add_argument(
            "--pred-length-offset",
            action="store_true",
            help="predicting the length difference between the target and source sentences",
        )
        parser.add_argument(
            "--sg-length-pred",
            action="store_true",
            help="stop the gradients back-propagated from the length predictor",
        )
        parser.add_argument(
            "--length-loss-factor",
            type=float,
            help="weights on the length prediction loss",
        )
        parser.add_argument(
            "--use-length-phoneme",
            action="store_true",
            help="use length token appended on the left. Shoule be used together with --left-pad-source",
        )
        parser.add_argument(
            "--use-length-ratio-phoneme",
            action="store_true",
            help="use length ratio token appended on the left. Shoule be used together with --left-pad-source",
        )
        parser.add_argument(
            "--use-golden-tgt-length",
            action="store_true",
            help="use golden tgt length while inference",
        )
        parser.add_argument(
            "--test-length-ratio-phoneme",
            type=int,
            default=1,
            help="the length ratio token used while inference, 0: short, 1: normal, 2: long",
        )
        parser.add_argument(
            "--short-ratio-thre",
            type=float,
            default=0.95,
            help="tgt/src ratio threshold, shorter than this are taken as short",
        )
        parser.add_argument(
            "--long-ratio-thre",
            type=float,
            default=1.15,
            help="tgt/src ratio threshold, longer than this are taken as long",
        )

        parser.add_argument(
            '--represent-length-by-lrpe',
            default=False, action='store_true',
            help='represent target length by length ratio positional encoding'
        )

        parser.add_argument(
            '--dur-sinpos',
            default=False, action='store_true',
            help='represent dur positional encoding'
        )

        parser.add_argument(
            '--max-target-duration',
            type=int,
            default=1500,
            help='represent dur positional encoding'
        )

        parser.add_argument(
            '--quant_N',
            type=float,
            default=1,
            help='represent target length by length ratio positional encoding'
        )

        parser.add_argument(
            '--represent-length-by-ldpe', 
            default=False, action='store_true',
            help='represent target length by length difference positional encoding')

        parser.add_argument(
            '--ordinary-sinpos', 
            default=False, action='store_true',
            help='use ordinary sinusoidal positional encoding (absolute position)')
        
        parser.add_argument(
            '--decoder-dur-input', 
            default=False, action='store_true',
            help='use ordinary sinusoidal positional encoding (absolute position)')

        parser.add_argument(
            '--decoder-total-length-input', 
            default=False, action='store_true',
            help='use ordinary sinusoidal positional encoding (absolute position)')
    
    
    def load_state_dict(
        self,
        state_dict,
        strict=True,
        model_cfg: Optional[DictConfig] = None,
        args: Optional[Namespace] = None,
    ):
        if model_cfg is None and args is not None:
            logger.warn(
                "using 'args' is deprecated, please update your code to use dataclass config"
            )
            model_cfg = convert_namespace_to_omegaconf(args).model

        self.upgrade_state_dict(state_dict)
        from fairseq.checkpoint_utils import prune_state_dict
        new_state_dict = prune_state_dict(state_dict, model_cfg)

        if self.decoder.embed_length is not None or self.encoder.embed_length is not None or self.encoder.embed_length_ratio is not None or self.decoder.dur_predictor is not None:
            model_dict = self.state_dict()

            remove_keys = []
            for k, v in new_state_dict.items():
                if k not in model_dict:
                    remove_keys.append(k)

            for k in remove_keys:
                new_state_dict.pop(k)

            model_dict.update(new_state_dict)
            return super().load_state_dict(model_dict)
        else:
            return super().load_state_dict(new_state_dict, strict)

    @classmethod
    def build_decoder(cls, args, tgt_dict, embed_tokens):
        decoder = TransformerLengthDecoder(args, tgt_dict, embed_tokens)
        return decoder
    
    @classmethod
    def build_encoder(cls, args, src_dict, embed_tokens):
        return TransformerLengthEncoder(args, src_dict, embed_tokens)
        

    def forward_decoder(
        self, 
        prev_output_tokens,
        prev_subwd_durs,
        encoder_out: Optional[Dict[str, List[Tensor]]],
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        ):

        now_step = prev_output_tokens.size(1)
        if now_step == 1 and self.encoder.embed_length is not None and self.encoder.embed_length_ratio is None:
            tgt_tokens = encoder_out["tgt_tokens"][0]
            length_tgt_golden = tgt_tokens.ne(self.pad).sum(1).long()
            length_tgt_golden = length_tgt_golden.clamp(min=1, max=255)

            if encoder_out["length_out"][0] is not None:
                length_out = F.log_softmax(encoder_out["length_out"][0], -1)
            else:
                length_out = self.decoder.forward_length(normalize=True, encoder_out=encoder_out)
            length_tgt = self.decoder.forward_length_prediction(
                length_out,
                encoder_out=encoder_out,
                topk=False,
            )

            if self.decoder.use_golden_tgt_length:
                length_tgt = length_tgt_golden
                print("Use golden length".format('use golden length while inference'))

            print_tgt = length_tgt.squeeze(-1).tolist()
            print_info = [str(print_tgt[i]) for i in range(0, len(print_tgt), 5)]
            print("LEN\t{}".format(' '.join(print_info)))
            if encoder_out["length_out"][0] is not None:
                length_tgt = self.encoder.embed_length(length_tgt.unsqueeze(-1))
            
            encoder_out["length_tgt"] = [length_tgt]
            max_length = length_tgt.clamp_(min=2).max()
            return self.decoder.forward(prev_output_tokens, encoder_out=encoder_out, incremental_state=incremental_state)
        elif self.decoder.represent_length_by_lrpe is True:
            return self.decoder.forward(prev_output_tokens,prev_subwd_durs=prev_subwd_durs, encoder_out=encoder_out, incremental_state=incremental_state)
        else:
            return self.decoder.forward(prev_output_tokens,prev_subwd_durs=prev_subwd_durs, encoder_out=encoder_out, incremental_state=incremental_state)

    def forward(
        self,
        src_tokens,
        src_lengths,
        prev_output_tokens,
        tgt_tokens,
        src_speech_lengths,
        tgt_speech_lengths,
        tgt_subwd_durs,
        return_all_hiddens: bool = True,
        features_only: bool = False,
    ):
        """
        Run the forward pass for an encoder-decoder model.

        Copied from the base class, but without ``**kwargs``,
        which are not supported by TorchScript.
        """
        if self.encoder.embed_length_ratio is not None: # length ratio control
            tgt_lengs = tgt_speech_lengths
            src_lengs = src_speech_lengths
            tgt_src_len = tgt_lengs / src_lengs
            encoder_out = self.encoder(
                src_tokens, src_lengths=[tgt_src_len, src_speech_lengths], return_all_hiddens=return_all_hiddens
            )
        else: # token level length control
            encoder_out = self.encoder(
                src_tokens, src_lengths=src_speech_lengths, return_all_hiddens=return_all_hiddens
            )

        if encoder_out["length_out"][0] is not None: # token level length control
            length_out = encoder_out["length_out"][0]
        elif self.encoder.embed_length_ratio is not None:  # length ratio control
            length_out = None
        else:
            # length prediction
            length_out = self.decoder.forward_length(
                normalize=False, encoder_out=encoder_out
            )

        length_tgt = self.decoder.forward_length_prediction(
            length_out, encoder_out, tgt_tokens,tgt_speech_lengths
        )

        length_out = {"out": length_out, "tgt": length_tgt, "factor": self.decoder.length_loss_factor}

        if encoder_out["length_out"][0] is not None:
            length_tgt = self.encoder.embed_length(length_tgt.unsqueeze(-1))
        elif self.encoder.embed_length_ratio is not None:
            length_tgt = None
        encoder_out["length_tgt"] = [length_tgt]

        decoder_out_x,extra,dur_out = self.decoder(
            prev_output_tokens,
            None,
            tgt_speech_lengths,
            tgt_subwd_durs,
            encoder_out=encoder_out,
            features_only=features_only,
            src_lengths=src_lengths,
            return_all_hiddens=return_all_hiddens,
        )
        decoder_out=decoder_out_x, extra
        return decoder_out, length_out,dur_out

class TransformerLengthEncoder(TransformerEncoder):
    def __init__(self, args, dictionary, embed_tokens):
        super().__init__(args, dictionary, embed_tokens)
        embed_dim = embed_tokens.embedding_dim
        self.use_length_phoneme = getattr(args, "use_length_phoneme", False)  # decoder bos 换位length tokendd
        self.sg_length_pred = getattr(args, "sg_length_pred", False)
        self.pred_length_offset = getattr(args, "pred_length_offset", False)
        self.length_loss_factor = getattr(args, "length_loss_factor", 0.1)
        self.use_length_ratio_phoneme = getattr(args, "use_length_ratio_phoneme", False) # coarse-grained length control
        self.test_length_ratio_phoneme = getattr(args, "test_length_ratio_phoneme", 1)
        self.embed_length = None
        self.embed_length_ratio = None
        if self.use_length_phoneme and not self.use_length_ratio_phoneme: # use token-level length control
            self.embed_length = Embedding(256, embed_dim, None)
            normal_(self.embed_length.weight.data)
        elif not self.use_length_phoneme and self.use_length_ratio_phoneme:  # use ratio length control
            self.short_ratio_thre = getattr(args, "short_ratio_thre", 0.95)
            self.long_ratio_thre = getattr(args, "long_ratio_thre", 1.15)
            self.embed_length_ratio = Embedding(3, embed_dim, None) # represent short, normal and long
            normal_(self.embed_length_ratio.weight.data)
    
    def forward_torchscript(self, net_input: Dict[str, Tensor]):
        """A TorchScript-compatible version of forward.

        Encoders which use additional arguments may want to override
        this method for TorchScript compatibility.
        """
        
        return self.forward(
            src_tokens=net_input["src_tokens"],
            src_lengths=net_input["src_speech_lengths"],
        )
       
 
    def forward_scriptable(
        self,
        src_tokens,
        src_speech_lengths: Optional[torch.Tensor] = None,  # speech lengths
        return_all_hiddens: bool = False,
        token_embeddings: Optional[torch.Tensor] = None,
    ):
        """
        Args:
            src_tokens (LongTensor): tokens in the source language of shape
                `(batch, src_len)`
            src_lengths (torch.LongTensor): lengths of each source sentence of
                shape `(batch)`
            return_all_hiddens (bool, optional): also return all of the
                intermediate hidden states (default: False).
            token_embeddings (torch.Tensor, optional): precomputed embeddings
                default `None` will recompute embeddings

        Returns:
            dict:
                - **encoder_out** (Tensor): the last encoder layer's output of
                  shape `(src_len, batch, embed_dim)`
                - **encoder_padding_mask** (ByteTensor): the positions of
                  padding elements of shape `(batch, src_len)`
                - **encoder_embedding** (Tensor): the (scaled) embedding lookup
                  of shape `(batch, src_len, embed_dim)`
                - **encoder_states** (List[Tensor]): all intermediate
                  hidden states of shape `(src_len, batch, embed_dim)`.
                  Only populated if *return_all_hiddens* is True.
        """

        # compute padding mask
        encoder_padding_mask = src_tokens.eq(self.padding_idx)
        has_pads = src_tokens.device.type == "xla" or encoder_padding_mask.any()

        x, encoder_embedding = self.forward_embedding(src_tokens, token_embeddings)

        # account for padding while computing the representation
        if has_pads:
            x = x * (1 - encoder_padding_mask.unsqueeze(-1).type_as(x))

        if self.embed_length is not None: # token level length control
            len_tokens = self.embed_length(src_tokens.new(src_tokens.size(0), 1).fill_(0))
            x = torch.cat([len_tokens, x], dim=1)  # x [B,src_len+1]
            encoder_padding_mask = torch.cat([encoder_padding_mask.new(encoder_padding_mask.size(0), 1).fill_(False),
                                                encoder_padding_mask], dim=1)
        elif self.embed_length_ratio is not None:
            if type(src_speech_lengths) is list: # while training
                tgt_src_len, src_speech_lengths = src_speech_lengths
                # short:0, normal:1, long:2
                long_ids = (tgt_src_len > self.long_ratio_thre).long()*2
                normal_ids = ((tgt_src_len >= self.short_ratio_thre) & (tgt_src_len <= self.long_ratio_thre)).long()
                length_ratio_ids = tgt_src_len.new(tgt_src_len.size(0), 1).fill_(0) + normal_ids.reshape(-1, 1) + long_ids.reshape(-1, 1)
                len_tokens = self.embed_length_ratio(length_ratio_ids.long())
            else: # in inference
                if self.test_length_ratio_phoneme == -1:
                    length_ratio_ids = torch.arange(0,3).unsqueeze(0).repeat(src_speech_lengths.size(0), 1).to(src_speech_lengths.device)
                    len_tokens = self.embed_length_ratio(length_ratio_ids.long()).mean(1, True)
                else:
                    length_ratio_ids = src_speech_lengths.new(src_speech_lengths.size(0), 1).fill_(self.test_length_ratio_phoneme)
                    len_tokens = self.embed_length_ratio(length_ratio_ids.long())
            x = torch.cat([len_tokens, x], dim=1)
            encoder_padding_mask = torch.cat([encoder_padding_mask.new(encoder_padding_mask.size(0), 1).fill_(False),
                                                encoder_padding_mask], dim=1)


        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        encoder_states = []

        if return_all_hiddens:
            encoder_states.append(x)

        # encoder layers
        for layer in self.layers:
            x = layer(
                x, encoder_padding_mask=encoder_padding_mask if has_pads else None
            )
            if return_all_hiddens:
                assert encoder_states is not None
                encoder_states.append(x)

        if self.layer_norm is not None:
            x = self.layer_norm(x)

        # The Pytorch Mobile lite interpreter does not supports returning NamedTuple in
        # `forward` so we use a dictionary instead.
        # TorchScript does not support mixed values so the values are all lists.
        # The empty list is equivalent to None.
        src_lengths = (
            src_tokens.ne(self.padding_idx)
            .sum(dim=1, dtype=torch.int32)
            .reshape(-1, 1)
            .contiguous()
        )
        length_out = None
        if self.embed_length is not None:  # token level length control
            length_feats = x[0, :, :] # B x C
            if self.sg_length_pred:
                length_feats = length_feats.detach()
            length_out = F.linear(length_feats, self.embed_length.weight)
            length_out[:, 0] += float('-inf')
            x = x[1:, :, :]
            encoder_padding_mask = encoder_padding_mask[:, 1:]

        return {
            "encoder_out": [x],  # T x B x C
            "encoder_padding_mask": [encoder_padding_mask],  # B x T
            "encoder_embedding": [encoder_embedding],  # B x T x C
            "encoder_states": encoder_states,  # List[T x B x C]
            "src_tokens": [src_tokens],
            "src_lengths": [src_lengths],
            "src_speech_lengths": [src_speech_lengths],
            "length_out": [length_out],
        }
    
    @torch.jit.export
    def reorder_encoder_out(self, encoder_out: Dict[str, List[Tensor]], new_order):
        """
        Reorder encoder output according to *new_order*.

        Args:
            encoder_out: output from the ``forward()`` method
            new_order (LongTensor): desired order

        Returns:
            *encoder_out* rearranged according to *new_order*
        """
        
        if len(encoder_out["encoder_out"]) == 0:
            new_encoder_out = []
        else:
            new_encoder_out = [encoder_out["encoder_out"][0].index_select(1, new_order)]
        if len(encoder_out["encoder_padding_mask"]) == 0:
            new_encoder_padding_mask = []
        else:
            new_encoder_padding_mask = [
                encoder_out["encoder_padding_mask"][0].index_select(0, new_order)
            ]
        if len(encoder_out["encoder_embedding"]) == 0:
            new_encoder_embedding = []
        else:
            new_encoder_embedding = [
                encoder_out["encoder_embedding"][0].index_select(0, new_order)
            ]

        if len(encoder_out["src_tokens"]) == 0:
            src_tokens = []
        else:
            src_tokens = [(encoder_out["src_tokens"][0]).index_select(0, new_order)]

        if len(encoder_out["src_lengths"]) == 0:
            src_lengths = []
        else:
            src_lengths = [(encoder_out["src_lengths"][0]).index_select(0, new_order)]
        
        if len(encoder_out["src_speech_lengths"]) == 0:
            src_speech_lengths = []
        else:
            src_speech_lengths = [(encoder_out["src_speech_lengths"][0]).index_select(0, new_order)]

        encoder_states = encoder_out["encoder_states"]
        if len(encoder_states) > 0:
            for idx, state in enumerate(encoder_states):
                encoder_states[idx] = state.index_select(1, new_order)
        
        if encoder_out["length_out"][0] is None:
            length_out = [None]
        else:
            length_out = [(encoder_out["length_out"][0]).index_select(0, new_order)]


        if "tgt_tokens" not in encoder_out or len(encoder_out["tgt_tokens"]) == 0 or encoder_out["tgt_tokens"][0] is None:
            tgt_tokens = []
        else:
            tgt_tokens = [(encoder_out["tgt_tokens"][0]).index_select(0, new_order)]

        return {
            "encoder_out": new_encoder_out,  # T x B x C
            "encoder_padding_mask": new_encoder_padding_mask,  # B x T
            "encoder_embedding": new_encoder_embedding,  # B x T x C
            "encoder_states": encoder_states,  # List[T x B x C]
            "src_tokens": src_tokens,  # B x T
            "src_lengths": src_lengths,  # B x 1
            "src_speech_lengths": src_speech_lengths,  # B x 1
            "length_out": length_out,
            "tgt_tokens": tgt_tokens,
        }


class TransformerLengthDecoder(TransformerDecoder):
    def __init__(self, args, dictionary, embed_tokens, no_encoder_attn=False):
        super().__init__(args, dictionary, embed_tokens, no_encoder_attn)
        
        self.dictionary = dictionary
        self.bos = dictionary.bos()
        self.unk = dictionary.unk()
        self.eos = dictionary.eos()

        self.encoder_embed_dim = args.encoder_embed_dim
        self.sg_length_pred = getattr(args, "sg_length_pred", False)
        self.pred_length_offset = getattr(args, "pred_length_offset", False)
        self.length_loss_factor = getattr(args, "length_loss_factor", 0.1)
        self.dur_loss_factor = getattr(args, "dur_loss_factor", 1.0)
        self.use_length_phoneme = getattr(args, "use_length_phoneme", False)
        self.use_length_ratio_phoneme = getattr(args, "use_length_ratio_phoneme", False)  # 另一种长度控制，不精细分类，x y 相对长度， 在source端加
        self.use_golden_tgt_length = getattr(args, "use_golden_tgt_length", False)
        self.use_dur_predictor = getattr(args, "use_dur_predictor", False)
        self.use_dur_condition = getattr(args, "use_dur_condition", False)
        self.embed_length = None
        left_pad=False
        # add positional embedding with langth control
        embed_dim = args.decoder_embed_dim
        padding_idx = embed_tokens.padding_idx
        self.ordinary_sinpos = args.ordinary_sinpos  # original sinusoidal PE
        self.represent_length_by_lrpe = args.represent_length_by_lrpe  # relative PE (pos/len)
        self.dur_sinpos = args.dur_sinpos  # relative PE (pos/len)
        self.represent_length_by_ldpe = args.represent_length_by_ldpe  # difference PE (len-pos)
        self.quant_N=args.quant_N
        self.decoder_dur_input = args.decoder_dur_input
        self.decoder_total_length_input = args.decoder_total_length_input
        self.decoder_learned_pos = args.decoder_learned_pos

        self.embed_positions_original = PositionalEmbedding(
            args.max_target_positions, embed_dim, padding_idx,
            left_pad=left_pad,
            learned=args.decoder_learned_pos,
        ) if not args.no_token_positional_embeddings and self.ordinary_sinpos else None

        self.embed_positions_lrpe = PositionalEmbedding(
            args.max_target_positions, embed_dim, padding_idx,
            left_pad=left_pad,
            learned=args.decoder_learned_pos,
            quant_N=self.quant_N
        ) if not args.no_token_positional_embeddings and self.represent_length_by_lrpe else None


        self.embed_positions_ldpe = PositionalEmbedding(
            args.max_target_positions, embed_dim, padding_idx,
            left_pad=left_pad,
            learned=args.decoder_learned_pos,
        ) if not args.no_token_positional_embeddings and self.represent_length_by_ldpe else None

        self.embed_positions_dur = PositionalEmbedding(
            args.max_target_duration, embed_dim, padding_idx,
            left_pad=left_pad,
            learned=args.decoder_learned_pos
        ) if not args.no_token_positional_embeddings and self.dur_sinpos else None

        if not self.use_length_phoneme and not self.use_length_ratio_phoneme:  # 引入length 长度
            self.embed_length = Embedding(256, self.encoder_embed_dim, None)
            normal_(self.embed_length.weight.data)

        if self.use_dur_predictor and not self.use_dur_condition:
            self.dur_predictor = DurationPredictor(
                args.decoder_embed_dim,
                n_chans=args.decoder_embed_dim,
                dropout_rate=0.5, padding='SAME',
                kernel_size=3)
        elif self.use_dur_predictor and self.use_dur_condition:
            self.dur_predictor = DurationPredictor(
                args.decoder_embed_dim * 2,
                n_chans=args.decoder_embed_dim,
                dropout_rate=0.5, padding='SAME',
                kernel_size=3)
            self.embed_duration = Embedding(1500, self.encoder_embed_dim, None)
            normal_(self.embed_duration.weight.data)
        if self.decoder_total_length_input:
            self.embed_decoder_total_length = Embedding(1500, self.encoder_embed_dim, None)
            normal_(self.embed_decoder_total_length.weight.data)
        if self.decoder_dur_input:
            self.embed_decoder_dur = Embedding(1500, self.encoder_embed_dim, None)
            normal_(self.embed_decoder_dur.weight.data)


    def forward(
        self,
        prev_output_tokens,
        prev_subwd_durs = None,
        tgt_speech_lengths = None,
        tgt_subwd_durs = None,
        encoder_out: Optional[Dict[str, List[Tensor]]] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        features_only: bool = False,
        full_context_alignment: bool = False,
        alignment_layer: Optional[int] = None,
        alignment_heads: Optional[int] = None,
        src_lengths: Optional[Any] = None,
        return_all_hiddens: bool = False,
    ):
        """
        Args:
            prev_output_tokens (LongTensor): previous decoder outputs of shape
                `(batch, tgt_len)`, for teacher forcing
            encoder_out (optional): output from the encoder, used for
                encoder-side attention, should be of size T x B x C
            incremental_state (dict): dictionary used for storing state during
                :ref:`Incremental decoding`
            features_only (bool, optional): only return features without
                applying output layer (default: False).
            full_context_alignment (bool, optional): don't apply
                auto-regressive mask to self-attention (default: False).

        Returns:
            tuple:
                - the decoder's output of shape `(batch, tgt_len, vocab)`
                - a dictionary with any model-specific outputs
        """
        x, extra, dur_out = self.extract_features(
            prev_output_tokens,
            prev_subwd_durs,
            tgt_speech_lengths,
            tgt_subwd_durs,
            encoder_out=encoder_out,
            incremental_state=incremental_state,
            full_context_alignment=full_context_alignment,
            alignment_layer=alignment_layer,
            alignment_heads=alignment_heads,
        ) # x (B,seq_len,512)
        
        if not features_only:
            x = self.output_layer(x) # x (batch, tgt_len, vocab)

        return x, extra, dur_out

    def extract_features(
        self,
        prev_output_tokens,
        prev_subwd_durs,
        tgt_speech_lengths,
        tgt_subwd_durs,
        encoder_out: Optional[Dict[str, List[Tensor]]],
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        full_context_alignment: bool = False,
        alignment_layer: Optional[int] = None,
        alignment_heads: Optional[int] = None,
    ):
        return self.extract_features_scriptable(
            prev_output_tokens,
            prev_subwd_durs,
            tgt_speech_lengths,
            tgt_subwd_durs,
            encoder_out,
            incremental_state,
            full_context_alignment,
            alignment_layer,
            alignment_heads,
        )

    def forward_length(self, normalize, encoder_out):
        enc_feats = encoder_out["encoder_out"][0]  # T x B x C
        if len(encoder_out["encoder_padding_mask"]) > 0:
            src_masks = encoder_out["encoder_padding_mask"][0]  # B x T
        else:
            src_masks = None
        enc_feats = _mean_pooling(enc_feats, src_masks)
        if self.sg_length_pred:
            enc_feats = enc_feats.detach()
        length_out = F.linear(enc_feats, self.embed_length.weight)
        return F.log_softmax(length_out, -1) if normalize else length_out
    
    def forward_length_prediction(self, length_out, encoder_out, tgt_tokens=None,tgt_speech_lengths=None, topk=False):
        enc_feats = encoder_out["encoder_out"][0]  # T x B x C
        if len(encoder_out["encoder_padding_mask"]) > 0:
            src_masks = encoder_out["encoder_padding_mask"][0]  # B x T
        else:
            src_masks = None
        if self.pred_length_offset:
            if src_masks is None:
                src_lengs = enc_feats.new_ones(enc_feats.size(1)).fill_(
                    enc_feats.size(0)
                )
            else:
                src_lengs = (~src_masks).transpose(0, 1).type_as(enc_feats).sum(0)
            src_lengs = src_lengs.long()

        if tgt_tokens is not None:  # train
            # obtain the length target
            tgt_lengs = tgt_tokens.ne(self.padding_idx).sum(1).long()
            tgt_lengs=tgt_speech_lengths # add for target phoneme length

            if self.pred_length_offset:
                length_tgt = tgt_lengs - src_lengs + 128
            else:
                length_tgt = tgt_lengs
            length_tgt = length_tgt.clamp(min=1, max=255)

        else: # infer
            # predict the length target (greedy for now)
            # TODO: implementing length-beam
            if topk:
                bsz_bm = length_out.size(0)
                # only works if beam size is set to 5
                bm = 5
                bsz = bsz_bm // bm 
                bm_offsets = (torch.arange(0, bsz) * bm).to(length_out.device)
                length_out = torch.index_select(length_out, dim=0, index=bm_offsets)
                pred_lengs = length_out.topk(5)[1].view(-1)
            else:  
                pred_lengs = length_out.max(-1)[1]
            if self.pred_length_offset:
                length_tgt = pred_lengs - 128 + src_lengs
            else:
                length_tgt = pred_lengs

        return length_tgt

    def extract_features_scriptable(
        self,
        prev_output_tokens,
        prev_subwd_durs,
        tgt_speech_lengths,
        tgt_subwd_durs,
        encoder_out: Optional[Dict[str, List[Tensor]]],
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        full_context_alignment: bool = False,
        alignment_layer: Optional[int] = None,
        alignment_heads: Optional[int] = None,
    ):
        """
        Similar to *forward* but only return features.

        Includes several features from "Jointly Learning to Align and
        Translate with Transformer Models" (Garg et al., EMNLP 2019).

        Args:
            full_context_alignment (bool, optional): don't apply
                auto-regressive mask to self-attention (default: False).
            alignment_layer (int, optional): return mean alignment over
                heads at this layer (default: last layer).
            alignment_heads (int, optional): only average alignment over
                this many heads (default: all heads).

        Returns:
            tuple:
                - the decoder's features of shape `(batch, tgt_len, embed_dim)`
                - a dictionary with any model-specific outputs
        """

        bs, slen = prev_output_tokens.size()
        if alignment_layer is None:
            alignment_layer = self.num_layers - 1

        enc: Optional[Tensor] = None
        padding_mask: Optional[Tensor] = None
        dur_condition = None
        dur_emb = None
        length_emb = None

        if encoder_out is not None and len(encoder_out["encoder_out"]) > 0:
            enc = encoder_out["encoder_out"][0]
            assert (
                enc.size()[1] == bs
            ), f"Expected enc.shape == (t, {bs}, c) got {enc.shape}"
        if encoder_out is not None and len(encoder_out["encoder_padding_mask"]) > 0:
            padding_mask = encoder_out["encoder_padding_mask"][0]
        
        if encoder_out is not None and "length_tgt" in encoder_out:
            if self.embed_length is not None:
                length_tgt = encoder_out["length_tgt"][0].unsqueeze(-1)
                length_emb = self.embed_length(length_tgt)
            else:
                length_emb = encoder_out["length_tgt"][0]
        

        # embed positions
        positions = None
        positions_as_res = None

        # if self.embed_positions is not None:
        #     positions = self.embed_positions(
        #         prev_output_tokens, incremental_state=incremental_state
        #     )
        
        # if incremental_state is not None:  # incremental_state None
        #     prev_output_tokens = prev_output_tokens[:, -1:]
        #     if positions is not None:
        #         positions = positions[:, -1:]

        if tgt_speech_lengths is None:  # infer
            prev_subwd_durs=prev_subwd_durs.to(prev_output_tokens.device)
            tgt_speech_lengths = encoder_out["src_speech_lengths"][0]  # 可通过调整ratio 超参控制语速比例
            
            tgt_speech_lengths=(tgt_speech_lengths*2.0).floor().clamp(min=0,max=1499)
           

        
        '''
            add new positional embedding for length control
        '''
        if self.ordinary_sinpos and self.decoder_learned_pos is False:
            positions_orig = self.embed_positions_original(
                prev_output_tokens,
                None,
                incremental_state=incremental_state,
            ) if self.embed_positions_original is not None else None
            if incremental_state is not None:
                positions = positions_orig[:, -1:]
            else:
                positions = positions_orig
        elif self.ordinary_sinpos and self.decoder_learned_pos is True:
            positions_orig = self.embed_positions_original(
                prev_output_tokens,
                incremental_state=incremental_state,
            ) if self.embed_positions_original is not None else None
            if incremental_state is not None:
                positions = positions_orig[:, -1:]
            else:
                positions = positions_orig
        
        if self.embed_positions_dur and self.decoder_learned_pos is False:
            if incremental_state is None:
                zeros=torch.zeros([tgt_subwd_durs.shape[0],1]).to(tgt_subwd_durs.device)
                _shift_tgt_subwd_durs=torch.cat((zeros,tgt_subwd_durs[:,:-1]),1)
                positions_dur = self.embed_positions_dur(
                    prev_output_tokens,
                    _shift_tgt_subwd_durs,
                    incremental_state=incremental_state,
                    length=tgt_speech_lengths,
                    sinpostype= 'duration',  # ratio
                ) if self.embed_positions_dur is not None else None
                positions_tmp = positions_dur
        
            elif incremental_state is not None:
                positions_dur = self.embed_positions_dur(
                prev_output_tokens,
                prev_subwd_durs,
                incremental_state=incremental_state,
                length=tgt_speech_lengths,
                sinpostype= 'duration',  # ratio
                ) if self.embed_positions_dur is not None else None
                positions_tmp = positions_dur.view(positions_dur.size(0), 1, -1)
            else:
                positions_tmp = positions_dur
            positions = positions + positions_tmp if positions is not None else positions_tmp

        elif self.embed_positions_dur and self.decoder_learned_pos is True:
            if incremental_state is None:
                zeros=torch.zeros([tgt_subwd_durs.shape[0],1]).to(tgt_subwd_durs.device)
                _shift_tgt_subwd_durs=torch.cat((zeros,tgt_subwd_durs[:,:-1]),1)
                
                positions_dur = self.embed_positions_dur(
                    prev_output_tokens,
                    incremental_state=incremental_state,
                    length = None,
                    positions= _shift_tgt_subwd_durs, # dur
                    postype = "duration"
                ) if self.embed_positions_dur is not None else None
                positions_tmp = positions_dur
        
            elif incremental_state is not None:
                positions_dur = self.embed_positions_dur(
                    prev_output_tokens,
                    incremental_state=incremental_state,
                    length = None,
                    positions= prev_subwd_durs, # dur
                    postype = "duration"
                ) if self.embed_positions_dur is not None else None
                positions_tmp = positions_dur.view(positions_dur.size(0), 1, -1)
            else:
                positions_tmp = positions_dur
            positions = positions + positions_tmp if positions is not None else positions_tmp

        if self.represent_length_by_lrpe and self.decoder_learned_pos is False:
            if incremental_state is None:
                zeros=torch.zeros([tgt_subwd_durs.shape[0],1]).to(tgt_subwd_durs.device)
                _shift_tgt_subwd_durs=torch.cat((zeros,tgt_subwd_durs[:,:-1]),1)
                positions_lrpe = self.embed_positions_lrpe(
                    prev_output_tokens,
                    _shift_tgt_subwd_durs,
                    incremental_state=incremental_state,
                    length=tgt_speech_lengths,
                    sinpostype= 'relative',  # ratio
                ) if self.embed_positions_lrpe is not None else None
                positions_tmp = positions_lrpe
        
            elif incremental_state is not None:
                positions_lrpe = self.embed_positions_lrpe(
                prev_output_tokens,
                prev_subwd_durs,
                incremental_state=incremental_state,
                length=tgt_speech_lengths,
                sinpostype= 'relative',  # ratio
                ) if self.embed_positions_lrpe is not None else None
                positions_tmp = positions_lrpe.view(positions_lrpe.size(0), 1, -1)
            else:
                positions_tmp = positions_lrpe
            positions = positions + positions_tmp if positions is not None else positions_tmp

        elif self.represent_length_by_lrpe and self.decoder_learned_pos is True:
            if incremental_state is None:
                zeros=torch.zeros([tgt_subwd_durs.shape[0],1]).to(tgt_subwd_durs.device)
                _shift_tgt_subwd_durs=torch.cat((zeros,tgt_subwd_durs[:,:-1]),1)
                 
                positions_lrpe = self.embed_positions_lrpe(
                    prev_output_tokens,
                    incremental_state=incremental_state,
                    length = tgt_speech_lengths,
                    positions= _shift_tgt_subwd_durs, # dur
                    postype = "relative"
                ) if self.embed_positions_lrpe is not None else None
                positions_tmp = positions_lrpe
        
            elif incremental_state is not None:
                positions_lrpe = self.embed_positions_lrpe(
                    prev_output_tokens,
                    incremental_state=incremental_state,
                    length = tgt_speech_lengths,
                    positions= prev_subwd_durs, # dur
                    postype = "relative"
                ) if self.embed_positions_lrpe is not None else None
                positions_tmp = positions_lrpe.view(positions_lrpe.size(0), 1, -1)
            else:
                positions_tmp = positions_lrpe
            positions = positions + positions_tmp if positions is not None else positions_tmp


        if incremental_state is not None:  # infer
            prev_output_tokens = prev_output_tokens[:, -1:]
        
     
        self_attn_padding_mask: Optional[Tensor] = None
        if self.cross_self_attention or prev_output_tokens.eq(self.padding_idx).any():
            self_attn_padding_mask = prev_output_tokens.eq(self.padding_idx)


        if length_emb is not None and (incremental_state is None or incremental_state == {}):  # token-level length control
            if prev_output_tokens.size(1) == 1:  # length token replace <BOS>
                prev_emb = length_emb
            else:
                prev_output_tokens = prev_output_tokens[:, 1:]
                old_prev_emb = self.embed_tokens(prev_output_tokens)
                prev_emb = torch.cat([length_emb, old_prev_emb], dim=1)
        else:
            prev_emb = self.embed_tokens(prev_output_tokens)
        

        
            
        x = self.embed_scale * prev_emb

        if self.quant_noise is not None:
            x = self.quant_noise(x)

        if self.project_in_dim is not None:
            x = self.project_in_dim(x)

        if positions is not None:
            x += positions

        if self.decoder_total_length_input:
            _dur=tgt_speech_lengths.unsqueeze(1)
            decoder_total_length_input_emb = self.embed_decoder_total_length(_dur)
            x+=decoder_total_length_input_emb
        
        if self.decoder_dur_input:
            if incremental_state is None:
                _dur=torch.cumsum(tgt_subwd_durs, dim=1)
            else:
                _dur=prev_subwd_durs
            decoder_dur_input_emb = self.embed_decoder_dur(_dur)
            x+=decoder_dur_input_emb

        if self.layernorm_embedding is not None:
            x = self.layernorm_embedding(x)

        x = self.dropout_module(x)

        if positions_as_res is not None:
            #I'm not sure this process is necessarry
            positions_as_res = F.dropout(positions_as_res, p=self.dropout, training=self.training)
            positions_as_res = positions_as_res.transpose(0, 1)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        # decoder layers
        attn: Optional[Tensor] = None
        inner_states: List[Optional[Tensor]] = [x]
        for idx, layer in enumerate(self.layers):
            if incremental_state is None and not full_context_alignment:
                self_attn_mask = self.buffered_future_mask(x)
            else:
                self_attn_mask = None
           
            x, layer_attn, _ = layer(
                x,
                enc,
                padding_mask,
                incremental_state,
                self_attn_mask=self_attn_mask,
                self_attn_padding_mask=self_attn_padding_mask,
                need_attn=bool((idx == alignment_layer)),
                need_head_weights=bool((idx == alignment_layer)),
            )
            inner_states.append(x)
            if layer_attn is not None and idx == alignment_layer:
                attn = layer_attn.float().to(x)

        if attn is not None:
            if alignment_heads is not None:
                attn = attn[:alignment_heads]

            # average probabilities over heads
            attn = attn.mean(dim=0)

        if self.layer_norm is not None:
            x = self.layer_norm(x)

        # T x B x C -> B x T x C
        x = x.transpose(0, 1)
        if self.project_out_dim is not None:
            x = self.project_out_dim(x)
        padding_idx = 0
        if incremental_state is None: # train
            _x_detach = x.detach() # [B,T,H]
            if self.use_dur_condition:
                mask = tgt_subwd_durs.ne(padding_idx).int()
                cusum_subwd_durs=torch.cumsum(tgt_subwd_durs, dim=1) * mask
                _dur=(tgt_speech_lengths.unsqueeze(1)-cusum_subwd_durs) * mask
                dur_condition= self.embed_duration(_dur)
                x_detach=torch.cat((_x_detach,dur_condition),dim=2)
            else:
                x_detach=_x_detach
            dur_pred= self.dur_predictor(x_detach, tgt_subwd_durs == 0) # dur_pred (B, tgt_len)

            dur_out={
                "pred_dur": dur_pred,
                "tgt_dur": tgt_subwd_durs,
                "factor": self.dur_loss_factor
            }
    
        else: # dur
            _x_detach = x.detach()
            src_tokens = encoder_out["src_tokens"][0]
            if self.use_dur_condition:
                _dur=(tgt_speech_lengths.unsqueeze(1)-prev_subwd_durs).clamp(min=0, max=1500)
                dur_condition= self.embed_duration(_dur)
                x_detach=torch.cat((_x_detach,dur_condition),dim=2)
            else:
                x_detach=_x_detach           
            dur_pred = self.dur_predictor.inference(x_detach)
            dur_out={
                "pred_dur": dur_pred,
                "tgt_dur": tgt_subwd_durs,
                "factor": self.dur_loss_factor
            }
            
        return x, {"attn": [attn], "inner_states": inner_states}, dur_out
    


def PositionalEmbedding(num_embeddings: int, embedding_dim:int, padding_idx:int, left_pad:bool=True, learned=False,quant_N=1):
    if learned:
        m = LearnedLengthControlPositionalEmbedding(num_embeddings + padding_idx + 1, embedding_dim, padding_idx)
        nn.init.normal_(m.weight, mean=0, std=embedding_dim ** -0.5)
        nn.init.constant_(m.weight[padding_idx], 0)
    else:
        m = SinusoidalLengthControlPositionalEmbedding(embedding_dim, padding_idx, left_pad, num_embeddings + padding_idx + 1, quant_N)
    return m

@register_model_architecture("transformer_speech_length", "transformer_speech_length")
def base_architecture(args):
    args.encoder_embed_path = getattr(args, "encoder_embed_path", None)
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 512)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 2048)
    args.encoder_layers = getattr(args, "encoder_layers", 6)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 8)
    args.encoder_normalize_before = getattr(args, "encoder_normalize_before", False)
    args.encoder_learned_pos = getattr(args, "encoder_learned_pos", False)
    args.decoder_embed_path = getattr(args, "decoder_embed_path", None)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", args.encoder_embed_dim)
    args.decoder_ffn_embed_dim = getattr(
        args, "decoder_ffn_embed_dim", args.encoder_ffn_embed_dim
    )
    args.decoder_layers = getattr(args, "decoder_layers", 6)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 8)
    args.decoder_normalize_before = getattr(args, "decoder_normalize_before", False)
    args.decoder_learned_pos = getattr(args, "decoder_learned_pos", False)
    args.attention_dropout = getattr(args, "attention_dropout", 0.0)
    args.activation_dropout = getattr(args, "activation_dropout", 0.0)
    args.activation_fn = getattr(args, "activation_fn", "relu")
    args.dropout = getattr(args, "dropout", 0.1)
    args.adaptive_softmax_cutoff = getattr(args, "adaptive_softmax_cutoff", None)
    args.adaptive_softmax_dropout = getattr(args, "adaptive_softmax_dropout", 0)
    args.share_decoder_input_output_embed = getattr(
        args, "share_decoder_input_output_embed", False
    )
    args.share_all_embeddings = getattr(args, "share_all_embeddings", False)
    args.no_token_positional_embeddings = getattr(
        args, "no_token_positional_embeddings", False
    )
    args.adaptive_input = getattr(args, "adaptive_input", False)
    args.no_cross_attention = getattr(args, "no_cross_attention", False)
    args.cross_self_attention = getattr(args, "cross_self_attention", False)

    args.decoder_output_dim = getattr(
        args, "decoder_output_dim", args.decoder_embed_dim
    )
    args.decoder_input_dim = getattr(args, "decoder_input_dim", args.decoder_embed_dim)

    args.no_scale_embedding = getattr(args, "no_scale_embedding", False)
    args.layernorm_embedding = getattr(args, "layernorm_embedding", False)
    args.tie_adaptive_weights = getattr(args, "tie_adaptive_weights", False)
    args.checkpoint_activations = getattr(args, "checkpoint_activations", False)
    args.offload_activations = getattr(args, "offload_activations", False)
    if args.offload_activations:
        args.checkpoint_activations = True
    args.encoder_layers_to_keep = getattr(args, "encoder_layers_to_keep", None)
    args.decoder_layers_to_keep = getattr(args, "decoder_layers_to_keep", None)
    args.encoder_layerdrop = getattr(args, "encoder_layerdrop", 0)
    args.decoder_layerdrop = getattr(args, "decoder_layerdrop", 0)
    args.quant_noise_pq = getattr(args, "quant_noise_pq", 0)
    args.quant_noise_pq_block_size = getattr(args, "quant_noise_pq_block_size", 8)
    args.quant_noise_scalar = getattr(args, "quant_noise_scalar", 0)

    # --- special arguments ---
    args.sg_length_pred = getattr(args, "sg_length_pred", False)
    args.pred_length_offset = getattr(args, "pred_length_offset", False)
    args.length_loss_factor = getattr(args, "length_loss_factor", 0.1)
    args.pred_dur_infer = getattr(args, "pred_dur_infer",True)
    


@register_model_architecture("transformer_speech_length", "transformer_speech_length_neu_zh_en")
def transformer_speech_length_iwslt_de_en(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 512)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 1024)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 4)
    args.encoder_layers = getattr(args, "encoder_layers", 6)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 512)
    args.decoder_ffn_embed_dim = getattr(args, "decoder_ffn_embed_dim", 1024)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 4)
    args.decoder_layers = getattr(args, "decoder_layers", 6)
    base_architecture(args)

@register_model_architecture("transformer_speech_length", "transformer_speech_length_neu_zh_en_2")
def transformer_wmt_en_de(args):
    base_architecture(args)