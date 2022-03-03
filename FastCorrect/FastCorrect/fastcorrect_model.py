# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn.functional as F
from fairseq import utils
from fastcorrect_generator import DecoderOut
from fairseq.models import register_model, register_model_architecture
from fairseq.models.nat import FairseqNATDecoder, FairseqNATModel, ensemble_decoder, ensemble_encoder
from fairseq.models.transformer import Embedding
from fairseq.modules.transformer_sentence_encoder import init_bert_params
from fairseq.modules import FairseqDropout
from torch import Tensor
from fairseq.models.transformer import (
    TransformerEncoder,
)

import logging
logger = logging.getLogger(__name__)

def Embeddingright(num_embeddings, embedding_dim, padding_idx):
    m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
    nn.init.normal_(m.weight, mean=0, std=embedding_dim ** -0.5)
    if padding_idx is not None:
        nn.init.constant_(m.weight[padding_idx], 0)
    return m

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


def _argmax(x, dim):
    return (x == x.max(dim, keepdim=True)[0]).type_as(x)


@register_model("fastcorrect")
class FastCorrectModel(FairseqNATModel):
    def __init__(self, args, encoder, decoder):
        super().__init__(args, encoder, decoder)


        self.to_be_edited_size = getattr(args, "to_be_edited_size", 1)

        if getattr(args, 'assist_edit_loss', False):
            print("add assist edit loss!")
            self.assist_edit_loss = True
        else:
            self.assist_edit_loss = False

        self.werdur_max_predict = getattr(args, 'werdur_max_predict', 5.0)
        print("werdur_max_predict: ", self.werdur_max_predict)

        self.werdur_loss_type = getattr(args, 'werdur_loss_type', 'l2')

        print("werdur_loss_type: ", self.werdur_loss_type)
        if self.werdur_loss_type == 'l2':
            self.werdur_loss_func = F.mse_loss
        elif self.werdur_loss_type == 'log_l2':
            self.werdur_loss_func = self.log_mse_loss
        elif self.werdur_loss_type == 'l1':
            self.werdur_loss_func = F.l1_loss
        elif self.werdur_loss_type == 'log_l1':
            self.werdur_loss_func = self.log_l1_loss

        else:
            raise ValueError("Unsupported werdur_loss_type")

    def log_mse_loss(self, hypo, ref, reduction='none'):
        hypo = torch.exp(hypo) - 1.0
        return F.mse_loss(hypo, ref, reduction=reduction)


    def log_l1_loss(self, hypo, ref, reduction='none'):
        hypo = torch.exp(hypo) - 1.0
        return F.l1_loss(hypo, ref, reduction=reduction)


    @property
    def allow_length_beam(self):
        return True

    @staticmethod
    def add_args(parser):
        FairseqNATModel.add_args(parser)

        # length prediction
        parser.add_argument(
            "--assist-edit-loss",
            action="store_true",
            default=False,
            help="whether to use assist edit loss",
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
            "--edit-emb-dim",
            type=int,
            help="dimension of edit emb",
        )
        parser.add_argument(
            "--to-be-edited-size",
            type=int,
            help="size of to be edited (2 for edited or not, 4 or insert/delete/change/not do",
        )
        parser.add_argument(
            "--werdur-max-predict",
            type=float,
            help="max value of werdur",
        )
        parser.add_argument(
            "--werdur-loss-type",
            type=str,
            help="type of werdur loss",
        )

    @classmethod
    def build_decoder(cls, args, tgt_dict, embed_tokens):
        decoder = FastCorrectDecoder(args, tgt_dict, embed_tokens)
        if getattr(args, "apply_bert_init", False):
            decoder.apply(init_bert_params)
        return decoder

    @classmethod
    def build_encoder(cls, args, src_dict, embed_tokens):
        encoder = FastCorrectEncoder(args, src_dict, embed_tokens)
        if getattr(args, "apply_bert_init", False):
            encoder.apply(init_bert_params)
        return encoder

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path,
        checkpoint_file="model.pt",
        data_name_or_path=".",
        **kwargs,
    ):
        """
        Load a :class:`~fairseq.models.FairseqModel` from a pre-trained model
        file. Downloads and caches the pre-trained model file if needed.

        The base implementation returns a
        :class:`~fairseq.hub_utils.GeneratorHubInterface`, which can be used to
        generate translations or sample from language models. The underlying
        :class:`~fairseq.models.FairseqModel` can be accessed via the
        *generator.models* attribute.

        Other models may override this to implement custom hub interfaces.

        Args:
            model_name_or_path (str): either the name of a pre-trained model to
                load or a path/URL to a pre-trained model state dict
            checkpoint_file (str, optional): colon-separated list of checkpoint
                files in the model archive to ensemble (default: 'model.pt')
            data_name_or_path (str, optional): point args.data to the archive
                at the given path/URL. Can start with '.' or './' to reuse the
                model archive path.
        """
        import hub_utils_fc

        x = hub_utils_fc.from_pretrained(
            model_name_or_path,
            checkpoint_file,
            data_name_or_path,
            archive_map=cls.hub_models(),
            **kwargs,
        )

        cls.upgrade_args(x["args"])

        logger.info(x["args"])
        return hub_utils_fc.GeneratorHubInterface(x["args"], x["task"], x["models"])


    def _compute_nll_loss(
        self, outputs, targets, masks=None, label_smoothing=0.0, name="loss", factor=1.0
    ):
        """
        outputs: batch x len x d_model
        targets: batch x len
        masks:   batch x len

        policy_logprob: if there is some policy
            depends on the likelihood score as rewards.
        """

        def mean_ds(x: Tensor, dim=None) -> Tensor:
            return (
                x.float().mean().type_as(x)
                if dim is None
                else x.float().mean(dim).type_as(x)
            )




        
        if masks is not None:
            outputs, targets = outputs[masks], targets[masks]

        if masks is not None and not masks.any():
            nll_loss = torch.tensor(0)
            loss = nll_loss
        else:
            logits = F.log_softmax(outputs, dim=-1)
            if targets.dim() == 1:
                losses = F.nll_loss(logits, targets.to(logits.device), reduction="none")

            else:  # soft-labels
                losses = F.kl_div(logits, targets.to(logits.device), reduction="none")
                losses = losses.sum(-1)


            nll_loss = mean_ds(losses)
            if label_smoothing > 0:
                loss = (
                    nll_loss * (1 - label_smoothing) - mean_ds(logits) * label_smoothing
                )
            else:
                loss = nll_loss

        loss = loss * factor
        return {"name": name, "loss": loss, "nll_loss": nll_loss, "factor": factor}, None

    def forward_encoder(self, encoder_inputs):
        src_tokens, src_lengths = encoder_inputs
        #attn_mask = None
        return self.encoder(src_tokens, src_lengths=src_lengths)

    def forward(
        self, src_tokens, src_lengths, prev_output_tokens, tgt_tokens, wer_dur=None, to_be_edited=None, for_wer_gather=None, **kwargs
    ):
        # encoding
        # attn_mask = None
        encoder_out = self.encoder(src_tokens, src_lengths=src_lengths, **kwargs)

        wer_dur_pred, to_be_edited_pred, closest_pred = self.decoder.forward_wer_dur_and_tbe(
            normalize=False, encoder_out=encoder_out
        )


        wer_dur = wer_dur.type_as(wer_dur_pred).clamp(0.0, self.werdur_max_predict)  # modify wer_dur is ok because in decoder only use for gather
        src_no_pad = (~(encoder_out.encoder_padding_mask))




        wer_dur_pred = wer_dur_pred.squeeze(-1)
        wer_dur_pred_loss_float = self.werdur_loss_func(wer_dur_pred, wer_dur, reduction='none').float()
        wer_dur_pred_loss = wer_dur_pred_loss_float[src_no_pad.bool()].mean().type_as(wer_dur_pred)

        if self.assist_edit_loss:
            if self.to_be_edited_size == 1:
                to_be_edited_pred_loss_float = F.binary_cross_entropy_with_logits(to_be_edited_pred.squeeze(-1), to_be_edited.type_as(to_be_edited_pred), reduction='none').float()
                to_be_edited_pred_loss = to_be_edited_pred_loss_float[src_no_pad.bool()].mean().type_as(to_be_edited_pred)
            else:
                raise ValueError("Unsupported condition!")
        else:
            to_be_edited_pred_loss = torch.Tensor([0.0])[0]




        word_ins_out = self.decoder(
            normalize=False,
            prev_output_tokens=prev_output_tokens,
            encoder_out=encoder_out,
            wer_dur=wer_dur,
            to_be_edited=to_be_edited, for_wer_gather=for_wer_gather, debug_src_tokens=src_tokens, debug_tgt_tokens=tgt_tokens
        )



        return_dict = {
            "wer_dur_loss": {
                "loss": wer_dur_pred_loss,
                "factor": self.decoder.length_loss_factor,
            },
        }
        return_dict["word_ins"], _ = self._compute_nll_loss(
            word_ins_out,
            tgt_tokens,
            tgt_tokens.ne(self.pad),
            self.args.label_smoothing,
            name="word_ins" + "-loss",
            factor=1.0,
        )

        if self.assist_edit_loss:
            return_dict['to_be_edited_loss'] = {
                "loss": to_be_edited_pred_loss,
                "factor": self.decoder.length_loss_factor,
            }
        return return_dict

    def forward_decoder(self, decoder_out, encoder_out, decoding_format=None, **kwargs):
        step = decoder_out.step
        output_tokens = decoder_out.output_tokens
        output_scores = decoder_out.output_scores
        history = decoder_out.history
        to_be_edited_pred = decoder_out.to_be_edited_pred
        wer_dur_pred = decoder_out.wer_dur_pred

        for_wer_gather = wer_dur_pred.cumsum(dim=-1)
        for_wer_gather = torch.nn.functional.one_hot(for_wer_gather, num_classes=for_wer_gather.max() + 1)[:, :-1, :-1].sum(-2).cumsum(dim=-1)


        # execute the decoder
        output_masks = output_tokens.ne(self.pad)
        _scores, _tokens = self.decoder(
            normalize=True,
            prev_output_tokens=output_tokens,
            encoder_out=encoder_out,
            step=step,
            wer_dur=wer_dur_pred,
            to_be_edited=to_be_edited_pred, for_wer_gather=for_wer_gather
        ).max(-1)

        output_tokens.masked_scatter_(output_masks, _tokens[output_masks])
        output_scores.masked_scatter_(output_masks, _scores[output_masks])
        if history is not None:
            history.append(output_tokens.clone())

        return decoder_out._replace(
            output_tokens=output_tokens,
            output_scores=output_scores,
            attn=None,
            history=history,
        )

    def initialize_output_tokens(self, encoder_out, src_tokens, edit_thre=0.0, print_werdur=False, werdur_gt_str=""):
        wer_dur_pred, to_be_edited_pred, closest_pred = self.decoder.forward_wer_dur_and_tbe(
            normalize=False, encoder_out=encoder_out
        )
        if 'log' in self.werdur_loss_type:
            wer_dur_pred = (torch.exp(wer_dur_pred) - 1.0).squeeze(-1).round().long().clamp_(min=0)
            length_tgt = wer_dur_pred.sum(-1)
        else:
            wer_dur_pred = wer_dur_pred.squeeze(-1).round().long().clamp_(min=0)
            length_tgt = wer_dur_pred.sum(-1)

        max_length = length_tgt.clamp_(min=2).max()
        #if len(src_tokens.shape) == 3:
        #    idx_length = utils.new_arange(src_tokens[:, :, 0], max_length)
        #else:
        idx_length = utils.new_arange(src_tokens, max_length)

        initial_output_tokens = src_tokens.new_zeros(
            src_tokens.size(0), max_length
        ).fill_(self.pad)
        initial_output_tokens.masked_fill_(
            idx_length[None, :] < length_tgt[:, None], self.unk
        )
        initial_output_tokens[:, 0] = self.bos
        initial_output_tokens.scatter_(1, length_tgt[:, None] - 1, self.eos)

        initial_output_scores = initial_output_tokens.new_zeros(
            *initial_output_tokens.size()
        ).type_as(encoder_out.encoder_out)

        return DecoderOut(
            output_tokens=initial_output_tokens,
            output_scores=initial_output_scores,
            attn=None,
            step=0,
            max_step=0,
            history=None,
            to_be_edited_pred=None,
            wer_dur_pred=wer_dur_pred
        ), encoder_out

    def regenerate_length_beam(self, decoder_out, beam_size):
        output_tokens = decoder_out.output_tokens
        length_tgt = output_tokens.ne(self.pad).sum(1)
        length_tgt = (
            length_tgt[:, None]
            + utils.new_arange(length_tgt, 1, beam_size)
            - beam_size // 2
        )
        length_tgt = length_tgt.view(-1).clamp_(min=2)
        max_length = length_tgt.max()
        idx_length = utils.new_arange(length_tgt, max_length)

        initial_output_tokens = output_tokens.new_zeros(
            length_tgt.size(0), max_length
        ).fill_(self.pad)
        initial_output_tokens.masked_fill_(
            idx_length[None, :] < length_tgt[:, None], self.unk
        )
        initial_output_tokens[:, 0] = self.bos
        initial_output_tokens.scatter_(1, length_tgt[:, None] - 1, self.eos)

        initial_output_scores = initial_output_tokens.new_zeros(
            *initial_output_tokens.size()
        ).type_as(decoder_out.output_scores)

        return decoder_out._replace(
            output_tokens=initial_output_tokens, output_scores=initial_output_scores
        )

class FastCorrectEncoder(TransformerEncoder):
    def __init__(self, args, dictionary, embed_tokens):
        super().__init__(args, dictionary, embed_tokens)
        self.ensemble_models = None

    @ensemble_encoder
    def forward(self, *args, **kwargs):
        return super().forward(*args, **kwargs)



class LayerNorm(torch.nn.LayerNorm):
    """Layer normalization module.
    :param int nout: output dim size
    :param int dim: dimension to be normalized
    """

    def __init__(self, nout, dim=-1, eps=1e-12):
        """Construct an LayerNorm object."""
        super(LayerNorm, self).__init__(nout, eps=eps)
        self.dim = dim

    def forward(self, x):
        """Apply layer normalization.
        :param torch.Tensor x: input tensor
        :return: layer normalized tensor
        :rtype torch.Tensor
        """
        if self.dim == -1:
            return super(LayerNorm, self).forward(x)
        return super(LayerNorm, self).forward(x.transpose(1, -1)).transpose(1, -1)


class DurationPredictor(torch.nn.Module):
    def __init__(self, idim, n_layers=2, n_chans=384, kernel_size=3, dropout_rate=0.1, ffn_layers=1, offset=1.0, ln_eps=1e-12, remove_edit_emb=False, to_be_edited_size=1, padding='SAME'):
        """Initilize duration predictor module.
        Args:
            idim (int): Input dimension.
            n_layers (int, optional): Number of convolutional layers.
            n_chans (int, optional): Number of channels of convolutional layers.
            kernel_size (int, optional): Kernel size of convolutional layers.
            dropout_rate (float, optional): Dropout rate.
            offset (float, optional): Offset value to avoid nan in log domain.
        """
        super(DurationPredictor, self).__init__()
        #'''
        self.offset = offset
        self.conv = torch.nn.ModuleList()
        self.kernel_size = kernel_size
        self.padding = padding
        self.remove_edit_emb = remove_edit_emb
        for idx in range(n_layers):
            in_chans = idim if idx == 0 else n_chans
            self.conv += [torch.nn.Sequential(
                torch.nn.Conv1d(in_chans, n_chans, kernel_size, stride=1, padding=0),
                torch.nn.ReLU(),
                LayerNorm(n_chans, dim=1, eps=ln_eps),
                FairseqDropout(dropout_rate, module_name="DP_dropout")
            )]
        if ffn_layers == 1:
            self.werdur_linear = torch.nn.Linear(n_chans, 1)
            self.edit_linear = torch.nn.Linear(n_chans, to_be_edited_size)
        else:
            assert ffn_layers == 2
            self.werdur_linear = torch.nn.Sequential(
                torch.nn.Linear(n_chans, n_chans // 2),
                torch.nn.ReLU(),
                FairseqDropout(dropout_rate, module_name="DP_dropout"),
                torch.nn.Linear(n_chans // 2, 1),
            )
            self.edit_linear = torch.nn.Sequential(
                torch.nn.Linear(n_chans, n_chans // 2),
                torch.nn.ReLU(),
                FairseqDropout(dropout_rate, module_name="DP_dropout"),
                torch.nn.Linear(n_chans // 2, to_be_edited_size),
            )
        #'''
        #self.werdur_linear = torch.nn.Linear(idim, 1)
        #self.edit_linear = torch.nn.Linear(idim, 1)

    def forward(self, xs, x_nonpadding=None):
        #'''
        xs = xs.transpose(1, -1)  # (B, idim, Tmax)
        for f in self.conv:
            if self.padding == 'SAME':
                xs = F.pad(xs, [self.kernel_size // 2, self.kernel_size // 2])
            elif self.padding == 'LEFT':
                xs = F.pad(xs, [self.kernel_size - 1, 0])
            xs = f(xs)  # (B, C, Tmax)
            if x_nonpadding is not None:
                xs = xs * x_nonpadding[:, None, :]

        xs = xs.transpose(1, -1)
        #'''
        werdur = self.werdur_linear(xs) * x_nonpadding[:, :, None]  # (B, Tmax)
        to_be_edited = self.edit_linear(xs) * x_nonpadding[:, :, None]  # (B, Tmax

        return werdur, to_be_edited


class FastCorrectDecoder(FairseqNATDecoder):
    def __init__(self, args, dictionary, embed_tokens, no_encoder_attn=False):
        super().__init__(
            args, dictionary, embed_tokens, no_encoder_attn=no_encoder_attn
        )
        self.dictionary = dictionary
        self.bos = dictionary.bos()
        self.unk = dictionary.unk()
        self.eos = dictionary.eos()
        # try:
        #     self.mask = dictionary.mask()
        # except:
        #     print("<mask> not found in dictionary!")
        #     self.mask = None

        self.encoder_embed_dim = args.encoder_embed_dim
        self.sg_length_pred = getattr(args, "sg_length_pred", False)
        self.length_loss_factor = getattr(args, "length_loss_factor", 0.1)
        self.to_be_edited_size = getattr(args, "to_be_edited_size", 1)
        self.edit_emb_dim = getattr(args, "edit_emb_dim", self.encoder_embed_dim // 2)


        if getattr(args, "dur_predictor_type", "") == 'v2':
            self.dur_predictor = DurationPredictor(idim=self.encoder_embed_dim, n_layers=5, n_chans=self.encoder_embed_dim, ffn_layers=2, ln_eps=1e-5, remove_edit_emb=False, to_be_edited_size=self.to_be_edited_size)
        else:
            raise ValueError("Other type is undefined")


    @ensemble_decoder
    def forward(self, normalize, encoder_out, prev_output_tokens, step=0, wer_dur=None, to_be_edited=None, for_wer_gather=None, debug_src_tokens=None, debug_tgt_tokens=None, **unused):

        features, _ = self.extract_features(
            prev_output_tokens,
            encoder_out=encoder_out,
            wer_dur=wer_dur,
            to_be_edited=to_be_edited, for_wer_gather=for_wer_gather, debug_src_tokens=debug_src_tokens, debug_tgt_tokens=debug_tgt_tokens
        )
        decoder_out = self.output_layer(features)
        return F.log_softmax(decoder_out, -1) if normalize else decoder_out

    @ensemble_decoder
    def forward_length(self, normalize, encoder_out):
        enc_feats = encoder_out.encoder_out  # T x B x C
        src_masks = encoder_out.encoder_padding_mask  # B x T or None
        enc_feats = _mean_pooling(enc_feats, src_masks)
        if self.sg_length_pred:
            enc_feats = enc_feats.detach()
        length_out = F.linear(enc_feats, self.embed_length.weight)
        return F.log_softmax(length_out, -1) if normalize else length_out


    @ensemble_decoder
    def forward_wer_dur_and_tbe(self, normalize, encoder_out):
        enc_feats = encoder_out.encoder_out  # T x B x C
        src_masks = encoder_out.encoder_padding_mask  # B x T or None
        encoder_embedding = encoder_out.encoder_embedding  # B x T x C
        enc_feats = enc_feats.transpose(0, 1)
        # enc_feats = _mean_pooling(enc_feats, src_masks)
        if self.sg_length_pred:
            enc_feats = enc_feats.detach()
        src_masks = (~src_masks)
        wer_dur_out, to_be_edited_out = self.dur_predictor(enc_feats, src_masks)
        closest = None


        return wer_dur_out, to_be_edited_out, closest

    def extract_features(
        self,
        prev_output_tokens,
        encoder_out=None,
        early_exit=None,
        wer_dur=None,
        to_be_edited=None, for_wer_gather=None, debug_src_tokens=None, debug_tgt_tokens=None,
        **unused
    ):
        """
        Similar to *forward* but only return features.

        Inputs:
            prev_output_tokens: Tensor(B, T)
            encoder_out: a dictionary of hidden states and masks

        Returns:
            tuple:
                - the decoder's features of shape `(batch, tgt_len, embed_dim)`
                - a dictionary with any model-specific outputs
            the LevenshteinTransformer decoder has full-attention to all generated tokens
        """
        # embedding
        src_embd = encoder_out.encoder_embedding
        src_mask = encoder_out.encoder_padding_mask
        src_mask = (
            ~src_mask
            if src_mask is not None
            else prev_output_tokens.new_ones(*src_embd.size()[:2]).bool()
        )

        x, decoder_padding_mask = self.forward_embedding(
            prev_output_tokens,
            self.forward_wer_dur_embedding(
                src_embd, src_mask, prev_output_tokens.ne(self.padding_idx), wer_dur, to_be_edited, for_wer_gather, debug_src_tokens=debug_src_tokens, debug_tgt_tokens=debug_tgt_tokens
            ),
        )



        # B x T x C -> T x B x C
        x = x.transpose(0, 1)
        attn = None
        inner_states = [x]

        # decoder layers
        for i, layer in enumerate(self.layers):

            # early exit from the decoder.
            if (early_exit is not None) and (i >= early_exit):
                break

            x, attn, _ = layer(
                x,
                encoder_out.encoder_out if encoder_out is not None else None,
                encoder_out.encoder_padding_mask if encoder_out is not None else None,
                self_attn_mask=None,
                self_attn_padding_mask=decoder_padding_mask,
            )
            inner_states.append(x)

        if self.layer_norm:
            x = self.layer_norm(x)

        # T x B x C -> B x T x C
        x = x.transpose(0, 1)

        if self.project_out_dim is not None:
            x = self.project_out_dim(x)

        return x, {"attn": attn, "inner_states": inner_states}

    def forward_embedding(self, prev_output_tokens, states=None):
        # embed positions
        positions = (
            self.embed_positions(prev_output_tokens)
            if self.embed_positions is not None
            else None
        )

        # embed tokens and positions
        if states is None:
            x = self.embed_scale * self.embed_tokens(prev_output_tokens)
            if self.project_in_dim is not None:
                x = self.project_in_dim(x)
        else:
            x = states

        if positions is not None:
            x += positions
        x = self.dropout_module(x)
        decoder_padding_mask = prev_output_tokens.eq(self.padding_idx)
        return x, decoder_padding_mask


    def forward_wer_dur_embedding(self, src_embeds, src_masks, tgt_masks, wer_dur, to_be_edited, for_wer_gather=None, debug_src_tokens=None, debug_tgt_tokens=None):

        batch_size, _, hidden_size = src_embeds.shape

        for_wer_gather = for_wer_gather[:, :, None].long()

        to_reshape = torch.gather(src_embeds, 1, for_wer_gather.repeat(1, 1, src_embeds.shape[2]))

        to_reshape = to_reshape * tgt_masks[:, :, None]

        return to_reshape


    def forward_length_prediction(self, length_out, encoder_out, tgt_tokens=None):
        enc_feats = encoder_out.encoder_out  # T x B x C
        src_masks = encoder_out.encoder_padding_mask  # B x T or None
        if self.pred_length_offset:
            if src_masks is None:
                src_lengs = enc_feats.new_ones(enc_feats.size(1)).fill_(
                    enc_feats.size(0)
                )
            else:
                src_lengs = (~src_masks).transpose(0, 1).type_as(enc_feats).sum(0)
            src_lengs = src_lengs.long()

        if tgt_tokens is not None:
            # obtain the length target
            tgt_lengs = tgt_tokens.ne(self.padding_idx).sum(1).long()
            if self.pred_length_offset:
                length_tgt = tgt_lengs - src_lengs + 128
            else:
                length_tgt = tgt_lengs
            length_tgt = length_tgt.clamp(min=0, max=255)

        else:
            # predict the length target (greedy for now)
            # TODO: implementing length-beam
            pred_lengs = length_out.max(-1)[1]
            if self.pred_length_offset:
                length_tgt = pred_lengs - 128 + src_lengs
            else:
                length_tgt = pred_lengs

        return length_tgt


@register_model_architecture(
    "fastcorrect", "fastcorrect"
)
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
    args.apply_bert_init = getattr(args, "apply_bert_init", False)

    args.decoder_output_dim = getattr(
        args, "decoder_output_dim", args.decoder_embed_dim
    )
    args.decoder_input_dim = getattr(args, "decoder_input_dim", args.decoder_embed_dim)

    # --- special arguments ---
    args.sg_length_pred = getattr(args, "sg_length_pred", False)
    args.pred_length_offset = getattr(args, "pred_length_offset", False)
    args.length_loss_factor = getattr(args, "length_loss_factor", 0.1)


