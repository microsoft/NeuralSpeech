# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn.functional as F
from fairseq import utils
from fairseq.iterative_refinement_generator import DecoderOut
from fairseq.models import register_model, register_model_architecture
from fairseq.models.nat import FairseqNATDecoder, FairseqNATModel, ensemble_encoder, ensemble_decoder, FairseqNATEncoder
from fairseq.models.transformer import Embedding, TransformerModel, TransformerEncoder, TransformerDecoder
from fairseq.modules.transformer_sentence_encoder import init_bert_params
from fairseq.modules import FairseqDropout
from fairseq.modules import PositionalEmbedding
from fairseq.models.fairseq_model import BaseFairseqModel
from torch import Tensor
from fairseq.models import FairseqDecoder, FairseqEncoder
from torch import nn
import math
from fairseq.models.fairseq_encoder import EncoderOut
from typing import Any, Dict, List, Optional, Tuple

import logging
logger = logging.getLogger(__name__)

DEFAULT_MAX_SOURCE_POSITIONS = 1024
DEFAULT_MAX_TARGET_POSITIONS = 1024

def Embeddingright(num_embeddings, embedding_dim, padding_idx):
    m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
    nn.init.normal_(m.weight, mean=0, std=embedding_dim ** -0.5)
    if padding_idx is not None:
        nn.init.constant_(m.weight[padding_idx], 0)
    return m



@register_model("softcorrect_corrector")
class SoftcorrectCorrectorModel(BaseFairseqModel):
    def __init__(self, args, encoder, decoder):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        assert isinstance(self.encoder, FairseqEncoder)
        assert isinstance(self.decoder, FairseqDecoder) or self.decoder is None
        self.args = args
        self.supports_align_args = True

        self.tgt_dict = decoder.dictionary if decoder else None
        if not decoder:
            print("Decoder not exit! Using bos eos from encoder!")
            self.bos = encoder.dictionary.bos()
            self.eos = encoder.dictionary.eos()
            self.pad = encoder.dictionary.pad()
            self.unk = encoder.dictionary.unk()
            try:
                self.gttoken = encoder.dictionary.gttoken()
            except:
                self.gttoken = None
        else:
            self.bos = decoder.dictionary.bos()
            self.eos = decoder.dictionary.eos()
            self.pad = decoder.dictionary.pad()
            self.unk = decoder.dictionary.unk()
            try:
                self.gttoken = decoder.dictionary.gttoken()
            except:
                self.gttoken = None

        self.ensemble_models = None

        if getattr(args, 'remove_edit_emb', False):
            print("Remove edit emb!")
            self.remove_edit_emb = True
        else:
            self.remove_edit_emb = False

        self.to_be_edited_size = getattr(args, "to_be_edited_size", 1)

        self.padding_idx = self.encoder.padding_idx

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
        self.encoder_embed_dim = args.encoder_embed_dim

        self.source_dup_factor = getattr(args, "source_dup_factor", -1)

        self.mask_ratio = getattr(args, "mask_ratio", 0.0)
        self.detector_mask_ratio = getattr(args, "detector_mask_ratio", 0.0)

        self.ft_error_distribution = getattr(args, "ft_error_distribution", None)
        self.duptoken_error_distribution = getattr(args, "duptoken_error_distribution", None)

        if self.ft_error_distribution:
            assert not self.detector_mask_ratio
            assert not self.mask_ratio
        if self.mask_ratio != 0.0 or self.ft_error_distribution:
            self.output_projection = torch.nn.Linear(
                self.encoder_embed_dim, len(self.encoder.dictionary), bias=False
            )
            torch.nn.init.normal_(
                self.output_projection.weight, mean=0, std=self.encoder_embed_dim ** -0.5
            )
        else:
            self.output_projection = None

        if self.detector_mask_ratio != 0.0:
            self.detector_projection = torch.nn.Sequential(
                torch.nn.Linear(self.encoder_embed_dim * 2, self.encoder_embed_dim * 2),
                torch.nn.LayerNorm(self.encoder_embed_dim * 2),
                torch.nn.GLU(),
                torch.nn.Linear(self.encoder_embed_dim, self.encoder_embed_dim * 2),
                torch.nn.LayerNorm(self.encoder_embed_dim * 2),
                torch.nn.GLU(),
                torch.nn.Linear(self.encoder_embed_dim, 1),
            )
        else:
            self.detector_projection = None


        # self.phone_embedding = None


        self.untouch_token_loss = getattr(args, "untouch_token_loss", 0.0)

    @property
    def allow_ensemble(self):
        return True

    def enable_ensemble(self, models):
        self.encoder.ensemble_models = [m.encoder for m in models]
        self.decoder.ensemble_models = [m.decoder for m in models]

    @classmethod
    def build_encoder(cls, args, src_dict, embed_tokens):
        encoder = SoftCorrectEncoder(args, src_dict, embed_tokens)
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


    @torch.jit.export
    def get_normalized_probs(
        self,
        net_output: Tuple[Tensor, Optional[Dict[str, List[Optional[Tensor]]]]],
        log_probs: bool,
        sample: Optional[Dict[str, Tensor]] = None,
    ):
        """Get normalized probabilities (or log probs) from a net's output."""
        return self.get_normalized_probs_scriptable(net_output, log_probs, sample)

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""

        # make sure all arguments are present in older models
        base_architecture(args)

        if args.encoder_layers_to_keep:
            args.encoder_layers = len(args.encoder_layers_to_keep.split(","))
        if args.decoder_layers_to_keep:
            args.decoder_layers = len(args.decoder_layers_to_keep.split(","))

        if getattr(args, "max_source_positions", None) is None:
            args.max_source_positions = DEFAULT_MAX_SOURCE_POSITIONS
        if getattr(args, "max_target_positions", None) is None:
            args.max_target_positions = DEFAULT_MAX_TARGET_POSITIONS

        src_dict, tgt_dict = task.source_dictionary, task.target_dictionary

        if args.share_all_embeddings:
            if src_dict != tgt_dict:
                raise ValueError("--share-all-embeddings requires a joined dictionary")
            if args.encoder_embed_dim != args.decoder_embed_dim:
                raise ValueError(
                    "--share-all-embeddings requires --encoder-embed-dim to match --decoder-embed-dim"
                )
            if args.decoder_embed_path and (
                args.decoder_embed_path != args.encoder_embed_path
            ):
                raise ValueError(
                    "--share-all-embeddings not compatible with --decoder-embed-path"
                )
            encoder_embed_tokens = cls.build_embedding(
                args, src_dict, args.encoder_embed_dim, args.encoder_embed_path
            )
            decoder_embed_tokens = encoder_embed_tokens
            args.share_decoder_input_output_embed = True
        else:
            encoder_embed_tokens = cls.build_embedding(
                args, src_dict, args.encoder_embed_dim, args.encoder_embed_path
            )
            decoder_embed_tokens = cls.build_embedding(
                args, tgt_dict, args.decoder_embed_dim, args.decoder_embed_path
            )

        encoder = cls.build_encoder(args, src_dict, encoder_embed_tokens)
        decoder = cls.build_decoder(args, tgt_dict, decoder_embed_tokens)
        return cls(args, encoder, decoder)

    @classmethod
    def build_embedding(cls, args, dictionary, embed_dim, path=None):
        num_embeddings = len(dictionary)
        padding_idx = dictionary.pad()

        emb = Embedding(num_embeddings, embed_dim, padding_idx)
        # if provided, load from preloaded dictionaries
        if path:
            embed_dict = utils.parse_embedding(path)
            utils.load_embedding(embed_dict, dictionary, emb)
        return emb


    def extract_features(self, src_tokens, src_lengths, prev_output_tokens, **kwargs):
        """
        Similar to *forward* but only return features.

        Returns:
            tuple:
                - the decoder's features of shape `(batch, tgt_len, embed_dim)`
                - a dictionary with any model-specific outputs
        """
        encoder_out = self.encoder(src_tokens, src_lengths=src_lengths, **kwargs)
        features = self.decoder.extract_features(
            prev_output_tokens, encoder_out=encoder_out, **kwargs
        )
        return features

    def output_layer(self, features, **kwargs):
        """Project features to the default output size (typically vocabulary size)."""
        return self.decoder.output_layer(features, **kwargs)

    def max_positions(self):
        """Maximum length supported by the model."""
        if not self.decoder:
            return (self.encoder.max_positions(), self.encoder.max_positions())
        else:
            return (self.encoder.max_positions(), self.decoder.max_positions())

    def max_decoder_positions(self):
        """Maximum length supported by the decoder."""
        return self.decoder.max_positions()


    def log_mse_loss(self, hypo, ref, reduction='none'):
        hypo = torch.exp(hypo) - 1.0
        return F.mse_loss(hypo, ref, reduction=reduction)

    def clipped_mse_loss(self, hypo, ref, reduction='none'):
        assert reduction == 'none'
        mse_loss = F.mse_loss(hypo, ref, reduction=reduction)
        mse_loss_pad = (mse_loss > self.clip_loss_thre ** 2).type_as(hypo).detach()
        return mse_loss * mse_loss_pad

    def log_l1_loss(self, hypo, ref, reduction='none'):
        hypo = torch.exp(hypo) - 1.0
        return F.l1_loss(hypo, ref, reduction=reduction)


    @property
    def allow_length_beam(self):
        return True

    @staticmethod
    def add_args(parser):

        #TransformerModel.add_args(parser)
        
        FairseqNATModel.add_args(parser)

        # length prediction
        parser.add_argument(
            "--src-embedding-copy",
            action="store_true",
            help="copy encoder word embeddings as the initial input of the decoder",
        )
        parser.add_argument(
            "--src-embedding-copy-exp",
            action="store_true",
            help="copy encoder word embeddings as the initial input of the decoder in exponontial way",
        )
        parser.add_argument(
            "--remove-edit-emb",
            action="store_true",
            default=False,
            help="whether to remove edit emb",
        )
        parser.add_argument(
            "--assist-edit-loss",
            action="store_true",
            default=False,
            help="whether to use assist edit loss",
        )
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
            help="dimension of edit emb",
        )

    @classmethod
    def build_decoder(cls, args, tgt_dict, embed_tokens):
        # if getattr(args, "mask_ratio", 0.0) != 0.0 or getattr(args, "detector_mask_ratio", 0.0) != 0.0 or getattr(args, "ft_error_distribution", None) is not None:
        return None

    def _compute_nll_loss(
        self, outputs, targets, masks=None, label_smoothing=0.0, name="loss", factor=1.0, skip_cloest=False, return_acc=False
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


        nll_loss_closest = None

        
        if masks is not None:
            outputs, targets = outputs[masks], targets[masks]

        if masks is not None and not masks.any():
            nll_loss = torch.tensor(0)
            loss = nll_loss
            if return_acc:
                acc = torch.tensor(0.0)
            else:
                acc = None
        else:
            logits = F.log_softmax(outputs, dim=-1)
            if targets.dim() == 1:
                losses = F.nll_loss(logits, targets.to(logits.device), reduction="none")
                if return_acc:
                    acc = (logits.max(-1)[1] == targets).float().mean()
                else:
                    acc = None
            else:  # soft-labels
                losses = F.kl_div(logits, targets.to(logits.device), reduction="none")
                losses = losses.sum(-1)
                assert not return_acc

            #nll_loss_closest = losses.float().type_as(losses).detach()

            nll_loss = mean_ds(losses)
            if label_smoothing > 0:
                loss = (
                    nll_loss * (1 - label_smoothing) - mean_ds(logits) * label_smoothing
                )
            else:
                loss = nll_loss

        loss = loss * factor

        return {"name": name, "loss": loss, "nll_loss": nll_loss, "factor": factor, "acc": acc}, nll_loss_closest

    def _compute_binary_loss(
            self, outputs, targets, masks=None, label_smoothing=0.0, name="loss", factor=1.0,
            return_acc=False
    ):
        """
        outputs: batch x len
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
            if return_acc:
                acc = torch.tensor(0.0)
            else:
                acc = None
        else:
            losses = F.binary_cross_entropy_with_logits(outputs, targets.to(outputs.device).type_as(outputs), reduction="none")
            if return_acc:
                acc = ((outputs >= 0).long() == targets).float().mean()
            else:
                acc = None

            # nll_loss_closest = losses.float().type_as(losses).detach()

            nll_loss = mean_ds(losses)
            loss = nll_loss

        loss = loss * factor

        return {"name": name, "loss": loss, "nll_loss": nll_loss, "factor": factor, "acc": acc}

    def _compute_ctc_loss(
            self, outputs, output_masks, targets, masks, blank, name="loss", factor=1.0
    ):
        """
        outputs: batch x len x d_model
        targets: batch x len
        masks:   batch x len

        policy_logprob: if there is some policy
            depends on the likelihood score as rewards.
        """

        outputs_len = output_masks.sum(-1)
        targets_len = masks.sum(-1)

        if masks is not None and not masks.any():
            nll_loss = torch.tensor(0)
            loss = nll_loss
        else:
            logits = F.log_softmax(outputs.float(), dim=-1)
            #logits = F.log_softmax(outputs, dim=-1)
            logits = logits.transpose(0, 1)
            with torch.backends.cudnn.flags(enabled=False):
                loss = torch.nn.functional.ctc_loss(logits, targets.to(logits.device), outputs_len, targets_len, blank=blank, reduction='mean', zero_infinity=True)


        loss = loss * factor
        #loss = loss.type_as(outputs)
        return {"name": name, "loss": loss, "nll_loss": torch.Tensor([0.0])[0].to(logits.device), "factor": factor}, torch.Tensor([0.0])[0].to(logits.device)

    def forward_encoder(self, encoder_inputs):
        if len(encoder_inputs) == 3:
            src_tokens, src_lengths, mask_type = encoder_inputs
        else:
            src_tokens, src_lengths = encoder_inputs
            mask_type = None
        attn_mask = None

        if mask_type is not None:
            source_nonpadding = (src_tokens != self.pad)
            source_token = src_tokens * (source_nonpadding.long())

            source_embedding = (self.encoder.embed_tokens(source_token) * (source_token != 0).long()[:, :, None])

            encoder_out = self.encoder(src_tokens, src_lengths=src_lengths, token_embeddings=source_embedding)

            if self.detector_projection is not None:
                detector_out = self.detector_projection(
                    torch.cat([encoder_out.encoder_out.transpose(0, 1), encoder_out.encoder_embedding], dim=-1)).squeeze(-1)
                return detector_out
            else:
                encoder_out = self.output_projection(encoder_out.encoder_out.transpose(0, 1))

                return torch.log_softmax(encoder_out, dim=-1)
        else:

            phone_feat = None
            return self.encoder(src_tokens, src_lengths=src_lengths, phone_feat=phone_feat)

    def forward_mlm(self, src_tokens, src_lengths, tgt_tokens, tgt_tokens_full, tgt_tokens_ctc, mask_type, attn_mask):
        assert mask_type is not None
        source_nonpadding = (src_tokens != self.pad)
        untouched_token = (mask_type == 0) * source_nonpadding
        identity_token = (mask_type == 1) * source_nonpadding
        mask_token = (mask_type == 2) * source_nonpadding
        homophone_token = (mask_type == 3) * source_nonpadding
        ctc_token = (mask_type == 4) * source_nonpadding   # reserve for future work
        random_token = (mask_type == 5) * source_nonpadding
        need_detect = (untouched_token + identity_token + homophone_token + random_token).bool()
        detect_label = (random_token + homophone_token).bool().long()

        source_token = src_tokens * (source_nonpadding.long())

        source_embedding = (self.encoder.embed_tokens(source_token) * (source_token != 0).long()[:, :, None])

        encoder_out = self.encoder(src_tokens, src_lengths=src_lengths, token_embeddings=source_embedding)

        if self.detector_projection is not None:
            detector_out = self.detector_projection(torch.cat([encoder_out.encoder_out.transpose(0, 1), encoder_out.encoder_embedding], dim=-1)).squeeze(-1)
            detector_loss = self._compute_binary_loss(
                detector_out,
                detect_label,
                need_detect,
                self.args.label_smoothing,
                name="Binary-loss",
                factor=1.0,
                return_acc=True
            )
        else:
            detector_loss = None

        encoder_out = self.output_projection(encoder_out.encoder_out.transpose(0, 1))

        ctc_gt_part = torch.zeros_like(encoder_out) + torch.nn.functional.one_hot(tgt_tokens_full, num_classes=encoder_out.shape[-1]) * 30
        encoder_out_ctc = encoder_out * ctc_token[:, :, None] + ctc_gt_part.detach() * ~ctc_token[:, :, None]


        result_dict = {}
        # prepare ctc probability!!!!!!!!!!!!
        if self.ft_error_distribution is not None or (self.mask_ratio != 0.0 and self.detector_mask_ratio == 0.0):  #bert generator has no ctc loss
            result_dict["cons_ctc_loss"], _ = self._compute_ctc_loss(
                encoder_out_ctc,
                src_tokens.ne(0),
                tgt_tokens_ctc,
                tgt_tokens_ctc.ne(self.pad),
                blank=0,
                name="ConsCTC" + "-loss",
                factor=1.0,
            )

        result_dict["identity_loss"], _ = self._compute_nll_loss(
            encoder_out,
            tgt_tokens_full,
            identity_token,
            self.args.label_smoothing,
            name="Id-loss",
            factor=1.0,
            skip_cloest=True,
            return_acc=True
        )

        result_dict["mask_loss"], _ = self._compute_nll_loss(
            encoder_out,
            tgt_tokens_full,
            mask_token,
            self.args.label_smoothing,
            name="Mask-loss",
            factor=1.0,
            skip_cloest=True,
            return_acc=True
        )

        result_dict["homophone_loss"], _ = self._compute_nll_loss(
            encoder_out,
            tgt_tokens_full,
            homophone_token.bool(),
            self.args.label_smoothing,
            name="Homophone-loss",
            factor=1.0,
            skip_cloest=True,
            return_acc=True
        )

        result_dict["random_loss"], _ = self._compute_nll_loss(
            encoder_out,
            tgt_tokens_full,
            random_token.bool(),
            self.args.label_smoothing,
            name="Random-loss",
            factor=1.0,
            skip_cloest=True,
            return_acc=True
        )

        if self.untouch_token_loss > 0.0:
            result_dict["untouch_loss"], _ = self._compute_nll_loss(
                encoder_out,
                tgt_tokens_full,
                untouched_token.bool(),
                self.args.label_smoothing,
                name="Untouch-loss",
                factor=self.untouch_token_loss,
                skip_cloest=True,
                return_acc=True
            )

        if detector_loss is not None:
            result_dict["detector_loss"] = detector_loss

        return result_dict







    def forward(
        self, src_tokens, src_lengths, prev_output_tokens, tgt_tokens, wer_dur=None, to_be_edited=None, for_wer_gather=None,
            closest_label=None, source_phone=None, mask_type=None, target_full=None, target_ctc=None, **kwargs
    ):
        # encoding
        attn_mask = None

        return self.forward_mlm(src_tokens, src_lengths, tgt_tokens, mask_type=mask_type, tgt_tokens_full=target_full, tgt_tokens_ctc=target_ctc, attn_mask=attn_mask)

    def forward_decoder(self, decoder_out, encoder_out, decoding_format=None, source_phone=None, **kwargs):
        step = decoder_out.step
        output_tokens = decoder_out.output_tokens
        output_scores = decoder_out.output_scores
        history = decoder_out.history

        # execute the decoder
        output_masks = output_tokens.ne(self.pad).bool()
        assert output_tokens[0].ne(self.pad).long().sum() == output_tokens.shape[1]
        _tokens = self.decoder(
            normalize=True,  #normalize=True
            prev_output_tokens=output_tokens,
            encoder_out=encoder_out,
            step=step,
            source_phone=source_phone
        )

        if history is not None:
            history.append(output_tokens.clone())

        return decoder_out._replace(
            output_tokens=_tokens,
            output_scores=None,
            attn=None,
            history=history,
        )


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

class SoftCorrectEncoder(TransformerEncoder):
    def __init__(self, args, dictionary, embed_tokens):
        super().__init__(args, dictionary, embed_tokens)

        if getattr(args, "emb_dropout", -1) != -1:
            print("Setting word embedding dropout of encoder to {}".format(args.emb_dropout))
            self.dropout_module = FairseqDropout(
                args.emb_dropout, module_name=self.__class__.__name__
            )
        else:
            self.dropout_module = FairseqDropout(
                args.dropout, module_name=self.__class__.__name__
            )

        self.ensemble_models = None

        embed_dim = embed_tokens.embedding_dim

        self.pos_before_reshape = getattr(args, "pos_before_reshape", False)
        self.nbest_input_num = getattr(args, "nbest_input_num", 1)

        if self.nbest_input_num != 1:
            self.embed_positions = (
                PositionalEmbedding(
                    args.max_source_positions,
                    embed_dim * self.nbest_input_num if self.pos_before_reshape else embed_dim,
                    self.padding_idx,
                    learned=args.encoder_learned_pos,
                )
                if not args.no_token_positional_embeddings
                else None
            )
        else:
            self.embed_positions = (
                PositionalEmbedding(
                    args.max_source_positions,
                    embed_dim,
                    self.padding_idx,
                    learned=args.encoder_learned_pos,
                )
                if not args.no_token_positional_embeddings
                else None
            )

        if self.nbest_input_num != 1:
            self.nbest_reshape = nn.Linear(args.nbest_input_num * embed_dim, embed_dim, bias=False)
        else:
            self.nbest_reshape = None


    @ensemble_encoder
    def forward(
        self,
        src_tokens,
        src_lengths,
        attn_mask=None,
        phone_feat=None,
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
            namedtuple:
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
        x, encoder_embedding = self.forward_embedding(src_tokens, token_embeddings, phone_feat=phone_feat)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        # compute padding mask
        if len(src_tokens.shape) == 3:
            encoder_padding_mask = src_tokens[:, :, 0].eq(self.padding_idx)
        else:
            encoder_padding_mask = src_tokens.eq(self.padding_idx)

        encoder_states = [] if return_all_hiddens else None

        # encoder layers
        for layer in self.layers:
            x = layer(x, encoder_padding_mask, attn_mask=attn_mask)
            if return_all_hiddens:
                assert encoder_states is not None
                encoder_states.append(x)

        if self.layer_norm is not None:
            x = self.layer_norm(x)

        return EncoderOut(
            encoder_out=x,  # T x B x C
            encoder_padding_mask=encoder_padding_mask,  # B x T
            encoder_embedding=encoder_embedding,  # B x T x C
            encoder_states=encoder_states,  # List[T x B x C]
            src_tokens=None,
            src_lengths=None,
        )


    def forward_embedding(
        self, src_tokens, token_embedding: Optional[torch.Tensor] = None, phone_feat: Optional[torch.Tensor] = None,
    ):
        # embed tokens and positions
        if token_embedding is None:
            token_embedding = self.embed_tokens(src_tokens)
        x = embed = self.embed_scale * token_embedding

        if len(src_tokens.shape) == 2:
            if self.embed_positions is not None:
                x = embed + self.embed_positions(src_tokens)
        else:
            if len(token_embedding.shape) == 3:
                x = embed + self.embed_positions(src_tokens[:, :, 0])
            else:
                assert self.nbest_reshape is not None
                if self.pos_before_reshape:
                    if self.embed_positions is not None:
                        x = x.reshape(x.shape[0], x.shape[1], x.shape[2] * x.shape[3])
                        x = x + self.embed_positions(src_tokens[:, :, 0])
                    x = self.nbest_reshape(x)
                else:
                    x = self.nbest_reshape(x.reshape(x.shape[0], x.shape[1], x.shape[2] * x.shape[3]))
                    if self.embed_positions is not None:
                        x = x + self.embed_positions(src_tokens[:, :, 0])

        if self.layernorm_embedding is not None:
            x = self.layernorm_embedding(x)
        x = self.dropout_module(x)
        if self.quant_noise is not None:
            x = self.quant_noise(x)
        return x, embed




@register_model_architecture(
    "softcorrect_corrector", "softcorrect_corrector"
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
    args.src_embedding_copy = getattr(args, "src_embedding_copy", False)
    args.no_scale_embedding = getattr(args, "no_scale_embedding", False)


class Swish(torch.nn.Module):
    """Construct an Swish object."""

    def forward(self, x):
        """Return Swich activation function."""
        return x * torch.sigmoid(x)

@register_model("softcorrect_detector")
class SoftcorrectDetectorModel(SoftcorrectCorrectorModel):
    def __init__(self, args, encoder, decoder):
        super().__init__(args, encoder, decoder)


        self.candidate_size = getattr(args, "candidate_size", 0)

        self.label_leak_prob = getattr(args, "label_leak_prob", 0.0)
        self.label_ignore_prob = getattr(args, "label_ignore_prob", 0.0)


        self.output_projection = None

        if self.detector_mask_ratio != 0.0 and self.candidate_size == 0:
            self.detector_projection = torch.nn.Sequential(
                torch.nn.Linear(self.encoder_embed_dim * 2, self.encoder_embed_dim * 2),
                torch.nn.LayerNorm(self.encoder_embed_dim * 2),
                torch.nn.GLU(),
                torch.nn.Linear(self.encoder_embed_dim, self.encoder_embed_dim * 2),
                torch.nn.LayerNorm(self.encoder_embed_dim * 2),
                torch.nn.GLU(),
                torch.nn.Linear(self.encoder_embed_dim, 1),
            )
        else:
            self.detector_projection = None

        self.encoder_training_type = getattr(args, "encoder_training_type", "detector")

        self.emb_dropout = getattr(args, "emb_dropout", -1)


        if self.candidate_size == -1:
            right_pos_before_reshape = args.pos_before_reshape
            args.pos_before_reshape = False
            layernorm_embedding = getattr(args, "layernorm_embedding", False)
            encoder_layers = args.encoder_layers
            args.encoder_layers = 12
            args.layernorm_embedding = False
            args.emb_dropout = -1  # saved in self.emb_dropout
            self.bert_generator_encoder = self.build_encoder(args=args, src_dict=self.encoder.dictionary, embed_tokens=Embedding(len(self.encoder.dictionary), self.encoder.embed_tokens.weight.shape[1], self.encoder.dictionary.pad()))
            self.bert_generator_output_projection = torch.nn.Linear(
                self.encoder_embed_dim, len(self.encoder.dictionary), bias=False
            )
            for param in self.bert_generator_encoder.parameters():
                param.requires_grad = False
            for param in self.bert_generator_output_projection.parameters():
                param.requires_grad = False
            args.encoder_layers = encoder_layers
            args.pos_before_reshape = right_pos_before_reshape
            args.emb_dropout = self.emb_dropout
            args.layernorm_embedding = layernorm_embedding
        else:
            assert self.candidate_size == 0, "candidate_size must in [-1, 0]"
            self.bert_generator_encoder = None
            self.bert_generator_output_projection = None

        self.nbest_input_num = getattr(args, "nbest_input_num", 1)
        self.nbest_void_insert_ratio = getattr(args, "nbest_void_insert_ratio", 0.0)
        self.nbest_input_sample_temp = getattr(args, "nbest_input_sample_temp", 1.0)
        self.nbest_input_sample_untouch_temp = getattr(args, "nbest_input_sample_untouch_temp", -1.0)
        # if self.nbest_input_sample_untouch_temp:
        #     print("Sampling temperature of touch/untouch is {}/{}!".format(self.nbest_input_sample_temp, self.nbest_input_sample_untouch_temp))
        self.force_same_ratio = getattr(args, "force_same_ratio", 0.0)
        assert self.encoder.dictionary.gttoken() + 1 == len(self.encoder.dictionary), "gttoken log require this"
        self.same_also_sample = getattr(args, "same_also_sample", 0.0)
        if self.same_also_sample == 0.0:
            self.same_also_sample = False





    def cal_acc(
            self, flags, masks=None, name="loss",
    ):

        if masks is not None:
            flags = flags[masks]

        if masks is not None and not masks.any():
            acc = torch.tensor(0.0)

        else:
            acc = flags.float().mean()
        return {"name": name, "loss": torch.tensor(0), "nll_loss": torch.tensor(0), "factor": 1.0,
                "acc": acc, "log_acc_only": True}

    def _compute_nll_loss(
        self, outputs, targets, masks=None, label_smoothing=0.0, name="loss", factor=1.0, skip_cloest=False,
            return_acc=False, log_acc_only=False, input_is_label=False, ignore_gttoken=False
    ):
        """
        outputs: batch x len x d_model
        targets: batch x len
        masks:   batch x len

        policy_logprob: if there is some policy
            depends on the likelihood score as rewards.
        """

        if ignore_gttoken:
            assert not input_is_label
        # if input_is_label:
        #     assert not ignore_gttoken

        def mean_ds(x: Tensor, dim=None) -> Tensor:
            return (
                x.float().mean().type_as(x)
                if dim is None
                else x.float().mean(dim).type_as(x)
            )

        if not skip_cloest:
            logits_clo = F.log_softmax(outputs, dim=-1)
            losses_clo = F.nll_loss(logits_clo.transpose(1,2), targets.to(outputs.device), reduction="none")
            masks_clo = masks.float()
            losses_clo = (losses_clo * masks_clo).sum(-1) / masks_clo.sum(-1)
            nll_loss_closest = losses_clo.type_as(outputs).detach()
        else:
            nll_loss_closest = None

        
        if masks is not None:
            outputs, targets = outputs[masks], targets[masks]

        if masks is not None and not masks.any():
            nll_loss = torch.tensor(0)
            loss = nll_loss
            if return_acc:
                acc = torch.tensor(0.0)
            else:
                acc = None
        else:
            if log_acc_only:
                assert return_acc
                if ignore_gttoken:
                    acc = (outputs[:, :-1].max(-1)[1] == targets).float().mean()
                else:
                    acc = (outputs.max(-1)[1] == targets).float().mean()
                return {"name": name, "loss": torch.tensor(0), "nll_loss": torch.tensor(0), "factor": factor, "acc": acc,
                        "log_acc_only": True}, nll_loss_closest
            else:
                logits = F.log_softmax(outputs, dim=-1)
                if targets.dim() == 1:
                    losses = F.nll_loss(logits, targets.to(logits.device), reduction="none")
                    if return_acc:
                        acc = (logits.max(-1)[1] == targets).float().mean()
                    else:
                        acc = None
                else:  # soft-labels
                    losses = F.kl_div(logits, targets.to(logits.device), reduction="none")
                    losses = losses.sum(-1)
                    assert not return_acc


            nll_loss = mean_ds(losses)
            if label_smoothing > 0:
                loss = (
                    nll_loss * (1 - label_smoothing) - mean_ds(logits) * label_smoothing
                )
            else:
                loss = nll_loss

        loss = loss * factor

        return {"name": name, "loss": loss, "nll_loss": nll_loss, "factor": factor, "acc": acc, "log_acc_only": log_acc_only}, nll_loss_closest


    def forward_encoder(self, encoder_inputs):
        if len(encoder_inputs) == 4:
            src_tokens, src_lengths, mask_type, return_origin = encoder_inputs
        elif len(encoder_inputs) == 3:
            src_tokens, src_lengths, mask_type = encoder_inputs
            return_origin = False
        else:
            src_tokens, src_lengths = encoder_inputs
            mask_type = None
            return_origin = False
        attn_mask = None

        if mask_type is not None:
            source_nonpadding = (src_tokens != self.pad)

            if self.nbest_input_num != 1:
                #assert self.phone_embedding is None
                assert (mask_type == 4).long().sum() == 0.0
                source_embedding = self.encoder.embed_tokens(src_tokens) * source_nonpadding.long()[:, :, :, None]
            else:
                dup_token = (mask_type == 4) * source_nonpadding
                source_dup = src_tokens * ((mask_type == 4).long() * source_nonpadding.long())
                assert source_dup.sum() == 0.0

            encoder_out = self.encoder(src_tokens, src_lengths=src_lengths, token_embeddings=source_embedding, attn_mask=attn_mask)
            if return_origin:
                return encoder_out
            if self.detector_projection is not None:
                detector_out = self.detector_projection(
                    torch.cat([encoder_out.encoder_out.transpose(0, 1), encoder_out.encoder_embedding], dim=-1)).squeeze(-1)
                return detector_out
            elif self.output_projection is None:
                encoder_out = F.linear(encoder_out.encoder_out.transpose(0, 1), self.encoder.embed_tokens.weight)

                return torch.log_softmax(encoder_out, dim=-1)
            else:
                encoder_out = self.output_projection(encoder_out.encoder_out.transpose(0, 1))
                return torch.log_softmax(encoder_out, dim=-1)
        return self.encoder(src_tokens, src_lengths=src_lengths)

    def forward_bert_and_prepare_tgt(self, src_tokens, src_lengths, tgt_tokens, tgt_tokens_full, tgt_tokens_ctc, mask_type, bert_generator_attn_mask=None, return_input_only=False):
        assert mask_type is not None
        source_nonpadding = (src_tokens != self.pad)
        batch_size, time_length = source_nonpadding.shape
        untouched_token = (mask_type == 0) * source_nonpadding
        identity_token = (mask_type == 1) * source_nonpadding
        mask_token = (mask_type == 2) * source_nonpadding
        homophone_token = (mask_type == 3) * source_nonpadding
        dup_token = (mask_type == 4) * source_nonpadding
        random_token = (mask_type == 5) * source_nonpadding
        inserted_token = (mask_type == 6) * source_nonpadding
        assert dup_token.long().sum() == 0
        source_token = src_tokens * ((mask_type != 4).long() * source_nonpadding.long())
        source_embedding = (self.bert_generator_encoder.embed_tokens(source_token) * (source_token != 0).long()[:, :, None])
        bert_generator_encoder_out = self.bert_generator_encoder(src_tokens, src_lengths=src_lengths, token_embeddings=source_embedding)

        bert_generator_encoder_out = self.bert_generator_output_projection(bert_generator_encoder_out.encoder_out.transpose(0, 1))
        bert_generator_encoder_out = bert_generator_encoder_out * source_nonpadding[:, :, None]

        bert_generator_encoder_out_raw = bert_generator_encoder_out

        bert_generator_acc_dict = {}

        bert_generator_acc_dict["bert_generator_identity_loss"], _ = self._compute_nll_loss(
            bert_generator_encoder_out_raw,
            tgt_tokens_full,
            identity_token,
            self.args.label_smoothing,
            name="BERT_Id-loss",
            factor=1.0,
            skip_cloest=True,
            return_acc=True,
            log_acc_only=True,
        )

        bert_generator_acc_dict["bert_generator_mask_loss"], _ = self._compute_nll_loss(
            bert_generator_encoder_out_raw,
            tgt_tokens_full,
            mask_token,
            self.args.label_smoothing,
            name="BERT_Mask-loss",
            factor=1.0,
            skip_cloest=True,
            return_acc=True,
            log_acc_only=True,
        )

        bert_generator_acc_dict["bert_generator_homophone_loss"], _ = self._compute_nll_loss(
            bert_generator_encoder_out_raw,
            tgt_tokens_full,
            homophone_token.bool(),
            self.args.label_smoothing,
            name="BERT_Homo-loss",
            factor=1.0,
            skip_cloest=True,
            return_acc=True,
            log_acc_only=True,
        )

        bert_generator_acc_dict["bert_generator_random_loss"], _ = self._compute_nll_loss(
            bert_generator_encoder_out_raw,
            tgt_tokens_full,
            random_token.bool(),
            self.args.label_smoothing,
            name="BERT_Rand-loss",
            factor=1.0,
            skip_cloest=True,
            return_acc=True,
            log_acc_only=True,
        )

        bert_generator_acc_dict["bert_generator_untouch_loss"], _ = self._compute_nll_loss(
            bert_generator_encoder_out_raw,
            tgt_tokens_full,
            untouched_token.bool(),
            self.args.label_smoothing,
            name="BERT_Untouch-loss",
            factor=self.untouch_token_loss,
            skip_cloest=True,
            return_acc=True,
            log_acc_only=True,
        )

        _, bert_generator_encoder_out_label = bert_generator_encoder_out.max(dim=-1)

        vocab_size = bert_generator_encoder_out.shape[-1]

        if self.nbest_input_num == 1:
            final_corrupt_token = bert_generator_encoder_out_label * source_nonpadding
            assert self.encoder_training_type != "bert"
            token_to_mask_flag = None
        else:
            final_corrupt_token_same = bert_generator_encoder_out_label[:, :, None].repeat(1, 1, self.nbest_input_num) * source_nonpadding[:, :, None]
            void_token_prob = F.one_hot(torch.full((1, 1), self.encoder.dictionary.void(), dtype=src_tokens.dtype).type_as(src_tokens), num_classes=vocab_size)
            if self.nbest_input_sample_untouch_temp == -1.0:
                adjusted_prob = torch.softmax(bert_generator_encoder_out.float() / self.nbest_input_sample_temp, dim=-1)
            elif self.nbest_input_sample_untouch_temp == 0.0:
                adjusted_prob_noise = torch.softmax(bert_generator_encoder_out.float() / self.nbest_input_sample_temp, dim=-1)
                adjusted_prob_untouch = F.one_hot(bert_generator_encoder_out.max(-1)[1], num_classes=vocab_size).type_as(adjusted_prob_noise)
                adjusted_prob = torch.where(
                    (mask_token + homophone_token + inserted_token + random_token + dup_token)[:, :, None],
                    adjusted_prob_noise, adjusted_prob_untouch)
            else:
                adjusted_prob_noise = torch.softmax(bert_generator_encoder_out.float() / self.nbest_input_sample_temp, dim=-1)
                adjusted_prob_untouch = torch.softmax(bert_generator_encoder_out.float() / self.nbest_input_sample_untouch_temp, dim=-1)
                adjusted_prob = torch.where((mask_token + homophone_token + inserted_token + random_token + dup_token)[:, :, None], adjusted_prob_noise, adjusted_prob_untouch)
            if self.same_also_sample:
                if self.same_also_sample != -1:
                    assert self.same_also_sample > 0
                    forsame_prob = torch.softmax(bert_generator_encoder_out.float() / self.same_also_sample, dim=-1)
                    final_corrupt_token_same = torch.multinomial(
                        forsame_prob.reshape([batch_size * time_length, vocab_size]),
                        1, replacement=True).reshape([batch_size, time_length, 1]).repeat(1, 1, self.nbest_input_num) * source_nonpadding[:, :, None]
                else:
                    final_corrupt_token_same = torch.multinomial(adjusted_prob.reshape([batch_size * time_length, vocab_size]),
                                            1, replacement=True).reshape([batch_size, time_length, 1]).repeat(1, 1, self.nbest_input_num) * source_nonpadding[:, :, None]
            void_token_prob = self.nbest_void_insert_ratio * void_token_prob + 2.0 * self.nbest_void_insert_ratio * void_token_prob * inserted_token[:, :, None]
            if self.nbest_input_sample_untouch_temp == 0.0:
                void_token_prob = void_token_prob * ~(untouched_token + identity_token)[:, :, None]
            final_corrupt_token_diff = torch.multinomial(adjusted_prob.reshape([batch_size * time_length, vocab_size]) + void_token_prob.reshape([batch_size * time_length, vocab_size]),
                                            self.nbest_input_num, replacement=True).reshape([batch_size, time_length, self.nbest_input_num])
            if self.force_same_ratio == 0.0:
                same_diff_var = (mask_token + homophone_token + inserted_token + random_token + dup_token)[:, :, None]
            else:
                same_diff_var = (torch.rand(final_corrupt_token_diff.shape[0], final_corrupt_token_diff.shape[1]).type_as(bert_generator_encoder_out) < 1.0 - self.force_same_ratio)[:, :, None] * source_nonpadding[:, :, None]
            final_corrupt_token = torch.where(same_diff_var, final_corrupt_token_diff, final_corrupt_token_same)

            if self.encoder_training_type == "bert":
                token_to_mask_flag = (torch.rand(final_corrupt_token.shape[0], final_corrupt_token.shape[1]).type_as(bert_generator_encoder_out) < self.mask_ratio / 2) * source_nonpadding
                final_corrupt_token = torch.where(
                    token_to_mask_flag.bool()[:, :, None],
                    final_corrupt_token.new_zeros(final_corrupt_token.shape).fill_(self.encoder.dictionary.mask()),
                    final_corrupt_token
                )
            else:
                token_to_mask_flag = None

        if return_input_only:
            return final_corrupt_token.detach(), bert_generator_acc_dict, token_to_mask_flag, bert_generator_encoder_out_label

        if self.nbest_input_num == 1:
            bert_generator_encoder_out_label_right = (bert_generator_encoder_out_label == tgt_tokens_full) * source_nonpadding
            bert_generator_encoder_out_label_wrong = (bert_generator_encoder_out_label != tgt_tokens_full) * source_nonpadding
        else:
            consist_flag = torch.ne(final_corrupt_token,
                                    final_corrupt_token[:, :, 0:1].repeat(1, 1, self.nbest_input_num)).long().sum(-1)
            consist_flag = (consist_flag == 0)
            right_wrong_flag = torch.eq(final_corrupt_token, tgt_tokens_full[:, :, None].repeat(1, 1, self.nbest_input_num)).long().sum(-1)
            bert_generator_encoder_out_label_right = (right_wrong_flag > 0) * source_nonpadding
            bert_generator_encoder_out_label_wrong = (right_wrong_flag == 0) * source_nonpadding

            bert_generator_acc_dict["bert_generator_diffright_nbest_loss"] = self.cal_acc(
                (1.0 - consist_flag.long()) * bert_generator_encoder_out_label_right,
                source_nonpadding.bool(),
                name="BERT_Diffright_nbest-loss",
            )

            bert_generator_acc_dict["bert_generator_diffwrong_nbest_loss"] = self.cal_acc(
                (1.0 - consist_flag.long()) * bert_generator_encoder_out_label_wrong,
                source_nonpadding.bool(),
                name="BERT_Diffwrong_nbest-loss",
            )

            bert_generator_acc_dict["bert_generator_sameright_nbest_loss"] = self.cal_acc(
                consist_flag.long() * bert_generator_encoder_out_label_right,
                source_nonpadding.bool(),
                name="BERT_Sameright_nbest-loss",
            )

            bert_generator_acc_dict["bert_generator_samewrong_nbest_loss"] = self.cal_acc(
                consist_flag.long() * bert_generator_encoder_out_label_wrong,
                source_nonpadding.bool(),
                name="BERT_Samewrong_nbest-loss",
            )

            bert_generator_acc_dict["bert_generator_random_nbest_loss"] = self.cal_acc(
                bert_generator_encoder_out_label_right,
                random_token.bool(),
                name="BERT_Rand_nbest-loss",
            )

            bert_generator_acc_dict["bert_generator_untouch_nbest_loss"] = self.cal_acc(
                bert_generator_encoder_out_label_right,
                untouched_token.bool(),
                name="BERT_Untouch_nbest-loss",
            )

            bert_generator_acc_dict["bert_generator_homophone_nbest_loss"] = self.cal_acc(
                bert_generator_encoder_out_label_right,
                homophone_token.bool(),
                name="BERT_Homo_nbest-loss",
            )

            bert_generator_acc_dict["bert_generator_mask_nbest_loss"] = self.cal_acc(
                bert_generator_encoder_out_label_right,
                mask_token.bool(),
                name="BERT_Mask_nbest-loss",
            )

            bert_generator_acc_dict["bert_generator_identity_nbest_loss"] = self.cal_acc(
                bert_generator_encoder_out_label_right,
                identity_token.bool(),
                name="BERT_Id_nbest-loss",
            )

        label_leak_var = torch.rand_like(bert_generator_encoder_out_label_right.type_as(bert_generator_encoder_out_raw))
        bert_generator_encoder_out_label_right_nonleak = bert_generator_encoder_out_label_right * (label_leak_var >= self.label_leak_prob)
        bert_generator_encoder_out_label_right_leak = bert_generator_encoder_out_label_right * (label_leak_var < self.label_leak_prob)

        label_ignore_var = torch.rand_like(bert_generator_encoder_out_label_wrong.type_as(bert_generator_encoder_out_raw))
        bert_generator_encoder_out_label_wrong_nonignore = bert_generator_encoder_out_label_wrong * (label_ignore_var >= self.label_ignore_prob)
        bert_generator_encoder_out_label_wrong_ignore = bert_generator_encoder_out_label_wrong * (label_ignore_var < self.label_ignore_prob)
        bert_generator_encoder_out_label_diff_nonignore = None
        bert_generator_encoder_out_label_diff_ignore = None


        if self.candidate_size == -1:
            return final_corrupt_token.detach(), None, None, bert_generator_acc_dict, bert_generator_encoder_out_label_wrong_ignore, \
                   bert_generator_encoder_out_label_wrong_nonignore, bert_generator_encoder_out_label_right_leak, bert_generator_encoder_out_label_right_nonleak, bert_generator_encoder_out_label_diff_ignore, bert_generator_encoder_out_label_diff_nonignore, bert_generator_encoder_out_label

        raise ValueError("Not supported candidate_size")

    def forward_mlm(self, src_tokens, src_lengths, tgt_tokens, prev_output_tokens, tgt_tokens_full, tgt_tokens_ctc, mask_type, attn_mask=None):

        assert mask_type is not None
        if len(src_tokens.shape) == 3:
            source_nonpadding = (src_tokens[:, :, 0] != self.pad)
        else:
            source_nonpadding = (src_tokens != self.pad)
        untouched_token = (mask_type == 0) * source_nonpadding
        identity_token = (mask_type == 1) * source_nonpadding
        mask_token = (mask_type == 2) * source_nonpadding
        homophone_token = (mask_type == 3) * source_nonpadding
        dup_token = (mask_type == 4) * source_nonpadding
        random_token = (mask_type == 5) * source_nonpadding
        inserted_token = (mask_type == 6) * source_nonpadding
        need_detect = (untouched_token + identity_token + homophone_token + random_token).bool()
        detect_label = (random_token + homophone_token).bool().long()

        if self.encoder_training_type in ["bert"]:
            corrupt_token_input, bert_generator_acc_dict, token_to_mask_flag, bert_generator_encoder_out_label = self.forward_bert_and_prepare_tgt(
                src_tokens, src_lengths, tgt_tokens, tgt_tokens_full, tgt_tokens_ctc, mask_type, bert_generator_attn_mask=None, return_input_only=True)
        elif self.encoder_training_type in ["detector"]:
            corrupt_token_input, candidates, candidates_label, bert_generator_acc_dict, bert_generator_encoder_out_label_wrong_ignore, bert_generator_encoder_out_label_wrong_nonignore, \
            bert_generator_encoder_out_label_right_leak, bert_generator_encoder_out_label_right_nonleak, bert_generator_encoder_out_label_diff_ignore, bert_generator_encoder_out_label_diff_nonignore, bert_generator_encoder_out_label = self.forward_bert_and_prepare_tgt(
                src_tokens, src_lengths, tgt_tokens, tgt_tokens_full, tgt_tokens_ctc, mask_type, bert_generator_attn_mask=None)
            token_to_mask_flag = None
        else:
            raise ValueError("Impossible encoder_training_type {}!".format(self.encoder_training_type))


        if self.nbest_input_num != 1:
            corrupt_token_input = corrupt_token_input.masked_fill(~(source_nonpadding[:, :, None].repeat(1, 1, self.nbest_input_num)), self.pad)
            source_embedding = self.encoder.embed_tokens(corrupt_token_input) * source_nonpadding.long()[:, :, None, None]
        else:
            corrupt_token_input = corrupt_token_input.masked_fill(~source_nonpadding, self.pad)
            source_embedding = self.encoder.embed_tokens(corrupt_token_input) * source_nonpadding.long()[:, :, None]


        encoder_out = self.encoder(corrupt_token_input, src_lengths=src_lengths, token_embeddings=source_embedding, attn_mask=attn_mask)
        encoder_out_vocab = F.linear(encoder_out.encoder_out.transpose(0, 1),
                               self.encoder.embed_tokens.weight)  # [B, T, vocab_size]
        result_dict = bert_generator_acc_dict



        if self.encoder_training_type == "bert":
            result_dict["mlm_loss"], _ = self._compute_nll_loss(
                encoder_out_vocab,
                tgt_tokens_full,
                token_to_mask_flag.bool(),
                self.args.label_smoothing,
                name="mlm-loss",
                factor=1.0,
                skip_cloest=True,
                return_acc=True,
                log_acc_only=False,
            )

            mask_for_candidate = (mask_token + dup_token + homophone_token + random_token + inserted_token).long() - token_to_mask_flag.long()
            mask_for_candidate = (mask_for_candidate > 0)
            result_dict["class_loss"], _ = self._compute_nll_loss(
                encoder_out_vocab,
                tgt_tokens_full,
                mask_for_candidate,
                self.args.label_smoothing,
                name="class-loss",
                factor=1.0,
                skip_cloest=True,
                return_acc=True,
                log_acc_only=False,
            )

            if self.untouch_token_loss != 0.0:
                mask_for_untouch = (identity_token + untouched_token).long() - token_to_mask_flag.long()
                mask_for_untouch = (mask_for_untouch > 0)
                result_dict["untouch_loss"], _ = self._compute_nll_loss(
                    encoder_out_vocab,
                    tgt_tokens_full,
                    mask_for_untouch,
                    self.args.label_smoothing,
                    name="Untouch-loss",
                    factor=self.untouch_token_loss,
                    skip_cloest=True,
                    return_acc=True,
                    log_acc_only=False,
                )
        elif self.encoder_training_type in ["detector"]:
            vocab_size = encoder_out_vocab.shape[-1]
            encoder_out_sgtgt = (1.0 - F.one_hot(tgt_tokens_full.long(), num_classes=vocab_size).type_as(encoder_out_vocab)) * encoder_out_vocab
            gttoken_labels = torch.zeros(tgt_tokens_full.shape).fill_(self.gttoken).type_as(tgt_tokens_full)

            result_dict["wrong_nonignore_loss"], _ = self._compute_nll_loss(
                encoder_out_vocab,
                tgt_tokens_full,
                bert_generator_encoder_out_label_wrong_nonignore,
                self.args.label_smoothing,
                name="Wrong_nonignore-loss",
                factor=5.0,
                skip_cloest=True,
                return_acc=True,
                log_acc_only=False,
            )

            result_dict["wrong_nonignore_nogttoken_loss"], _ = self._compute_nll_loss(
                encoder_out_vocab,
                tgt_tokens_full,
                bert_generator_encoder_out_label_wrong_nonignore,
                self.args.label_smoothing,
                name="Wrong_nonignore_nogttoken-loss",
                factor=1.0,
                skip_cloest=True,
                return_acc=True,
                log_acc_only=True,
                ignore_gttoken=True
            )

            result_dict["wrong_ignore_loss"], _ = self._compute_nll_loss(
                encoder_out_sgtgt,
                gttoken_labels,
                bert_generator_encoder_out_label_wrong_ignore,
                self.args.label_smoothing,
                name="Wrong_ignore-loss",
                factor=1.0,
                skip_cloest=True,
                return_acc=True,
                log_acc_only=False,
            )

            assert bert_generator_encoder_out_label_diff_ignore is None


            result_dict["right_leak_loss"], _ = self._compute_nll_loss(
                encoder_out_vocab,
                tgt_tokens_full,
                bert_generator_encoder_out_label_right_leak,
                self.args.label_smoothing,
                name="Right_leak-loss",
                factor=1.0,
                skip_cloest=True,
                return_acc=True,
                log_acc_only=False,
            )

            result_dict["right_leak_nogttoken_loss"], _ = self._compute_nll_loss(
                encoder_out_vocab,
                tgt_tokens_full,
                bert_generator_encoder_out_label_right_leak,
                self.args.label_smoothing,
                name="Right_leak_nogttoken-loss",
                factor=1.0,
                skip_cloest=True,
                return_acc=True,
                log_acc_only=True,
                ignore_gttoken=True
            )

            result_dict["right_nonleak_loss"], _ = self._compute_nll_loss(
                encoder_out_sgtgt,
                gttoken_labels,
                bert_generator_encoder_out_label_right_nonleak,
                self.args.label_smoothing,
                name="Right_nonleak-loss",
                factor=1.0,
                skip_cloest=True,
                return_acc=True,
                log_acc_only=False,
            )

        return result_dict






    def forward(
        self, src_tokens, src_lengths, prev_output_tokens, tgt_tokens, wer_dur=None, to_be_edited=None, for_wer_gather=None,
            closest_label=None, source_phone=None, mask_type=None, target_full=None, target_ctc=None, decoder_input=None, encoder_label=None, **kwargs
    ):
        # encoding
        attn_mask = None



        return self.forward_mlm(src_tokens, src_lengths, tgt_tokens, prev_output_tokens, mask_type=mask_type, tgt_tokens_full=target_full, tgt_tokens_ctc=target_ctc)






@register_model_architecture(
    "softcorrect_detector", "softcorrect_detector"
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
    args.src_embedding_copy = getattr(args, "src_embedding_copy", False)
    args.no_scale_embedding = getattr(args, "no_scale_embedding", False)
