# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from fairseq import metrics, utils
from fairseq.criterions import FairseqCriterion, register_criterion
from .label_smoothed_cross_entropy import (
    LabelSmoothedCrossEntropyCriterion,
    LabelSmoothedCrossEntropyCriterionConfig,
)
from fairseq.dataclass import FairseqDataclass
from torch import Tensor
from omegaconf import II

from fairseq.  modules.tts_modules import DurationPredictorLoss

import sys


def label_smoothed_nll_loss(lprobs, target, epsilon, ignore_index=None, reduce=True):
    if target.dim() == lprobs.dim() - 1:
        target = target.unsqueeze(-1)
    nll_loss = -lprobs.gather(dim=-1, index=target)
    smooth_loss = -lprobs.sum(dim=-1, keepdim=True)
    if ignore_index is not None:
        pad_mask = target.eq(ignore_index)
        nll_loss.masked_fill_(pad_mask, 0.0)
        smooth_loss.masked_fill_(pad_mask, 0.0)
    else:
        nll_loss = nll_loss.squeeze(-1)
        smooth_loss = smooth_loss.squeeze(-1)
    if reduce:
        # JL: change sum to mean here
        nll_loss = nll_loss.mean()
        smooth_loss = smooth_loss.mean()
    eps_i = epsilon / (lprobs.size(-1) - 1)
    loss = (1.0 - epsilon - eps_i) * nll_loss + eps_i * smooth_loss
    return loss, nll_loss


@register_criterion(
    "label_smoothed_cross_entropy_speech_length",
    dataclass=LabelSmoothedCrossEntropyCriterionConfig,
)
class LabelSmoothedCrossEntropySpeechLengthCriterion(LabelSmoothedCrossEntropyCriterion):
    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        input: src_tokens B * 
               tgt_tokens B * T
        output: (batch, tgt_len, vocab)
        """
        src_tokens, src_lengths, prev_output_tokens,src_speech_lengths,tgt_speech_lengths, tgt_subwd_lengths = (
            sample["net_input"]["src_tokens"],
            sample["net_input"]["src_lengths"],
            sample["net_input"]["prev_output_tokens"],
            sample["net_input"]["src_speech_lengths"],
            sample["net_input"]["tgt_speech_lengths"],
            sample["net_input"]["tgt_subwd_lengths"]
        )
        tgt_tokens = sample["target"]
       
        net_output, length_output, duration_output = model(src_tokens, src_lengths, prev_output_tokens, tgt_tokens,src_speech_lengths,tgt_speech_lengths,tgt_subwd_lengths)
      
        # length_loss = self.compute_length_loss(length_output["out"], length_output["tgt"], length_output["factor"])
        loss, nll_loss = self.compute_loss(model, net_output, sample, reduce=reduce)
        dur_loss = self.dur_loss(duration_output["pred_dur"], duration_output["tgt_dur"],duration_output["factor"])
        
        # loss = loss + length_loss + dur_loss
        length_loss=0
        loss = loss + dur_loss

        # sample_size = (
        #     sample["target"].size(0) if self.sentence_avg else sample["ntokens"]
        # )
        sample_size = 1
        logging_output = {
            "loss": loss.data,
            "nll_loss": nll_loss.data,
            "length_loss": 0 if length_loss==0 else utils.item(length_loss.data / length_output["factor"]),
            "dur_loss": 0 if dur_loss==0 else utils.item(dur_loss.data / duration_output["factor"]),
            "ntokens": sample["ntokens"],
            "nsentences": sample["target"].size(0),
            "sample_size": sample_size,
        }
        if self.report_accuracy:
            n_correct, total = self.compute_accuracy(model, net_output, sample)
            logging_output["n_correct"] = utils.item(n_correct.data)
            logging_output["total"] = utils.item(total.data)
        return loss, sample_size, logging_output
    
    def compute_length_loss(self, length_out, length_target, factor):
        
        def mean_ds(x: Tensor, dim=None) -> Tensor:
            return (
                x.float().mean().type_as(x)
                if dim is None
                else x.float().mean(dim).type_as(x)
            )
        if length_out is None:
            return 0
        logits = F.log_softmax(length_out, dim=-1)
        if length_target.dim() == 1:
            losses = F.nll_loss(logits, length_target.to(logits.device), reduction="none")
        else:  # soft-labels
            losses = F.kl_div(logits, length_target.to(logits.device), reduction="none")
            losses = losses.sum(-1)
        loss = mean_ds(losses) * factor
        return loss
    
    def dur_loss(self, dur_pred, dur_gt,dur_loss_factor):
        dur_loss_fn = DurationPredictorLoss()
        nonpadding = (dur_gt != 0).float()
        
        ph_dur_loss = dur_loss_fn(dur_pred, dur_gt, nonpadding) * dur_loss_factor
        return ph_dur_loss
       
    def get_lprobs_and_target(self, model, net_output, sample):
        lprobs = model.get_normalized_probs(net_output, log_probs=True)
        target = model.get_targets(sample, net_output)
        if self.ignore_prefix_size > 0:
            if getattr(lprobs, "batch_first", False):
                lprobs = lprobs[:, self.ignore_prefix_size :, :].contiguous()
                target = target[:, self.ignore_prefix_size :].contiguous()
            else:
                lprobs = lprobs[self.ignore_prefix_size :, :, :].contiguous()
                target = target[self.ignore_prefix_size :, :].contiguous()
        return lprobs.view(-1, lprobs.size(-1)), target.view(-1)

    def compute_loss(self, model, net_output, sample, reduce=True):
        lprobs, target = self.get_lprobs_and_target(model, net_output, sample)
        loss, nll_loss = label_smoothed_nll_loss(
            lprobs,
            target,
            self.eps,
            ignore_index=self.padding_idx,
            reduce=reduce,
        )
        return loss, nll_loss

    @classmethod
    def reduce_metrics(cls, logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        nll_loss_sum = sum(log.get("nll_loss", 0) for log in logging_outputs)
        length_loss_sum = sum(log.get("length_loss", 0) for log in logging_outputs)
        dur_loss_sum = sum(log.get("dur_loss", 0) for log in logging_outputs)
        ntokens = sum(log.get("ntokens", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)
        nsentences = sum(log.get("nsentences", 0) for log in logging_outputs)

        metrics.log_scalar(
            "loss", loss_sum / sample_size / math.log(2), sample_size, round=3
        )
        metrics.log_scalar(
            "nll_loss", nll_loss_sum / sample_size / math.log(2), ntokens, round=3
        )
        metrics.log_scalar(
            "length_loss", length_loss_sum / sample_size / math.log(2), sample_size, round=3
        )
        metrics.log_scalar(
            "dur_loss", dur_loss_sum / sample_size / math.log(2), sample_size, round=3
        )
        metrics.log_derived(
            "ppl", lambda meters: utils.get_perplexity(meters["nll_loss"].avg)
        )

        total = utils.item(sum(log.get("total", 0) for log in logging_outputs))
        if total > 0:
            metrics.log_scalar("total", total)
            n_correct = utils.item(
                sum(log.get("n_correct", 0) for log in logging_outputs)
            )
            metrics.log_scalar("n_correct", n_correct)
            metrics.log_derived(
                "accuracy",
                lambda meters: round(
                    meters["n_correct"].sum * 100.0 / meters["total"].sum, 3
                )
                if meters["total"].sum > 0
                else float("nan"),
            )

