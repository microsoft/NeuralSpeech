# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from collections import namedtuple

import numpy as np
import torch
from fairseq import utils
import time

DecoderOut = namedtuple(
    "FastCorrectDecoderOut",
    ["output_tokens", "output_scores", "attn", "step", "max_step", "history", "to_be_edited_pred", "wer_dur_pred"],
)


class SoftcorrectCorrectorGenerator(object):
    def __init__(
        self,
        tgt_dict,
        models=None,
        eos_penalty=0.0,
        max_iter=10,
        max_ratio=2,
        beam_size=1,
        decoding_format=None,
        retain_dropout=False,
        adaptive=True,
        retain_history=False,
        reranking=False,
        edit_thre=0.0,
        print_werdur=False
    ):
        """
        Generates translations based on iterative refinement.

        Args:
            tgt_dict: target dictionary
            eos_penalty: if > 0.0, it penalized early-stopping in decoding
            max_iter: maximum number of refinement iterations
            max_ratio: generate sequences of maximum length ax, where x is the source length
            decoding_format: decoding mode in {'unigram', 'ensemble', 'vote', 'dp', 'bs'}
            retain_dropout: retaining dropout in the inference
            adaptive: decoding with early stop
        """
        self.bos = tgt_dict.bos()
        self.pad = tgt_dict.pad()
        self.unk = tgt_dict.unk()
        self.eos = tgt_dict.eos()
        self.vocab_size = len(tgt_dict)
        self.vocab_symbols = tgt_dict.symbols
        self.eos_penalty = eos_penalty
        self.max_iter = max_iter
        self.max_ratio = max_ratio
        self.beam_size = beam_size
        self.reranking = reranking
        self.decoding_format = decoding_format
        self.retain_dropout = retain_dropout
        self.retain_history = retain_history
        self.adaptive = adaptive
        self.models = models
        self.edit_thre = edit_thre
        self.print_werdur = print_werdur

    def generate_batched_itr(
        self,
        data_itr,
        maxlen_a=None,
        maxlen_b=None,
        cuda=False,
        timer=None,
        prefix_size=0,
    ):
        """Iterate over a batched dataset and yield individual translations.

        Args:
            maxlen_a/b: generate sequences of maximum length ax + b,
                where x is the source sentence length.
            cuda: use GPU for generation
            timer: StopwatchMeter for timing generations.
        """

        for sample in data_itr:
            if "net_input" not in sample:
                continue
            if timer is not None:
                timer.start()
            with torch.no_grad():
                hypos = self.generate(
                    self.models,
                    sample,
                    prefix_tokens=sample["target"][:, :prefix_size]
                    if prefix_size > 0
                    else None,
                )
            if timer is not None:
                timer.stop(sample["ntokens"])
            for i, id in enumerate(sample["id"]):
                # remove padding
                src = utils.strip_pad(sample["net_input"]["src_tokens"][i, :], self.pad)
                ref = utils.strip_pad(sample["target"][i, :], self.pad)
                yield id, src, ref, hypos[i]

    @torch.no_grad()
    def generate(self, models, sample, prefix_tokens=None, constraints=None, werdur_gt_str="", force_mask_type=None):
        if constraints is not None:
            raise NotImplementedError(
                "Constrained decoding with the IterativeRefinementGenerator is not supported"
            )

        # TODO: iterative refinement generator does not support ensemble for now.
        if not self.retain_dropout:
            for model in models:
                model.eval()

        model, reranker = models[0], None
        if self.reranking:
            assert len(models) > 1, "Assuming the last checkpoint is the reranker"
            assert (
                self.beam_size > 1
            ), "Reranking requires multiple translation for each example"

            reranker = models[-1]
            models = models[:-1]

        if len(models) > 1 and hasattr(model, "enable_ensemble"):
            assert model.allow_ensemble, "{} does not support ensembling".format(
                model.__class__.__name__
            )
            model.enable_ensemble(models)

        # TODO: better encoder inputs?
        src_tokens = sample["net_input"]["src_tokens"]
        src_lengths = sample["net_input"]["src_lengths"]
        if "mask_type" in sample["net_input"]:
            mask_type = sample["net_input"]["mask_type"]
        else:
            mask_type = None
        #print("mask_type:", mask_type)
        #print("src_tokens:", src_tokens)
        bsz, src_len = src_tokens.size()[:2]



        # initialize
        # print("before encoder:", time.time())
        encoder_out = model.forward_encoder([src_tokens, src_lengths, mask_type])
        if mask_type is not None:
            if force_mask_type == "dec_infer":
                assert len(encoder_out.shape) == 3
                if len(src_tokens.shape) == 2:
                    tgt_logit = torch.gather(encoder_out, dim=-1,
                                             index=sample["net_input"]["target_full"][:, :, None]).squeeze(-1)
                    tgt_rank = (encoder_out > tgt_logit[:, :, None]).long().sum(-1) + 1
                    _, tgt_top5 = torch.topk(encoder_out, 5, dim=-1)
                    tgt_top5 = [[model.encoder.dictionary[int(tgt_top5[0][i][j])] for j in range(5)]for i in range(len(tgt_top5[0]))]
                elif len(src_tokens.shape) == 3:
                    tgt_logit = torch.gather(encoder_out, dim=-1, index=src_tokens)
                    tgt_rank = (encoder_out > tgt_logit[:, :, 0:1]).long().sum(-1) + 1
                    _, tgt_top5 = torch.topk(encoder_out, 5, dim=-1)
                    tgt_top5 = [[model.encoder.dictionary[int(tgt_top5[0][i][j])] for j in range(5)] for i in range(len(tgt_top5[0]))]
                else:
                    raise ValueError("Impossible input shape {}".format(src_tokens.shape))
                return [
                    [
                        {
                            'steps': 0,
                            'tokens': tgt_logit.cpu(),
                            'positional_scores': None,
                            'score': tgt_logit.sum(),
                            'hypo_attn': None,
                            'alignment': None
                        }
                    ]
                ]
            hypo_tokens = encoder_out[0].max(-1)[1]
            final_tokens = []
            assert src_tokens.shape[1] == mask_type.shape[1] == hypo_tokens.shape[0]
            assert src_tokens.shape[0] == 1
            last_CTC_token = None
            for iter_token in range(len(hypo_tokens)):
                if int(mask_type[0][iter_token]) in [0, 1, 2]:
                    final_tokens.append(src_tokens[0][iter_token])
                    last_CTC_token = None
                elif int(mask_type[0][iter_token]) in [3]:
                    final_tokens.append(hypo_tokens[iter_token])
                    last_CTC_token = None
                else:
                    assert int(mask_type[0][iter_token]) in [4]
                    if hypo_tokens[iter_token] != 0 and hypo_tokens[iter_token] != last_CTC_token:
                        last_CTC_token = hypo_tokens[iter_token]
                        final_tokens.append(hypo_tokens[iter_token])
                    elif hypo_tokens[iter_token] == 0:
                        last_CTC_token = None


            finalized = [
                [
                    {
                        'steps': 0,
                        'tokens': torch.LongTensor(final_tokens),
                        'positional_scores': None,
                        'score': encoder_out[0].max(-1)[0].sum(),
                        'hypo_attn': None,
                        'alignment': None
                        }
                ]
            ]
            return finalized
        else:
            raise ValueError("Must have mask type")


    def rerank(self, reranker, finalized, encoder_input, beam_size):
        def rebuild_batch(finalized):
            finalized_tokens = [f[0]["tokens"] for f in finalized]
            finalized_maxlen = max(f.size(0) for f in finalized_tokens)
            final_output_tokens = (
                finalized_tokens[0]
                .new_zeros(len(finalized_tokens), finalized_maxlen)
                .fill_(self.pad)
            )
            for i, f in enumerate(finalized_tokens):
                final_output_tokens[i, : f.size(0)] = f
            return final_output_tokens

        final_output_tokens = rebuild_batch(finalized)
        final_output_tokens[
            :, 0
        ] = self.eos  # autoregressive model assumes starting with EOS

        reranker_encoder_out = reranker.encoder(*encoder_input)
        length_beam_order = (
            utils.new_arange(
                final_output_tokens, beam_size, reranker_encoder_out.encoder_out.size(1)
            )
            .t()
            .reshape(-1)
        )
        reranker_encoder_out = reranker.encoder.reorder_encoder_out(
            reranker_encoder_out, length_beam_order
        )
        reranking_scores = reranker.get_normalized_probs(
            reranker.decoder(final_output_tokens[:, :-1], reranker_encoder_out),
            True,
            None,
        )
        reranking_scores = reranking_scores.gather(2, final_output_tokens[:, 1:, None])
        reranking_masks = final_output_tokens[:, 1:].ne(self.pad)
        reranking_scores = (
            reranking_scores[:, :, 0].masked_fill_(~reranking_masks, 0).sum(1)
        )
        reranking_scores = reranking_scores / reranking_masks.sum(1).type_as(
            reranking_scores
        )

        for i in range(len(finalized)):
            finalized[i][0]["score"] = reranking_scores[i]

        return finalized
