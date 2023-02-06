# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import itertools
import logging

logger = logging.getLogger(__name__)

import torch
from fairseq import utils
from fairseq.tasks import register_task
from fairseq.tasks.translation import TranslationTask
from fairseq.utils import new_arange
from fairseq.data import (
    AppendTokenDataset,
    ConcatDataset,
    PrependTokenDataset,
    StripTokenDataset,
    TruncateDataset,
    data_utils,
    encoders,
    indexed_dataset,
)
from fairseq import tokenizer
from dictionary_sc import Dictionary_sc
from corrector_ds import LanguagePairDataset

def load_langpair_dataset(
    data_path,
    split,
    src,
    src_dict,
    tgt,
    tgt_dict,
    combine,
    dataset_impl,
    upsample_primary,
    left_pad_source,
    left_pad_target,
    max_source_positions,
    max_target_positions,
    prepend_bos=False,
    load_alignments=False,
    truncate_source=False,
    append_source_id=False,
    num_buckets=0,
    shuffle=True,
    pad_to_multiple=1,
    cal_wer_dur=False,
    src_with_werdur=False,
    append_eos_to_target=False,
    nbest_infer=0,
    homophone_dict_path="",
    mask_ratio=0.0,
    detector_mask_ratio=0.0,
    error_distribution=None,
    ft_error_distribution=None,
    duptoken_error_distribution=None,
):
    def split_exists(split, src, tgt, lang, data_path):
        filename = os.path.join(data_path, "{}.{}-{}.{}".format(split, src, tgt, lang))
        return indexed_dataset.dataset_exists(filename, impl=dataset_impl)

    src_datasets = []
    tgt_datasets = []

    for k in itertools.count():
        split_k = split + (str(k) if k > 0 else "")

        # infer langcode
        if split_exists(split_k, src, tgt, src, data_path):
            prefix = os.path.join(data_path, "{}.{}-{}.".format(split_k, src, tgt))
        elif split_exists(split_k, tgt, src, src, data_path):
            prefix = os.path.join(data_path, "{}.{}-{}.".format(split_k, tgt, src))
        else:
            if k > 0:
                break
            else:
                raise FileNotFoundError(
                    "Dataset not found: {} ({})".format(split, data_path)
                )

        src_dataset = data_utils.load_indexed_dataset(
            prefix + src, src_dict, dataset_impl
        )
        if truncate_source:
            src_dataset = AppendTokenDataset(
                TruncateDataset(
                    StripTokenDataset(src_dataset, src_dict.eos()),
                    max_source_positions - 1,
                ),
                src_dict.eos(),
            )
        src_datasets.append(src_dataset)

        tgt_dataset = data_utils.load_indexed_dataset(
            prefix + tgt, tgt_dict, dataset_impl
        )
        if tgt_dataset is not None:
            tgt_datasets.append(tgt_dataset)

        logger.info(
            "{} {} {}-{} {} examples".format(
                data_path, split_k, src, tgt, len(src_datasets[-1])
            )
        )

        if not combine:
            break

    assert len(src_datasets) == len(tgt_datasets) or len(tgt_datasets) == 0

    if len(src_datasets) == 1:
        src_dataset = src_datasets[0]
        tgt_dataset = tgt_datasets[0] if len(tgt_datasets) > 0 else None
    else:
        sample_ratios = [1] * len(src_datasets)
        sample_ratios[0] = upsample_primary
        src_dataset = ConcatDataset(src_datasets, sample_ratios)
        if len(tgt_datasets) > 0:
            tgt_dataset = ConcatDataset(tgt_datasets, sample_ratios)
        else:
            tgt_dataset = None

    if prepend_bos:
        assert hasattr(src_dict, "bos_index") and hasattr(tgt_dict, "bos_index")
        src_dataset = PrependTokenDataset(src_dataset, src_dict.bos())
        if tgt_dataset is not None:
            tgt_dataset = PrependTokenDataset(tgt_dataset, tgt_dict.bos())

    eos = None
    if append_source_id:
        src_dataset = AppendTokenDataset(
            src_dataset, src_dict.index("[{}]".format(src))
        )
        if tgt_dataset is not None:
            tgt_dataset = AppendTokenDataset(
                tgt_dataset, tgt_dict.index("[{}]".format(tgt))
            )
        eos = tgt_dict.index("[{}]".format(tgt))

    align_dataset = None
    if load_alignments:
        align_path = os.path.join(data_path, "{}.align.{}-{}".format(split, src, tgt))
        if indexed_dataset.dataset_exists(align_path, impl=dataset_impl):
            align_dataset = data_utils.load_indexed_dataset(
                align_path, None, dataset_impl
            )

    tgt_dataset_sizes = tgt_dataset.sizes if tgt_dataset is not None else None
    return LanguagePairDataset(
        src_dataset,
        src_dataset.sizes,
        src_dict,
        tgt_dataset,
        tgt_dataset_sizes,
        tgt_dict,
        left_pad_source=left_pad_source,
        left_pad_target=left_pad_target,
        align_dataset=align_dataset,
        eos=eos,
        num_buckets=num_buckets,
        shuffle=shuffle,
        pad_to_multiple=pad_to_multiple,
        cal_wer_dur=cal_wer_dur,
        src_with_werdur=src_with_werdur,
        append_eos_to_target=append_eos_to_target,
        bos_prepended_outside=prepend_bos,
        nbest_infer=nbest_infer,
        homophone_dict_path=homophone_dict_path,
        mask_ratio=mask_ratio,
        detector_mask_ratio=detector_mask_ratio,
        error_distribution=error_distribution,
        ft_error_distribution=ft_error_distribution,
        duptoken_error_distribution=duptoken_error_distribution,
    )

@register_task("softcorrect_task")
class SoftcorrectTask(TranslationTask):
    """
    Translation (Sequence Generation) task for Levenshtein Transformer
    See `"Levenshtein Transformer" <https://arxiv.org/abs/1905.11006>`_.
    """

    @staticmethod
    def add_args(parser):
        """Add task-specific arguments to the parser."""
        # fmt: off
        TranslationTask.add_args(parser)
        parser.add_argument(
            '--noise',
            default='random_delete',
            choices=['random_delete', 'random_mask', 'no_noise', 'full_mask'])
        parser.add_argument(
            '--cal-wer-dur', action="store_true", default=False,
           help='Whether to cal-wer-dur in dataset')
        parser.add_argument(
            '--use-wer-dur', action="store_true", default=False,
           help='Whether to use wer dur in model')
        parser.add_argument(
            '--src-with-werdur', action="store_true", default=False,
           help='Whether the werdur is in dataset')
        parser.add_argument(
            '--break-alignment', action="store_true", default=False,
            help="break the alignment, using single padding"
        )
        parser.add_argument("--pos-before-reshape", action="store_true", default=False,
                           help="whether apply pos embedding before reshape")
        parser.add_argument(
            "--homophone-dict-path",
            type=str,
            default="",
            help="path to the homophone dict",
        )
        parser.add_argument(
            "--mask-ratio",
            type=float,
            default=0.0,
            help="mask rato",
        )
        parser.add_argument(
            "--detector-mask-ratio",
            type=float,
            default=0.0,
            help="mask ratio of detector",
        )
        parser.add_argument(
            "--error-distribution",
            type=str,
            default=None,
            help="simulated error distribution",
        )
        parser.add_argument(
            "--ft-error-distribution",
            type=str,
            default=None,
            help="error distribution in finetune",
        )
        parser.add_argument(
            "--duptoken-error-distribution",
            type=str,
            default=None,
            help="error distribution of duped token",
        )
        parser.add_argument(
            '--untouch-token-loss', type=float,
            default=0.0,
            help='lambda of untouch token'
        )
        parser.add_argument('--pad-first-dictionary', action='store_true',
            help='in dictionary, whether pad is index 0')
        parser.add_argument(
            '--candidate-size', type=int,
            default=0,
            help='number of candidate number'
        )
        parser.add_argument(
            '--label-leak-prob', type=float,
            default=0.0,
            help='prob of label leak'
        )
        parser.add_argument(
            '--label-ignore-prob', type=float,
            default=0.0,
            help='prob of ignore label (predict nota) when wrong'
        )
        parser.add_argument(
            '--bert-generator-encoder-model-path', type=str,
            default="",
            help='path of bert_generator encoder model'
        )
        parser.add_argument(
            '--main-encoder-warmup-path', type=str,
            default="",
            help='path of warmup main encoder model'
        )
        parser.add_argument(
            '--nbest-void-insert-ratio', type=float,
            default=0.0,
            help='ratio of adding rand <void> into nbest input'
        )
        parser.add_argument(
            '--nbest-input-num', type=int,
            default=1,
            help='number of nbest input'
        )
        parser.add_argument(
            '--nbest-input-sample-temp', type=float,
            default=1.0,
            help='temperature of sampling nbest input'
        )
        parser.add_argument(
            '--nbest-input-sample-untouch-temp', type=float,
            default=-1.0,
            help='temperature of sampling nbest input if the source toke is unnoised'
        )
        parser.add_argument(
            '--encoder-training-type', type=str,
            default="detector",
            help='loss type of encoder training'
        )
        parser.add_argument(
            '--force-same-ratio', type=float,
            default=0.0,
            help='nbest ratio that is forced to same'
        )
        parser.add_argument(
            '--emb-dropout', type=float, default=-1.0,
            help='dropout of encoder embedding dropout'
        )
        parser.add_argument(
            '--same-also-sample', type=float, default=0.0,
            help='same also sample'
        )
        # fmt: on

    def load_dataset(self, split, epoch=1, combine=False, **kwargs):
        """Load a given dataset split.

        Args:
            split (str): name of the split (e.g., train, valid, test)
        """
        paths = utils.split_paths(self.args.data)
        assert len(paths) > 0
        data_path = paths[(epoch - 1) % len(paths)]

        # infer langcode
        src, tgt = self.args.source_lang, self.args.target_lang

        self.datasets[split] = load_langpair_dataset(
            data_path,
            split,
            src,
            self.src_dict,
            tgt,
            self.tgt_dict,
            combine=combine,
            dataset_impl=self.args.dataset_impl,
            upsample_primary=self.args.upsample_primary,
            left_pad_source=self.args.left_pad_source,
            left_pad_target=self.args.left_pad_target,
            max_source_positions=self.args.max_source_positions,
            max_target_positions=self.args.max_target_positions,
            prepend_bos=True,
            cal_wer_dur=self.args.cal_wer_dur,
            src_with_werdur=self.args.src_with_werdur,
            append_eos_to_target=self.args.cal_wer_dur ,  # add this although eos already add in data preprocess
            homophone_dict_path=self.args.homophone_dict_path,
            mask_ratio=self.args.mask_ratio,
            detector_mask_ratio=self.args.detector_mask_ratio,
            error_distribution=(None if not self.args.error_distribution else [float(i) for i in self.args.error_distribution.split(",")]),
            ft_error_distribution=(None if not self.args.ft_error_distribution else [float(i) for i in self.args.ft_error_distribution.split(",")]),
            duptoken_error_distribution=(None if not self.args.duptoken_error_distribution else [float(i) for i in self.args.duptoken_error_distribution.split(",")]),
        )

    @classmethod
    def setup_task(cls, args, **kwargs):
        """Setup the task (e.g., load dictionaries).

        Args:
            args (argparse.Namespace): parsed command-line arguments
        """
        args.left_pad_source = utils.eval_bool(args.left_pad_source)
        args.left_pad_target = utils.eval_bool(args.left_pad_target)

        paths = utils.split_paths(args.data)
        assert len(paths) > 0
        # find language pair automatically
        if args.source_lang is None or args.target_lang is None:
            args.source_lang, args.target_lang = data_utils.infer_language_pair(
                paths[0]
            )
        if args.source_lang is None or args.target_lang is None:
            raise Exception(
                "Could not infer language pair, please provide it explicitly"
            )

        # load dictionaries
        src_dict = cls.load_dictionary(
            os.path.join(paths[0], "dict.{}.txt".format(args.source_lang)), pad_first=args.pad_first_dictionary
        )
        tgt_dict = cls.load_dictionary(
            os.path.join(paths[0], "dict.{}.txt".format(args.target_lang)), pad_first=args.pad_first_dictionary
        )
        assert src_dict.pad() == tgt_dict.pad()
        assert src_dict.eos() == tgt_dict.eos()
        assert src_dict.unk() == tgt_dict.unk()
        logger.info("[{}] dictionary: {} types".format(args.source_lang, len(src_dict)))
        logger.info("[{}] dictionary: {} types".format(args.target_lang, len(tgt_dict)))

        return cls(args, src_dict, tgt_dict)

    @classmethod
    def load_dictionary(cls, filename, pad_first=False):
        """Load the dictionary from the filename

        Args:
            filename (str): the filename
        """
        return Dictionary_sc.load(filename, pad_first=pad_first)

    @classmethod
    def build_dictionary(
        cls, filenames, workers=1, threshold=-1, nwords=-1, padding_factor=8
    ):
        """Build the dictionary

        Args:
            filenames (list): list of filenames
            workers (int): number of concurrent workers
            threshold (int): defines the minimum word count
            nwords (int): defines the total number of words in the final dictionary,
                including special symbols
            padding_factor (int): can be used to pad the dictionary size to be a
                multiple of 8, which is important on some hardware (e.g., Nvidia
                Tensor Cores).
        """
        d = Dictionary_sc()
        for filename in filenames:
            Dictionary_sc.add_file_to_dictionary(
                filename, d, tokenizer.tokenize_line, workers
            )
        d.finalize(threshold=threshold, nwords=nwords, padding_factor=padding_factor)
        return d

    def inject_noise(self, target_tokens):
        def _random_delete(target_tokens):
            pad = self.tgt_dict.pad()
            bos = self.tgt_dict.bos()
            eos = self.tgt_dict.eos()

            max_len = target_tokens.size(1)
            target_mask = target_tokens.eq(pad)
            target_score = target_tokens.clone().float().uniform_()
            target_score.masked_fill_(
                target_tokens.eq(bos) | target_tokens.eq(eos), 0.0
            )
            target_score.masked_fill_(target_mask, 1)
            target_score, target_rank = target_score.sort(1)
            target_length = target_mask.size(1) - target_mask.float().sum(
                1, keepdim=True
            )

            # do not delete <bos> and <eos> (we assign 0 score for them)
            target_cutoff = (
                2
                + (
                    (target_length - 2)
                    * target_score.new_zeros(target_score.size(0), 1).uniform_()
                ).long()
            )
            target_cutoff = target_score.sort(1)[1] >= target_cutoff

            prev_target_tokens = (
                target_tokens.gather(1, target_rank)
                .masked_fill_(target_cutoff, pad)
                .gather(1, target_rank.masked_fill_(target_cutoff, max_len).sort(1)[1])
            )
            prev_target_tokens = prev_target_tokens[
                :, : prev_target_tokens.ne(pad).sum(1).max()
            ]

            return prev_target_tokens

        def _random_mask(target_tokens):
            pad = self.tgt_dict.pad()
            bos = self.tgt_dict.bos()
            eos = self.tgt_dict.eos()
            unk = self.tgt_dict.unk()

            target_masks = (
                target_tokens.ne(pad) & target_tokens.ne(bos) & target_tokens.ne(eos)
            )
            target_score = target_tokens.clone().float().uniform_()
            target_score.masked_fill_(~target_masks, 2.0)
            target_length = target_masks.sum(1).float()
            target_length = target_length * target_length.clone().uniform_()
            target_length = target_length + 1  # make sure to mask at least one token.

            _, target_rank = target_score.sort(1)
            target_cutoff = new_arange(target_rank) < target_length[:, None].long()
            prev_target_tokens = target_tokens.masked_fill(
                target_cutoff.scatter(1, target_rank, target_cutoff), unk
            )
            return prev_target_tokens

        def _full_mask(target_tokens):
            pad = self.tgt_dict.pad()
            bos = self.tgt_dict.bos()
            eos = self.tgt_dict.eos()
            unk = self.tgt_dict.unk()

            target_mask = (
                target_tokens.eq(bos) | target_tokens.eq(eos) | target_tokens.eq(pad)
            )
            return target_tokens.masked_fill(~target_mask, unk)

        if self.args.noise == "random_delete":
            return _random_delete(target_tokens)
        elif self.args.noise == "random_mask":
            return _random_mask(target_tokens)
        elif self.args.noise == "full_mask":
            return _full_mask(target_tokens)
        elif self.args.noise == "no_noise":
            return target_tokens
        else:
            raise NotImplementedError

    def build_generator(self, models, args, **unused):
        # add models input to match the API for SequenceGenerator
        from softcorrect_corrector_generator import SoftcorrectCorrectorGenerator

        # print("edit_thre:", getattr(args, "edit_thre", 0.0))

        return SoftcorrectCorrectorGenerator(
            self.target_dictionary,
            eos_penalty=getattr(args, "iter_decode_eos_penalty", 0.0),
            max_iter=getattr(args, "iter_decode_max_iter", 10),
            beam_size=getattr(args, "iter_decode_with_beam", 1),
            reranking=getattr(args, "iter_decode_with_external_reranker", False),
            decoding_format=getattr(args, "decoding_format", None),
            adaptive=not getattr(args, "iter_decode_force_max_iter", False),
            retain_history=getattr(args, "retain_iter_history", False),
            edit_thre=getattr(args, "edit_thre", 0.0),
            print_werdur=getattr(args, "print_werdur", False),
            retain_dropout=getattr(args, "retain_dropout", False)
        )

    def build_dataset_for_inference(self, src_tokens, src_lengths, constraints=None, nbest_infer=0, force_mask_type=None, duptoken_error_distribution=None):
        if constraints is not None:
            # Though see Susanto et al. (ACL 2020): https://www.aclweb.org/anthology/2020.acl-main.325/
            raise NotImplementedError(
                "Constrained decoding with the translation_lev task is not supported"
            )

        return LanguagePairDataset(
            src_tokens, src_lengths, self.source_dictionary, append_bos=True, homophone_dict_path="", nbest_infer=nbest_infer,  force_mask_type=force_mask_type, duptoken_error_distribution=duptoken_error_distribution
        )

    def train_step(
        self, sample, model, criterion, optimizer, update_num, ignore_grad=False
    ):
        model.train()
        sample["prev_target"] = self.inject_noise(sample["target"])
        loss, sample_size, logging_output = criterion(model, sample)
        if ignore_grad:
            loss *= 0
        optimizer.backward(loss)
        return loss, sample_size, logging_output

    def valid_step(self, sample, model, criterion):
        model.eval()
        with torch.no_grad():
            sample["prev_target"] = self.inject_noise(sample["target"])
            loss, sample_size, logging_output = criterion(model, sample)
        return loss, sample_size, logging_output

    def inference_step(
        self, generator, models, sample, prefix_tokens=None, constraints=None, werdur_gt_str="", force_mask_type=None, duptoken_error_distribution=None
    ):
        with torch.no_grad():
            return generator.generate(
                models, sample, prefix_tokens=prefix_tokens, constraints=constraints, werdur_gt_str=werdur_gt_str, force_mask_type=force_mask_type
            )  # can use try except to prevent paramter error
