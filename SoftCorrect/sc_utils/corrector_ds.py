# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging

import numpy as np
import torch
from fairseq.data import FairseqDataset
import random                                           
import math
import copy
import json
import data_utils_sc

logger = logging.getLogger(__name__)


def collate(
    samples,
    pad_idx,
    eos_idx,
    left_pad_source=True,
    left_pad_target=False,
    input_feeding=True,
    pad_to_length=None,
    pad_to_multiple=1,
):
    if len(samples) == 0:
        return {}

    def merge(key, left_pad, move_eos_to_beginning=False, pad_to_length=None, merge_phone=False):
        if merge_phone:
            return data_utils_sc.collate_2d_phones(
                [s[key] for s in samples],
                pad_idx,
                eos_idx,
                left_pad,
                move_eos_to_beginning,
                pad_to_length=pad_to_length,
                pad_to_multiple=pad_to_multiple,
            )
        elif len(samples[0][key].shape) == 1:
            return data_utils_sc.collate_tokens(
                [s[key] for s in samples],
                pad_idx,
                eos_idx,
                left_pad,
                move_eos_to_beginning,
                pad_to_length=pad_to_length,
                pad_to_multiple=pad_to_multiple,
            )
        elif len(samples[0][key].shape) == 2:
            return data_utils_sc.collate_2d_tokens(
                [s[key] for s in samples],
                pad_idx,
                eos_idx,
                left_pad,
                move_eos_to_beginning,
                pad_to_length=pad_to_length,
                pad_to_multiple=pad_to_multiple,
            )
        else:
            raise ValueError("Unsupported condition!")

    def check_alignment(alignment, src_len, tgt_len):
        if alignment is None or len(alignment) == 0:
            return False
        if (
            alignment[:, 0].max().item() >= src_len - 1
            or alignment[:, 1].max().item() >= tgt_len - 1
        ):
            logger.warning("alignment size mismatch found, skipping alignment!")
            return False
        return True

    def compute_alignment_weights(alignments):
        """
        Given a tensor of shape [:, 2] containing the source-target indices
        corresponding to the alignments, a weight vector containing the
        inverse frequency of each target index is computed.
        For e.g. if alignments = [[5, 7], [2, 3], [1, 3], [4, 2]], then
        a tensor containing [1., 0.5, 0.5, 1] should be returned (since target
        index 3 is repeated twice)
        """
        align_tgt = alignments[:, 1]
        _, align_tgt_i, align_tgt_c = torch.unique(
            align_tgt, return_inverse=True, return_counts=True
        )
        align_weights = align_tgt_c[align_tgt_i[np.arange(len(align_tgt))]]
        return 1.0 / align_weights.float()

    id = torch.LongTensor([s["id"] for s in samples])
    src_tokens = merge(
        "source",
        left_pad=left_pad_source,
        pad_to_length=pad_to_length["source"] if pad_to_length is not None else None,
    )
    # sort by descending source length
    if len(samples[0]["source"].shape) == 1:
        src_lengths = torch.LongTensor(
            [s["source"].ne(pad_idx).long().sum() for s in samples]
        )
    elif len(samples[0]["source"].shape) == 2:
        src_lengths = torch.LongTensor(
            [s["source"][:, 0].ne(pad_idx).long().sum() for s in samples]
        )
    else:
        raise ValueError("Unsupported condition!")
    src_lengths, sort_order = src_lengths.sort(descending=True)
    id = id.index_select(0, sort_order)
    src_tokens = src_tokens.index_select(0, sort_order)

    prev_output_tokens = None
    target = None

    if samples[0].get("wer_dur", None) is not None:
        wer_dur = merge(
            "wer_dur",
            left_pad=left_pad_source,
            pad_to_length=pad_to_length["source"] if pad_to_length is not None else None,
        )
        wer_dur = wer_dur.index_select(0, sort_order)
        to_be_edited = merge(
            "to_be_edited",
            left_pad=left_pad_source,
            pad_to_length=pad_to_length["source"] if pad_to_length is not None else None,
        )
        to_be_edited = to_be_edited.index_select(0, sort_order)
        if samples[0].get("closest_label", None) is not None:
            closest_label = merge(
                "closest_label",
                left_pad=left_pad_source,
                pad_to_length=pad_to_length["source"] if pad_to_length is not None else None,
            )
            closest_label = closest_label.index_select(0, sort_order)
        else:
            closest_label = None
    else:
        closest_label = None
        wer_dur = None
        to_be_edited = None

    if samples[0].get("source_phone", None) is not None:
        source_phone = merge(
            "source_phone",
            left_pad=left_pad_source,
            pad_to_length=pad_to_length["source"] if pad_to_length is not None else None,
            merge_phone=True
        )
        source_phone = source_phone.index_select(0, sort_order)
    else:
        source_phone = None

    if samples[0].get("mask_type", None) is not None:
        mask_type = merge(
            "mask_type",
            left_pad=left_pad_source,
            pad_to_length=pad_to_length["source"] if pad_to_length is not None else None,
        )
        mask_type = mask_type.index_select(0, sort_order)
    else:
        mask_type = None

    if samples[0].get("target_full", None) is not None:
        target_full = merge(
            "target_full",
            left_pad=left_pad_source,
            pad_to_length=pad_to_length["source"] if pad_to_length is not None else None,
        )
        target_full = target_full.index_select(0, sort_order)
    else:
        target_full = None

    if samples[0].get("target_ctc", None) is not None:
        target_ctc = merge(
            "target_ctc",
            left_pad=left_pad_source,
            pad_to_length=pad_to_length["source"] if pad_to_length is not None else None,
        )
        target_ctc = target_ctc.index_select(0, sort_order)
    else:
        target_ctc = None

    if samples[0].get("target", None) is not None:
        target = merge(
            "target",
            left_pad=left_pad_target,
            pad_to_length=pad_to_length["target"]
            if pad_to_length is not None
            else None,
        )
        target = target.index_select(0, sort_order)
        if samples[0].get("for_wer_gather", None) is not None:
            for_wer_gather = merge(
                "for_wer_gather",
                left_pad=left_pad_target,
                pad_to_length=pad_to_length["target"]
                if pad_to_length is not None
                else None,
            )
            for_wer_gather = for_wer_gather.index_select(0, sort_order)
        else:
            for_wer_gather = None

        tgt_lengths = torch.LongTensor(
            [s["target"].ne(pad_idx).long().sum() for s in samples]
        ).index_select(0, sort_order)
        ntokens = tgt_lengths.sum().item()

        if samples[0].get("prev_output_tokens", None) is not None:
            prev_output_tokens = merge("prev_output_tokens", left_pad=left_pad_target)
        elif input_feeding:
            # we create a shifted version of targets for feeding the
            # previous output token(s) into the next decoder step
            prev_output_tokens = merge(
                "target",
                left_pad=left_pad_target,
                move_eos_to_beginning=True,
                pad_to_length=pad_to_length["target"]
                if pad_to_length is not None
                else None,
            )
    else:
        ntokens = src_lengths.sum().item()
        for_wer_gather = None

    if samples[0].get("wer_dur", None) is not None:
        #print("source_phone in colltor:", samples[0]["source_phone"])
        batch = {
            "id": id,
            "nsentences": len(samples),
            "ntokens": ntokens,
            "net_input": {
                "src_tokens": src_tokens,
                "src_lengths": src_lengths,
                "wer_dur": wer_dur,
                "to_be_edited": to_be_edited,
                "for_wer_gather": for_wer_gather,
                "closest_label": closest_label,
                "source_phone": source_phone,
                "mask_type": mask_type,
                "target_full": target_full,
                "target_ctc": target_ctc,
            },
            "target": target,
        }
    else:
        batch = {
            "id": id,
            "nsentences": len(samples),
            "ntokens": ntokens,
            "net_input": {
                "src_tokens": src_tokens,
                "src_lengths": src_lengths,
                "source_phone": source_phone,
                "mask_type": mask_type,
                "target_full": target_full,
                "target_ctc": target_ctc,
            },
            "target": target,
        }
    if prev_output_tokens is not None:
        batch["net_input"]["prev_output_tokens"] = prev_output_tokens.index_select(
            0, sort_order
        )

    if samples[0].get("alignment", None) is not None:
        bsz, tgt_sz = batch["target"].shape
        src_sz = batch["net_input"]["src_tokens"].shape[1]

        offsets = torch.zeros((len(sort_order), 2), dtype=torch.long)
        offsets[:, 1] += torch.arange(len(sort_order), dtype=torch.long) * tgt_sz
        if left_pad_source:
            offsets[:, 0] += src_sz - src_lengths
        if left_pad_target:
            offsets[:, 1] += tgt_sz - tgt_lengths

        alignments = [
            alignment + offset
            for align_idx, offset, src_len, tgt_len in zip(
                sort_order, offsets, src_lengths, tgt_lengths
            )
            for alignment in [samples[align_idx]["alignment"].view(-1, 2)]
            if check_alignment(alignment, src_len, tgt_len)
        ]

        if len(alignments) > 0:
            alignments = torch.cat(alignments, dim=0)
            align_weights = compute_alignment_weights(alignments)

            batch["alignments"] = alignments
            batch["align_weights"] = align_weights

    if samples[0].get("constraints", None) is not None:
        # Collate the packed constraints across the samples, padding to
        # the length of the longest sample.
        lens = [sample.get("constraints").size(0) for sample in samples]
        max_len = max(lens)
        constraints = torch.zeros((len(samples), max(lens))).long()
        for i, sample in enumerate(samples):
            constraints[i, 0 : lens[i]] = samples[i].get("constraints")
        batch["constraints"] = constraints

    return batch


class LanguagePairDataset(FairseqDataset):
    """
    A pair of torch.utils.data.Datasets.

    Args:
        src (torch.utils.data.Dataset): source dataset to wrap
        src_sizes (List[int]): source sentence lengths
        src_dict (~fairseq.data.Dictionary): source vocabulary
        tgt (torch.utils.data.Dataset, optional): target dataset to wrap
        tgt_sizes (List[int], optional): target sentence lengths
        tgt_dict (~fairseq.data.Dictionary, optional): target vocabulary
        left_pad_source (bool, optional): pad source tensors on the left side
            (default: True).
        left_pad_target (bool, optional): pad target tensors on the left side
            (default: False).
        shuffle (bool, optional): shuffle dataset elements before batching
            (default: True).
        input_feeding (bool, optional): create a shifted version of the targets
            to be passed into the model for teacher forcing (default: True).
        remove_eos_from_source (bool, optional): if set, removes eos from end
            of source if it's present (default: False).
        append_eos_to_target (bool, optional): if set, appends eos to end of
            target if it's absent (default: False).
        align_dataset (torch.utils.data.Dataset, optional): dataset
            containing alignments.
        constraints (Tensor, optional): 2d tensor with a concatenated, zero-
            delimited list of constraints for each sentence.
        append_bos (bool, optional): if set, appends bos to the beginning of
            source/target sentence.
        num_buckets (int, optional): if set to a value greater than 0, then
            batches will be bucketed into the given number of batch shapes.
        src_lang_id (int, optional): source language ID, if set, the collated batch
            will contain a field 'src_lang_id' in 'net_input' which indicates the
            source language of the samples.
        tgt_lang_id (int, optional): target language ID, if set, the collated batch
            will contain a field 'tgt_lang_id' which indicates the target language
             of the samples.
    """

    def __init__(
        self,
        src,
        src_sizes,
        src_dict,
        tgt=None,
        tgt_sizes=None,
        tgt_dict=None,
        left_pad_source=True,
        left_pad_target=False,
        shuffle=True,
        input_feeding=True,
        remove_eos_from_source=False,
        append_eos_to_target=False,
        align_dataset=None,
        constraints=None,
        append_bos=False,
        eos=None,
        num_buckets=0,
        src_lang_id=None,
        tgt_lang_id=None,
        pad_to_multiple=1,
        cal_wer_dur=False,
        src_with_werdur=False,
        bos_prepended_outside=False,
        nbest_infer=0,
        homophone_dict_path="",
        mask_ratio=0.0,
        detector_mask_ratio=0.0,
        error_distribution=None,
        ft_error_distribution=None,
        duptoken_error_distribution=None,
        force_mask_type=None,
    ):
        if tgt_dict is not None:
            assert src_dict.pad() == tgt_dict.pad()
            assert src_dict.eos() == tgt_dict.eos()
            assert src_dict.unk() == tgt_dict.unk()
        if tgt is not None:
            assert len(src) == len(
                tgt
            ), "Source and target must contain the same number of examples"
        self.src = src
        self.tgt = tgt
        self.src_sizes = np.array(src_sizes)
        self.tgt_sizes = np.array(tgt_sizes) if tgt_sizes is not None else None
        self.sizes = (
            np.vstack((self.src_sizes, self.tgt_sizes)).T
            if self.tgt_sizes is not None
            else self.src_sizes
        )
        self.src_dict = src_dict
        self.tgt_dict = tgt_dict
        self.left_pad_source = left_pad_source
        self.left_pad_target = left_pad_target
        self.shuffle = shuffle
        self.input_feeding = input_feeding
        self.remove_eos_from_source = remove_eos_from_source
        self.append_eos_to_target = append_eos_to_target
        self.align_dataset = align_dataset
        if self.align_dataset is not None:
            assert (
                self.tgt_sizes is not None
            ), "Both source and target needed when alignments are provided"
        self.constraints = constraints
        self.append_bos = append_bos
        self.eos = eos if eos is not None else src_dict.eos()
        self.src_lang_id = src_lang_id
        self.tgt_lang_id = tgt_lang_id
        if num_buckets > 0:
            from fairseq.data import BucketPadLengthDataset

            self.src = BucketPadLengthDataset(
                self.src,
                sizes=self.src_sizes,
                num_buckets=num_buckets,
                pad_idx=self.src_dict.pad(),
                left_pad=self.left_pad_source,
            )
            self.src_sizes = self.src.sizes
            logger.info("bucketing source lengths: {}".format(list(self.src.buckets)))
            if self.tgt is not None:
                self.tgt = BucketPadLengthDataset(
                    self.tgt,
                    sizes=self.tgt_sizes,
                    num_buckets=num_buckets,
                    pad_idx=self.tgt_dict.pad(),
                    left_pad=self.left_pad_target,
                )
                self.tgt_sizes = self.tgt.sizes
                logger.info(
                    "bucketing target lengths: {}".format(list(self.tgt.buckets))
                )

            # determine bucket sizes using self.num_tokens, which will return
            # the padded lengths (thanks to BucketPadLengthDataset)
            num_tokens = np.vectorize(self.num_tokens, otypes=[np.long])
            self.bucketed_num_tokens = num_tokens(np.arange(len(self.src)))
            self.buckets = [
                (None, num_tokens) for num_tokens in np.unique(self.bucketed_num_tokens)
            ]
        else:
            self.buckets = None
        self.pad_to_multiple = pad_to_multiple
        self.cal_wer_dur = cal_wer_dur
        self.src_with_werdur = src_with_werdur
        self.nbest_infer = nbest_infer
        self.bos_prepended_outside = bos_prepended_outside
        if self.cal_wer_dur:
            assert not self.src_with_werdur
        if self.src_with_werdur:
            assert not self.cal_wer_dur
        self.phone_size = None
        self.vocab_phone_dict = None
        if homophone_dict_path:
            #print("homophone_dict_path:", homophone_dict_path)
            self.homophone_dict = {}
            self.top_char_set = []
            with open(homophone_dict_path, 'r', encoding='utf-8') as infile:
                for num, line in enumerate(infile.readlines()):
                    line = line.strip().split('\t')
                    if num < 3000:
                        self.top_char_set.append(self.src_dict.index(line[0]))
                    assert len(line) == 2
                    assert self.src_dict.index(line[0]) != self.src_dict.unk()
                    self.homophone_dict[self.src_dict.index(line[0])] = [self.src_dict.index(i) for i in line[1].split()]
                    for i in self.homophone_dict[self.src_dict.index(line[0])]:
                        assert i != self.src_dict.unk()

        else:
            self.homophone_dict = None
        self.mask_ratio = mask_ratio
        self.detector_mask_ratio = detector_mask_ratio
        self.error_distribution = error_distribution
        if self.detector_mask_ratio != 0.0:
            assert self.error_distribution[3] == 0.0
        if error_distribution:
            assert sum(error_distribution) == 1.0
            assert ft_error_distribution is None

        self.duptoken_error_distribution = duptoken_error_distribution
        if self.duptoken_error_distribution is not None:
            assert len(self.duptoken_error_distribution) == 2
            assert sum(self.duptoken_error_distribution) == 1.0

        if ft_error_distribution:
            assert error_distribution is None
            # assert duptoken_error_distribution is None
            self.ft_error_distribution = ft_error_distribution
            self.correct2dup = ft_error_distribution[0]
            self.wrong2dup = ft_error_distribution[1]
            print("Convert {} correct word to dup token and {} wrong word to CTC dup token".format(self.correct2dup,
                                                                                         self.wrong2dup))
        else:
            self.correct2dup = None
            self.wrong2dup = None
            self.ft_error_distribution = None
        self.force_mask_type = force_mask_type
        #print(self.force_mask_type)


    def get_batch_shapes(self):
        return self.buckets

    def random_mask(self, i, max_length):
        result = [i]
        prob = random.random()
        if prob < 0.25:
            if i - 1 >= 1:
                result.append(i-1)
        elif prob > 0.75:
            if i - 1 >= 1:
                result.append(i-1)
            if i - 1 >= 2:
                result.append(i-2)
        prob = random.random()
        if prob < 0.25:
            if i+1 < max_length - 1:
                result.append(i+1)
        elif prob > 0.75:
            if i+1 < max_length - 1:
                result.append(i+1)
            if i+1 < max_length - 2:
                result.append(i+2)
        return result

    def apply_mask(self, src_item_list, target_full_list, target_ctc_list, mask_type_list, token, mask_choose_id, dup_type=0, next_token=None, is_pretrain=True):
        token = int(token)
        if next_token is not None:
            assert dup_type == 2
            assert mask_choose_id == 4
            next_token = int(next_token)
        if mask_choose_id == 1:
            assert is_pretrain
            # same token
            src_item_list.append(token)
            target_full_list.append(token)
            mask_type_list.append(1)
            if len(src_item_list) == 1 or mask_type_list[-2] == 4 or target_full_list[-1] != target_full_list[-2]:
                target_ctc_list.append(token)
        elif mask_choose_id == 2:
            # <mask>
            assert is_pretrain
            src_item_list.append(self.src_dict.mask())
            target_full_list.append(token)
            mask_type_list.append(2)
            if len(src_item_list) == 1 or mask_type_list[-2] == 4 or target_full_list[-1] != target_full_list[-2]:
                target_ctc_list.append(token)
        elif mask_choose_id == 3:
            # homophone
            candidate_logit = [6, 5, 5, 4, 4, 3, 3, 3, 2, 2, 2, 1, 1, 1, 1]
            if token not in self.homophone_dict.keys():
                assert self.apply_mask_toall
                src_item_list.append(token)
                target_full_list.append(token)
                mask_type_list.append(1)
                if len(src_item_list) == 1 or mask_type_list[-2] == 4 or target_full_list[-1] != target_full_list[-2]:
                    target_ctc_list.append(token)
                # raise ValueError("Impossile condition!")
                # homophone = np.random.choice(homophone_dictionary["freq_token"])
            else:
                candidate = self.homophone_dict[token]
                prob_candidate = [i / sum(candidate_logit[:len(candidate)]) for i in candidate_logit[:len(candidate)]]
                homophone = np.random.choice(candidate, p=prob_candidate)
                src_item_list.append(homophone)
                target_full_list.append(token)
                mask_type_list.append(3)
                if len(src_item_list) == 1 or mask_type_list[-2] == 4 or target_full_list[-1] != target_full_list[-2]:
                    target_ctc_list.append(token)
        elif mask_choose_id == 4:
            if dup_type != 0:
                if random.random() < 0.5:
                    candidate_logit = [6, 5, 5, 4, 4, 3, 3, 3, 2, 2, 2, 1, 1, 1, 1]
                    if dup_type == 1 or random.random() < 0.5:
                        candidate = self.homophone_dict[token]
                    else:
                        candidate = self.homophone_dict[next_token]
                    prob_candidate = [i / sum(candidate_logit[:len(candidate)]) for i in
                                      candidate_logit[:len(candidate)]]
                    homophone = np.random.choice(candidate, p=prob_candidate)
                else:
                    if dup_type == 1 or random.random() < 0.5:
                        homophone = token
                    else:
                        homophone = next_token
                for _ in range(3):
                    src_item_list.append(homophone)
                    target_full_list.append(token)
                    mask_type_list.append(4)

                target_ctc_list.append(token)
                if dup_type == 2:
                    target_full_list[-1] = next_token
                    assert next_token is not None
                    target_ctc_list.append(next_token)
        elif mask_choose_id == 5:
            # <mask> to random token
            # assert is_pretrain
            src_item_list.append(int(np.random.choice(self.top_char_set)))
            target_full_list.append(token)
            mask_type_list.append(5)
            if len(src_item_list) == 1 or mask_type_list[-2] == 4 or target_full_list[-1] != target_full_list[-2]:
                target_ctc_list.append(token)
        elif mask_choose_id == 6:
            # <mask>
            assert is_pretrain
            src_item_list.append(self.src_dict.mask())
            target_full_list.append(self.src_dict.void())
            mask_type_list.append(6)
            if len(src_item_list) == 1 or mask_type_list[-2] == 4 or target_full_list[-1] != target_full_list[-2]:
                target_ctc_list.append(self.src_dict.void())
        else:
            raise ValueError("impossible mask_choose_id")
        return src_item_list, target_full_list, target_ctc_list, mask_type_list



    def build_example_for_mask(self, src_item, index, werdur_info=None, tgt_item=None, error_dis=None):
        if error_dis is None:
            error_dis = self.error_distribution
        target = src_item
        mask_type_list = []
        src_item_list = []
        target_full_list = []
        target_ctc_list = []

        if self.force_mask_type is not None:
            if self.force_mask_type == "detector_infer" or self.force_mask_type == "dec_infer":
                self.force_mask_type = [0 for _ in range(len(src_item))]
            assert len(src_item) == len(self.force_mask_type)
            for pos in range(len(src_item)):
                if self.force_mask_type[pos] in [0, 3]:
                    src_item_list.append(src_item[pos])
                    target_full_list.append(src_item[pos])
                    mask_type_list.append(self.force_mask_type[pos])
                    target_ctc_list.append(src_item[pos])
                elif self.force_mask_type[pos] == 4:
                    for _ in range(3):
                        src_item_list.append(src_item[pos])
                        target_full_list.append(src_item[pos])
                        mask_type_list.append(4)
                    target_ctc_list.append(src_item[pos])
                else:
                    raise ValueError("Not support mask type")
        else:
            # for pos in range(len(src_item)):
            pos = 0
            while pos < len(src_item):
                if pos == 0 or pos == len(src_item) - 1:
                    src_item_list.append(src_item[pos])
                    target_full_list.append(src_item[pos])
                    mask_type_list.append(0)
                    target_ctc_list.append(src_item[pos])
                    pos += 1
                    continue

                if int(src_item[pos]) not in self.homophone_dict.keys():
                    src_item_list.append(src_item[pos])
                    target_full_list.append(src_item[pos])
                    mask_type_list.append(0)
                    if pos == 0 or mask_type_list[-2] == 4 or target_full_list[-1] != target_full_list[-2]:
                        target_ctc_list.append(src_item[pos])
                    pos += 1

                else:
                    add_mask_var = random.random()
                    if add_mask_var <= self.mask_ratio:
                        addition_pos = 0
                        dup_type = 0  #0: not dup, 1 dup to itself, 2 dup to homo
                        mask_choose_var = random.random()

                        prob_thre = error_dis[0]
                        mask_choose_id = 0
                        while mask_choose_var > prob_thre:
                            mask_choose_id = mask_choose_id + 1
                            prob_thre = prob_thre + error_dis[mask_choose_id]

                        if mask_choose_id + 1 == 4:
                            if self.duptoken_error_distribution is not None:
                                if pos == len(src_item) - 1 or int(src_item[pos + 1]) not in self.homophone_dict.keys():
                                    addition_pos = 0
                                    dup_type = 1
                                elif pos == len(src_item) - 2 and (src_item[pos - 1] == src_item[pos] == src_item[pos + 1]):
                                    addition_pos = 0
                                    dup_type = 1
                                elif (pos < len(src_item) - 2) and (int(src_item[pos - 1] == src_item[pos]) + int(src_item[pos] == src_item[pos+1]) + int(src_item[pos+1] == src_item[pos+2])> 1):
                                    addition_pos = 0
                                    dup_type = 1
                                elif random.random() < self.duptoken_error_distribution[0]:
                                    addition_pos = 0
                                    dup_type = 1
                                else:
                                    addition_pos = 1
                                    dup_type = 2
                        if dup_type != 2:
                            src_item_list, target_full_list, target_ctc_list, mask_type_list = self.apply_mask(
                                src_item_list, target_full_list, target_ctc_list, mask_type_list, src_item[pos], mask_choose_id + 1,
                                dup_type=dup_type)
                        else:
                            src_item_list, target_full_list, target_ctc_list, mask_type_list = self.apply_mask(src_item_list,
                                              target_full_list, target_ctc_list, mask_type_list, src_item[pos], mask_choose_id + 1, dup_type=dup_type, next_token=src_item[pos+1])
                        if mask_choose_id != 5:
                            pos = pos + addition_pos + 1
                    else:
                        src_item_list.append(src_item[pos])
                        target_full_list.append(src_item[pos])
                        mask_type_list.append(0)
                        if pos == 0 or mask_type_list[-2] == 4 or target_full_list[-1] != target_full_list[-2]:
                            target_ctc_list.append(src_item[pos])
                        pos += 1

        mask_type = torch.LongTensor(mask_type_list)
        src_item = torch.LongTensor(src_item_list)
        target_full = torch.LongTensor(target_full_list)
        target_ctc = torch.LongTensor(target_ctc_list)

        example = {
            "id": index,
            "source": src_item,
            "target": target,
            "target_full": target_full,
            "wer_dur": None,
            "to_be_edited": None,
            "for_wer_gather": None,
            "source_phone": None,
            "mask_type": mask_type,
            "target_ctc": target_ctc
        }
        return example

    def build_example_for_detector(self, src_item, index, werdur_info=None, tgt_item=None, is_infer=True):
        if self.force_mask_type is not None:
            assert self.force_mask_type == "detector_infer" or self.force_mask_type == "dec_infer"
            example = self.build_example_for_mask(src_item, index, werdur_info, tgt_item, error_dis=[1.0, 0.0, 0.0, 0.0, 0.0])
            assert (example["mask_type"] > 1).long().sum() == 0
            return example
        else:
            detector_mask_var = random.random()
            if detector_mask_var > self.detector_mask_ratio:
                return self.build_example_for_mask(src_item, index, werdur_info, tgt_item)
            else:
                return self.build_example_for_mask(src_item, index, werdur_info, tgt_item, error_dis=[0.1, 0.0, 0.6, 0.0, 0.3])

    def build_example_for_finetune(self, src_item, index, tgt_item, to_be_edited, werdur):

        mask_type_list = []
        decoder_input_list = []
        target_full_list = []
        target_ctc_list = []
        # encoder_label_list = []

        pos = 0
        tgt_pos = 0
        while pos < len(src_item):
            if pos == 0 or pos == len(src_item) - 1:
                assert werdur[pos] == 1
                assert to_be_edited[pos] == 1
                assert src_item[pos] == tgt_item[tgt_pos]
                decoder_input_list.append(src_item[pos])
                target_full_list.append(tgt_item[tgt_pos])
                mask_type_list.append(0)
                target_ctc_list.append(tgt_item[tgt_pos])
                # encoder_label_list.append([int(tgt_item[tgt_pos]), self.src_dict.pad(), self.src_dict.pad()])
                pos += 1
                tgt_pos += 1
                continue
            else:
                add_mask_var = random.random()
                if int(to_be_edited[pos]) == 1:  #right token
                    assert int(werdur[pos]) == 1
                    assert src_item[pos] == tgt_item[tgt_pos], (pos, tgt_pos, src_item[pos], tgt_item[tgt_pos], src_item, tgt_item, werdur, to_be_edited)
                    if add_mask_var < self.correct2dup:
                        target_ctc_list.append(tgt_item[tgt_pos])
                        for _ in range(3):
                            decoder_input_list.append(src_item[pos])
                            target_full_list.append(tgt_item[tgt_pos])
                            mask_type_list.append(4)
                    else:
                        decoder_input_list.append(src_item[pos])
                        target_full_list.append(tgt_item[tgt_pos])
                        mask_type_list.append(0)
                        if tgt_pos == 0 or mask_type_list[-2] == 4 or target_full_list[-1] != target_full_list[-2]:
                            target_ctc_list.append(tgt_item[tgt_pos])
                    # encoder_label_list.append([int(tgt_item[tgt_pos]), self.src_dict.pad(), self.src_dict.pad()])
                    pos += 1
                    tgt_pos += 1
                elif int(werdur[pos]) == 0:
                    # encoder_label_list.append([self.src_dict.void(), self.src_dict.pad(), self.src_dict.pad()])
                    pos += 1
                elif int(werdur[pos]) == 1:
                    if src_item[pos] == tgt_item[tgt_pos]:
                        assert int(tgt_item[tgt_pos]) == 3, (src_item[pos], tgt_item[tgt_pos], pos, tgt_pos, src_item, tgt_item, werdur, to_be_edited)
                    #assert src_item[pos] != tgt_item[tgt_pos], (src_item[pos], tgt_item[tgt_pos], pos, tgt_pos, src_item, tgt_item, werdur, to_be_edited)
                    if add_mask_var < self.wrong2dup:
                        target_ctc_list.append(tgt_item[tgt_pos])
                        for j in range(3):
                            decoder_input_list.append(src_item[pos])
                            target_full_list.append(tgt_item[tgt_pos])
                            mask_type_list.append(4)
                    else:
                        decoder_input_list.append(src_item[pos])
                        target_full_list.append(tgt_item[tgt_pos])
                        mask_type_list.append(3)
                        if tgt_pos == 0 or mask_type_list[-2] == 4 or target_full_list[-1] != target_full_list[-2]:
                            target_ctc_list.append(tgt_item[tgt_pos])
                    # encoder_label_list.append([int(tgt_item[tgt_pos]), self.src_dict.pad(), self.src_dict.pad()])
                    pos += 1
                    tgt_pos += 1
                else:
                    assert 3 >= int(werdur[pos]) > 1, (werdur[pos], src_item, tgt_item, werdur, to_be_edited)
                    for j in range(int(werdur[pos])):
                        target_ctc_list.append(tgt_item[tgt_pos + j])
                    for j in range(3):
                        decoder_input_list.append(src_item[pos])
                        target_full_list.append(tgt_item[tgt_pos])
                        mask_type_list.append(4)
                    # if int(werdur[pos]) == -2:
                    #     encoder_label_list.append([int(tgt_item[tgt_pos]), int(tgt_item[tgt_pos + 1]), self.src_dict.pad()])
                    # else:
                    #     encoder_label_list.append(
                    #         [int(tgt_item[tgt_pos]), int(tgt_item[tgt_pos + 1]), int(tgt_item[tgt_pos + 2])])
                    tgt_pos += abs(werdur[pos])
                    pos += 1
        assert pos == len(src_item)
        assert tgt_pos == len(tgt_item)

        mask_type = torch.LongTensor(mask_type_list)
        decoder_input = torch.LongTensor(decoder_input_list)
        target_full = torch.LongTensor(target_full_list)
        target_ctc = torch.LongTensor(target_ctc_list)
        # encoder_label = torch.LongTensor(encoder_label_list)

        example = {
            "id": index,
            "source": decoder_input,
            "target": tgt_item,
            "target_full": target_full,
            "wer_dur": None,
            "to_be_edited": None,
            "for_wer_gather": None,
            "source_phone": None,
            "mask_type": mask_type,
            "target_ctc": target_ctc,
        }
        return example

    def __getitem__(self, index):
        #if self.mask_ratio != 0.0:
        #    assert self.tgt is None
        tgt_item = self.tgt[index] if self.tgt is not None else None
        src_item = self.src[index]
        # Append EOS to end of tgt sentence if it does not have an EOS and remove
        # EOS from end of src sentence if it exists. This is useful when we use
        # use existing datasets for opposite directions i.e., when we want to
        # use tgt_dataset as src_dataset and vice versa
        if self.append_eos_to_target:
            eos = self.tgt_dict.eos() if self.tgt_dict else self.src_dict.eos()
            if self.tgt and self.tgt[index][-1] != eos:
                tgt_item = torch.cat([self.tgt[index], torch.LongTensor([eos])])

        if self.append_bos:
            bos = self.tgt_dict.bos() if self.tgt_dict else self.src_dict.bos()
            if self.tgt and self.tgt[index][0] != bos:
                tgt_item = torch.cat([torch.LongTensor([bos]), self.tgt[index]])

            bos = self.src_dict.bos()
            if self.src[index][0] != bos:
                src_item = torch.cat([torch.LongTensor([bos]), self.src[index]])

        if self.remove_eos_from_source:
            eos = self.src_dict.eos()
            if self.src[index][-1] == eos:
                src_item = self.src[index][:-1]

        if (self.mask_ratio != 0.0 or self.force_mask_type or self.detector_mask_ratio != 0.0):
            if self.src_with_werdur:
                # assert not
                src_item_length = int(len(src_item))
                if self.append_bos or self.bos_prepended_outside:  # origin 8, append_bos: 9
                    assert src_item_length % 2 == 1
                    werdur_info = src_item[(src_item_length + 1) // 2:].clone() - 32768
                    werdur_info = torch.cat([torch.LongTensor([1]), werdur_info], dim=-1)
                    src_item = src_item[:(src_item_length + 1) // 2]
                else:
                    assert src_item_length % 2 == 0
                    werdur_info = src_item[(src_item_length) // 2:].clone() - 32768
                    # werdur_info = torch.cat([torch.LongTensor([1]), werdur_info], dim=-1)
                    src_item = src_item[:(src_item_length) // 2]
            else:
                werdur_info = None
            if self.nbest_infer > 1:
                src_item_length = int(len(src_item))
                bos = self.tgt_dict.bos() if self.tgt_dict else self.src_dict.bos()
                eos = self.tgt_dict.eos() if self.tgt_dict else self.src_dict.eos()
                if self.append_bos or self.bos_prepended_outside:  # origin 8, append_bos: 9
                    src_item = src_item[1:-1]  # remove EOS
                    assert len(src_item) % self.nbest_infer == 0
                    src_item = torch.reshape(src_item,
                                             [self.nbest_infer, int(len(src_item) / self.nbest_infer)]).transpose(0, 1)
                    src_item = torch.cat([torch.LongTensor([[bos for iter_i in range(self.nbest_infer)]]), src_item,
                                          torch.LongTensor([[eos for iter_i in range(self.nbest_infer)]])], dim=0)


                else:
                    src_item = src_item[:-1]  # remove EOS

                    assert len(src_item) % self.nbest_infer == 0
                    src_item = torch.reshape(src_item,
                                             [int(len(src_item) / self.nbest_infer), self.nbest_infer]).transpose(0, 1)
                    src_item = torch.cat(
                        [src_item, torch.LongTensor([[eos for iter_i in range(self.nbest_infer)]])], dim=0)

                example = {
                    "id": index,
                    "source": src_item,
                    "target": None,
                    "target_full": None,
                    "wer_dur": None,
                    "to_be_edited": None,
                    "for_wer_gather": None,
                    "source_phone": None,
                    "mask_type": torch.zeros(src_item.shape[0]),
                    "target_ctc": None
                }
                return example
            elif self.force_mask_type == "detector_infer" or self.force_mask_type == "dec_infer":
                return self.build_example_for_detector(src_item, index, werdur_info, tgt_item, is_infer=True)
            elif self.detector_mask_ratio != 0.0:
                return self.build_example_for_detector(src_item, index, werdur_info, tgt_item)
            else:
                return self.build_example_for_mask(src_item, index, werdur_info, tgt_item)


        if self.src_with_werdur:
            # assert not 
            src_item_length = int(len(src_item))
            #print(src_item_length, src_item)
            if self.append_bos or self.bos_prepended_outside:  # origin 8, append_bos: 9
                assert src_item_length % 2 == 1
                werdur_info = src_item[(src_item_length+1)//2:].clone() - 32768
                werdur_info = torch.cat([torch.LongTensor([1]), werdur_info], dim=-1)
                src_item = src_item[:(src_item_length+1)//2]
            else:
                assert src_item_length % 2 == 0
                werdur_info = src_item[(src_item_length)//2:].clone() - 32768
                # werdur_info = torch.cat([torch.LongTensor([1]), werdur_info], dim=-1)
                src_item = src_item[:(src_item_length)//2]


            to_be_edited = werdur_info.clamp(0, 1)
            wer_dur = torch.abs(werdur_info)
            assert self.ft_error_distribution is not None
            return self.build_example_for_finetune(src_item, index, tgt_item, to_be_edited, wer_dur)
        else:
            example = {
                "id": index,
                "source": src_item,
                "target": tgt_item,
            }
        if self.align_dataset is not None:
            example["alignment"] = self.align_dataset[index]
        if self.constraints is not None:
            example["constraints"] = self.constraints[index]
        return example

    def __len__(self):
        return len(self.src)

    def collater(self, samples, pad_to_length=None):
        """Merge a list of samples to form a mini-batch.

        Args:
            samples (List[dict]): samples to collate
            pad_to_length (dict, optional): a dictionary of
                {'source': source_pad_to_length, 'target': target_pad_to_length}
                to indicate the max length to pad to in source and target respectively.

        Returns:
            dict: a mini-batch with the following keys:

                - `id` (LongTensor): example IDs in the original input order
                - `ntokens` (int): total number of tokens in the batch
                - `net_input` (dict): the input to the Model, containing keys:

                  - `src_tokens` (LongTensor): a padded 2D Tensor of tokens in
                    the source sentence of shape `(bsz, src_len)`. Padding will
                    appear on the left if *left_pad_source* is ``True``.
                  - `src_lengths` (LongTensor): 1D Tensor of the unpadded
                    lengths of each source sentence of shape `(bsz)`
                  - `prev_output_tokens` (LongTensor): a padded 2D Tensor of
                    tokens in the target sentence, shifted right by one
                    position for teacher forcing, of shape `(bsz, tgt_len)`.
                    This key will not be present if *input_feeding* is
                    ``False``.  Padding will appear on the left if
                    *left_pad_target* is ``True``.
                  - `src_lang_id` (LongTensor): a long Tensor which contains source
                    language IDs of each sample in the batch

                - `target` (LongTensor): a padded 2D Tensor of tokens in the
                  target sentence of shape `(bsz, tgt_len)`. Padding will appear
                  on the left if *left_pad_target* is ``True``.
                - `tgt_lang_id` (LongTensor): a long Tensor which contains target language
                   IDs of each sample in the batch
        """
        res = collate(
            samples,
            pad_idx=self.src_dict.pad(),
            eos_idx=self.eos,
            left_pad_source=self.left_pad_source,
            left_pad_target=self.left_pad_target,
            input_feeding=self.input_feeding,
            pad_to_length=pad_to_length,
            pad_to_multiple=self.pad_to_multiple,
        )
        if self.src_lang_id is not None or self.tgt_lang_id is not None:
            src_tokens = res["net_input"]["src_tokens"]
            bsz = src_tokens.size(0)
            if self.src_lang_id is not None:
                res["net_input"]["src_lang_id"] = (
                    torch.LongTensor([[self.src_lang_id]]).expand(bsz, 1).to(src_tokens)
                )
            if self.tgt_lang_id is not None:
                res["tgt_lang_id"] = (
                    torch.LongTensor([[self.tgt_lang_id]]).expand(bsz, 1).to(src_tokens)
                )
        return res

    def num_tokens(self, index):
        """Return the number of tokens in a sample. This value is used to
        enforce ``--max-tokens`` during batching."""
        if self.src_with_werdur:
            return max(
                self.src_sizes[index] // 2,
                self.tgt_sizes[index] if self.tgt_sizes is not None else 0,
            )
        else:
            return max(
                self.src_sizes[index],
                self.tgt_sizes[index] if self.tgt_sizes is not None else 0,
            )

    def size(self, index):
        """Return an example's size as a float or tuple. This value is used when
        filtering a dataset with ``--max-positions``."""
        return (
            self.src_sizes[index],
            self.tgt_sizes[index] if self.tgt_sizes is not None else 0,
        )

    def ordered_indices(self):
        """Return an ordered list of indices. Batches will be constructed based
        on this order."""
        if self.shuffle:
            indices = np.random.permutation(len(self)).astype(np.int64)
        else:
            indices = np.arange(len(self), dtype=np.int64)
        if self.buckets is None:
            # sort by target length, then source length
            if self.tgt_sizes is not None:
                indices = indices[np.argsort(self.tgt_sizes[indices], kind="mergesort")]
            return indices[np.argsort(self.src_sizes[indices], kind="mergesort")]
        else:
            # sort by bucketed_num_tokens, which is:
            #   max(padded_src_len, padded_tgt_len)
            return indices[
                np.argsort(self.bucketed_num_tokens[indices], kind="mergesort")
            ]

    @property
    def supports_prefetch(self):
        return getattr(self.src, "supports_prefetch", False) and (
            getattr(self.tgt, "supports_prefetch", False) or self.tgt is None
        )

    def prefetch(self, indices):
        self.src.prefetch(indices)
        if self.tgt is not None:
            self.tgt.prefetch(indices)
        if self.align_dataset is not None:
            self.align_dataset.prefetch(indices)

    def filter_indices_by_size(self, indices, max_sizes):
        """Filter a list of sample indices. Remove those that are longer
            than specified in max_sizes.

        Args:
            indices (np.array): original array of sample indices
            max_sizes (int or list[int] or tuple[int]): max sample size,
                can be defined separately for src and tgt (then list or tuple)

        Returns:
            np.array: filtered sample array
            list: list of removed indices
        """
        return data_utils_sc.filter_paired_dataset_indices_by_size(
            self.src_sizes,
            self.tgt_sizes,
            indices,
            max_sizes,
        )
