# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging

import numpy as np
import torch
from fairseq.data import FairseqDataset, data_utils
import random                                           
import math

logger = logging.getLogger(__name__)

def collate_2d_tokens(
    values,
    pad_idx,
    eos_idx=None,
    left_pad=False,
    move_eos_to_beginning=False,
    pad_to_length=None,
    pad_to_multiple=1,
):
    """Convert a list of 1d tensors into a padded 2d tensor."""
    hidden_size = values[0].size(1)
    size = max(v.size(0) for v in values)
    size = size if pad_to_length is None else max(size, pad_to_length)
    if pad_to_multiple != 1 and size % pad_to_multiple != 0:
        size = int(((size - 0.1) // pad_to_multiple + 1) * pad_to_multiple)
    res = values[0].new(len(values), size, hidden_size).fill_(pad_idx)

    def copy_tensor(src, dst):
        assert dst.numel() == src.numel()
        if move_eos_to_beginning:
            if eos_idx is None:
                # if no eos_idx is specified, then use the last token in src
                dst[0] = src[-1]
            else:
                dst[0] = eos_idx
            dst[1:] = src[:-1]
        else:
            dst.copy_(src)

    for i, v in enumerate(values):
        copy_tensor(v, res[i][size - len(v) :] if left_pad else res[i][: len(v)])
    return res

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

    def merge(key, left_pad, move_eos_to_beginning=False, pad_to_length=None):
        if len(samples[0][key].shape) == 1:
            return data_utils.collate_tokens(
                [s[key] for s in samples],
                pad_idx,
                eos_idx,
                left_pad,
                move_eos_to_beginning,
                pad_to_length=pad_to_length,
                pad_to_multiple=pad_to_multiple,
            )
        elif len(samples[0][key].shape) == 2:
            return collate_2d_tokens(
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

    if samples[0].get("target", None) is not None:
        target = merge(
            "target",
            left_pad=left_pad_target,
            pad_to_length=pad_to_length["target"]
            if pad_to_length is not None
            else None,
        )
        target = target.index_select(0, sort_order)
        if samples[0].get("wer_dur", None) is not None:
            for_wer_gather = merge(
                "for_wer_gather",
                left_pad=left_pad_target,
                pad_to_length=pad_to_length["target"]
                if pad_to_length is not None
                else None,
            )
            for_wer_gather = for_wer_gather.index_select(0, sort_order)

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
    if samples[0].get("wer_dur", None) is not None:
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
        src_with_nbest_werdur=0,
        bos_prepended_outside=False,
        merge_nbest_werdur='',
        break_alignment=False,
        copy_beam1=False,
        to_be_edited_mask = "",
        nbest_infer=0,
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
        self.src_with_nbest_werdur = src_with_nbest_werdur
        self.nbest_infer = nbest_infer
        self.merge_nbest_werdur = merge_nbest_werdur
        self.break_alignment = break_alignment
        self.copy_beam1 = copy_beam1
        if self.copy_beam1:
            assert self.src_with_nbest_werdur or nbest_infer
        if nbest_infer:
            assert not self.src_with_nbest_werdur
            assert not self.src_with_werdur
            assert not self.cal_wer_dur
        self.bos_prepended_outside = bos_prepended_outside
        self.to_be_edited_mask = to_be_edited_mask
        if self.cal_wer_dur:
            assert not self.src_with_werdur
            assert not self.src_with_nbest_werdur
            assert not nbest_infer
        if self.src_with_werdur:
            assert not self.cal_wer_dur
            assert not self.src_with_nbest_werdur
            assert not nbest_infer
        if self.src_with_nbest_werdur:
            assert not self.src_with_werdur
            assert not self.cal_wer_dur
            assert not nbest_infer
        if self.merge_nbest_werdur:
            assert not self.src_with_werdur
            assert not self.cal_wer_dur


    def get_batch_shapes(self):
        return self.buckets

    def calculate_wer_dur(self, hypo_list, ref_list):
        len_hyp = len(hypo_list)
        len_ref = len(ref_list)
        cost_matrix = np.zeros((len_hyp + 1, len_ref + 1), dtype=np.int16)

        # 0-equal；2-insertion；3-deletion；1-substitution
        ops_matrix = np.zeros((len_hyp + 1, len_ref + 1), dtype=np.int8)

        for i in range(len_hyp + 1):
            cost_matrix[i][0] = i
        for j in range(len_ref + 1):
            cost_matrix[0][j] = j

        id_ind = 0
        for i in range(1, len_hyp + 1):
            for j in range(1, len_ref + 1):
                ideal_index = i * len_ref / len_hyp
                if hypo_list[i-1] == ref_list[j-1]:
                    cost_matrix[i][j] = cost_matrix[i-1][j-1]
                else:
                    substitution = cost_matrix[i-1][j-1] + 1
                    insertion = cost_matrix[i-1][j] + 1
                    deletion = cost_matrix[i][j-1] + 1

                    compare_val = [substitution, insertion, deletion]   # 优先级

                    if (substitution > insertion) and (insertion == deletion) :
                        min_val = insertion
                        if ideal_index >= j:
                            operation_idx = 2
                        else:
                            operation_idx = 3
                    else:
                        min_val = min(compare_val)
                        operation_idx = compare_val.index(min_val) + 1
                    cost_matrix[i][j] = min_val
                    ops_matrix[i][j] = operation_idx

        i = len_hyp
        j = len_ref
        # nb_map = {"N": len_ref, "C": 0, "W": 0, "I": 0, "D": 0, "S": 0}
        char_map = []
        current_chars = []
        res_chars = []
        while i >= 0 or j >= 0:
            i_idx = max(0, i)
            j_idx = max(0, j)

            if ops_matrix[i_idx][j_idx] == 0:     # correct
                if i-1 >= 0 and j-1 >= 0:
                    # match_idx.append((j-1, i-1))
                    # nb_map['C'] += 1
                    current_chars.append(ref_list[j-1])
                    char_map.append([hypo_list[i-1], current_chars])
                    current_chars = []

                i -= 1
                j -= 1

            # elif ops_matrix[i_idx][j_idx] == 1:   # insert
            elif ops_matrix[i_idx][j_idx] == 2:   # insert
                char_map.append([hypo_list[i-1], current_chars])
                current_chars = []
                i -= 1
                # nb_map['I'] += 1
            # elif ops_matrix[i_idx][j_idx] == 2:   # delete
            elif ops_matrix[i_idx][j_idx] == 3:   # delete
                current_chars.append(ref_list[j-1])
                j -= 1
                # nb_map['D'] += 1
            # elif ops_matrix[i_idx][j_idx] == 3:   # substitute
            elif ops_matrix[i_idx][j_idx] == 1:   # substitute
                current_chars.append(ref_list[j-1])
                char_map.append([hypo_list[i-1], current_chars])
                current_chars = []
                i -= 1
                j -= 1
                # nb_map['S'] += 1
            else:
                raise ValueError("Impossible condition!")

            if i < 0 and j >= 0:
                # nb_map['D'] += 1
                res_chars.append(ref_list[j])
            elif j < 0 and i >= 0:
                char_map.append([hypo_list[i], current_chars])
                current_chars = []
                # nb_map['I'] += 1
            # else:
            #     raise ValueError("Impossible condition!")

        if res_chars:
            char_map[-1][-1].extend(res_chars)


        char_map.reverse()
        for i in range(len(char_map)):
            char_map[i][-1].reverse()

        # match_idx.reverse()
        # wrong_cnt = cost_matrix[len_hyp][len_ref]
        # nb_map["W"] = wrong_cnt

        # print("ref: %s" % " ".join(ref_list))
        # print("hyp: %s" % " ".join(hypo_list))
        # print(nb_map)
        # print("match_idx: %s" % str(match_idx))

        
        result_map = [len(i[1]) for i in char_map]
        to_be_modify = [int( (len(i[1]) == 1 and i[1][0] == i[0]) ) for i in char_map]
        # to_be_modify = []
        # for i in char_map:
        #     if (len(char_map[i][1]) = 1 and char_map[i][1][0] == char_map[i][0])
        #         to_be_modify.append(0)
        #     else:
        #         for j in range(len(char_map[i][1])):
        #             to_be_modify.append(1)
        #if len(to_be_modify) >= 180:
        #    print(char_map)
        #    print(to_be_modify)


        assert sum(result_map) == len_ref
        assert len(result_map) == len_hyp

        for_wer_gather = []
        for i in range(len(result_map)):    
            for j in range(result_map[i]):
                for_wer_gather.append(i)
        
        # return wrong_cnt, match_idx, nb_map, char_map
        return result_map, to_be_modify, for_wer_gather





    def break_beam_alignment(self, src_item, werdur_info):
        #print(werdur_info)
        # werdur_info_mean = torch.abs(werdur_info).float().mean(-1)
        # werdur_info_label = ((werdur_info.float().mean(-1) == 1.0).float() * 2 - 1).long()
        new_werdur_info = [[] for _ in range(self.src_with_nbest_werdur)]
        new_src_item = [[] for _ in range(self.src_with_nbest_werdur)]
        void_token = self.tgt_dict.indices['<void>'] if self.tgt_dict else self.src_dict.indices['<void>']

        max_length = src_item.shape[0]

        for i in range(max_length):
            for j in range(self.src_with_nbest_werdur):
                if src_item[i][j] != void_token:
                    new_src_item[j].append(src_item[i][j])
                    new_werdur_info[j].append(werdur_info[i][j])
                else:
                    assert werdur_info[i][j] == 0

        final_length = max(len(i) for i in new_src_item)

        for i in range(self.src_with_nbest_werdur):
            for j in range(final_length - len(new_src_item[i])):
                new_src_item[i].append(void_token)
                new_werdur_info[i].append(0)

        return torch.LongTensor(new_src_item).transpose(0, 1), torch.LongTensor(new_werdur_info).transpose(0, 1)

    def break_beam_alignment_infer(self, src_item):
        #print(werdur_info)
        # werdur_info_mean = torch.abs(werdur_info).float().mean(-1)
        # werdur_info_label = ((werdur_info.float().mean(-1) == 1.0).float() * 2 - 1).long()
        new_src_item = [[] for _ in range(self.nbest_infer)]
        void_token = self.tgt_dict.indices['<void>'] if self.tgt_dict else self.src_dict.indices['<void>']

        max_length = src_item.shape[0]

        for i in range(max_length):
            for j in range(self.nbest_infer):
                if src_item[i][j] != void_token:
                    new_src_item[j].append(src_item[i][j])

        final_length = max(len(i) for i in new_src_item)

        for i in range(self.nbest_infer):
            for j in range(final_length - len(new_src_item[i])):
                new_src_item[i].append(void_token)

        return torch.LongTensor(new_src_item).transpose(0, 1)

    def __getitem__(self, index):
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


        if self.src_with_werdur:
            assert not self.src_with_nbest_werdur
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
            for_wer_gather_list = []
            to_be_edited_list = [1 for _ in range(len(wer_dur))]
            for i in range(len(wer_dur)):
                if self.to_be_edited_mask == 'v1':
                    if int(to_be_edited[i]) == 0:
                        to_be_masked = self.random_mask(i, len(wer_dur))
                        for j in to_be_masked:
                            to_be_edited_list[j] = 0
                elif self.to_be_edited_mask == 'v2':
                    if int(to_be_edited[i]) == 0:
                        if int(wer_dur[i]) == 1:
                            to_be_masked = self.random_mask(i, len(wer_dur))
                            for j in to_be_masked:
                                to_be_edited_list[j] = 0
                        else:
                            to_be_edited_list[i] = 0
                    
                for j in range(abs(int(wer_dur[i]))):
                    for_wer_gather_list.append(i)
            assert to_be_edited_list[0] == 1
            assert to_be_edited_list[-1] == 1
            for_wer_gather = torch.LongTensor(for_wer_gather_list)
            if self.to_be_edited_mask == 'v1' or self.to_be_edited_mask == 'v2':
                to_be_edited = torch.LongTensor(to_be_edited_list)
            try:
                assert len(wer_dur) == len(src_item)
                assert len(tgt_item) == len(for_wer_gather)
            except:
                print("src string:")
                print(self.src_dict.string(src_item))
                print("tgt string:")
                print(self.tgt_dict.string(tgt_item))
                print(src_item, tgt_item); print(wer_dur, to_be_edited, for_wer_gather)
                raise ValueError()
            example = {
                "id": index,
                "source": src_item,
                "target": tgt_item,
                "wer_dur": wer_dur,
                "to_be_edited": to_be_edited,
                "for_wer_gather": for_wer_gather,
            }
        elif self.src_with_nbest_werdur:
            # assert not
            src_item_length = int(len(src_item))
            bos = self.tgt_dict.bos() if self.tgt_dict else self.src_dict.bos()
            eos = self.tgt_dict.eos() if self.tgt_dict else self.src_dict.eos()
            # print(src_item_length, src_item)
            if self.append_bos or self.bos_prepended_outside:  # origin 8, append_bos: 9
                assert (src_item_length - self.src_with_nbest_werdur) % 2 == 1
                #print(src_item)
                werdur_info = src_item[(src_item_length - self.src_with_nbest_werdur + 1) // 2:][:-1].clone() - 32768  #remove EOS
                # werdur_info = torch.cat([torch.LongTensor([1]), werdur_info], dim=-1)
                src_item = src_item[1:(src_item_length - self.src_with_nbest_werdur + 1) // 2][:-1]  #remove EOS
                #print(src_item, werdur_info)
                assert len(werdur_info) % self.src_with_nbest_werdur == 0
                assert len(src_item) % self.src_with_nbest_werdur == 0
                assert len(werdur_info) / self.src_with_nbest_werdur - len(src_item) / self.src_with_nbest_werdur == 1.0

                werdur_info = torch.reshape(werdur_info, [self.src_with_nbest_werdur, int(len(werdur_info) / self.src_with_nbest_werdur)]).transpose(0,1)
                src_item = torch.reshape(src_item, [self.src_with_nbest_werdur, int(len(src_item) / self.src_with_nbest_werdur)]).transpose(0,1)

                if self.copy_beam1:
                    werdur_info = werdur_info[:, 0][:, None].repeat([1, self.src_with_nbest_werdur])
                    src_item = src_item[:, 0][:, None].repeat([1, self.src_with_nbest_werdur])

                closest_label = werdur_info[-1, :].clone()
                werdur_info = werdur_info[:-1, :]

                assert src_item.shape == werdur_info.shape

                if self.merge_nbest_werdur == '':
                    werdur_info = torch.cat([torch.LongTensor([[1 for iter_i in range(self.src_with_nbest_werdur)]]), werdur_info, torch.LongTensor([[1 for iter_i in range(self.src_with_nbest_werdur)]])], dim=0)
                    src_item = torch.cat([torch.LongTensor([[bos for iter_i in range(self.src_with_nbest_werdur)]]), src_item, torch.LongTensor([[eos for iter_i in range(self.src_with_nbest_werdur)]])], dim=0)
                else:
                    raise ValueError("Bad merge_nbest_werdur!" + self.merge_nbest_werdur)
                    werdur_info = torch.cat(
                        [torch.LongTensor([1]), werdur_info,
                         torch.LongTensor([1])], dim=0)
                    src_item = torch.cat(
                        [torch.LongTensor([[bos for iter_i in range(self.src_with_nbest_werdur)]]), src_item,
                         torch.LongTensor([[eos for iter_i in range(self.src_with_nbest_werdur)]])], dim=0)

            else:
                raise ValueError("many logit not update(merge_nbest_werdur)")
                assert (src_item_length - self.src_with_nbest_werdur) % 2 == 0
                werdur_info = src_item[(src_item_length - self.src_with_nbest_werdur) // 2:][:-1].clone() - 32768  #remove EOS
                # werdur_info = torch.cat([torch.LongTensor([1]), werdur_info], dim=-1)
                src_item = src_item[:(src_item_length - self.src_with_nbest_werdur) // 2][:-1]  #remove EOS

                assert len(werdur_info) % self.src_with_nbest_werdur == 0
                assert len(src_item) % self.src_with_nbest_werdur == 0
                assert len(werdur_info) / self.src_with_nbest_werdur - len(src_item) / self.src_with_nbest_werdur == 1.0

                werdur_info = torch.reshape(werdur_info,
                                            [len(werdur_info) / self.src_with_nbest_werdur, self.src_with_nbest_werdur]).transpose(0,1)
                src_item = torch.reshape(src_item,
                                         [len(src_item) / self.src_with_nbest_werdur, self.src_with_nbest_werdur]).transpose(0,1)

                closest_label = werdur_info[-1, :].clone()
                werdur_info = werdur_info[:-1, :]

                assert closest_label.shape == werdur_info.shape

                werdur_info = torch.cat(
                    [werdur_info, torch.LongTensor([[1 for iter_i in range(self.src_with_nbest_werdur)]])], dim=0)
                src_item = torch.cat(
                    [src_item, torch.LongTensor([[eos for iter_i in range(self.src_with_nbest_werdur)]])], dim=0)

            if self.break_alignment:
                #print(src_item)
                #print(werdur_info)
                src_item, werdur_info = self.break_beam_alignment(src_item, werdur_info)
                #print(src_item)
                #print(werdur_info)

            to_be_edited = werdur_info.clamp(0, 1)
            wer_dur = torch.abs(werdur_info)
            if not self.merge_nbest_werdur:
                for_wer_gather_list = []
                # to_be_edited_list = [1 for _ in range(len(wer_dur))]
                for k in range(self.src_with_nbest_werdur):
                    add_to_wer_gather_list = []
                    for i in range(len(wer_dur)):
                        # if self.to_be_edited_mask == 'v1':
                        #     if int(to_be_edited[i]) == 0:
                        #         to_be_masked = self.random_mask(i, len(wer_dur))
                        #         for j in to_be_masked:
                        #             to_be_edited_list[j] = 0
                        # elif self.to_be_edited_mask == 'v2':
                        #     if int(to_be_edited[i]) == 0:
                        #         if int(wer_dur[i]) == 1:
                        #             to_be_masked = self.random_mask(i, len(wer_dur))
                        #             for j in to_be_masked:
                        #                 to_be_edited_list[j] = 0
                        #         else:
                        #             to_be_edited_list[i] = 0
                        for j in range(abs(int(wer_dur[i][k]))):
                            add_to_wer_gather_list.append(i)
                    for_wer_gather_list.append(add_to_wer_gather_list)
                # assert to_be_edited_list[0] == 1
                # assert to_be_edited_list[-1] == 1
                '''
                print(wer_dur)
                print(to_be_edited)
                print(src_item)
                print(src_item.shape, tgt_item.shape)
                print(for_wer_gather_list)
                '''
                for_wer_gather = torch.LongTensor(for_wer_gather_list).transpose(0, 1)
            else:
                for_wer_gather_list = []
                for i in range(len(wer_dur)):
                    for j in range(abs(int(wer_dur[i]))):
                        for_wer_gather_list.append(i)
                for_wer_gather = torch.LongTensor(for_wer_gather_list)
            # if self.to_be_edited_mask == 'v1' or self.to_be_edited_mask == 'v2':
            #     to_be_edited = torch.LongTensor(to_be_edited_list)
            try:
                #if self.src_with_nbest_werdur:
                #    assert 2 == 3
                assert len(wer_dur) == len(src_item)
                assert len(tgt_item) == len(for_wer_gather)
            except:
                print("src string:")
                print(self.src_dict.string(src_item))
                print("tgt string:")
                print(self.tgt_dict.string(tgt_item))
                print(src_item, tgt_item)
                if self.src_with_nbest_werdur:
                    print("wer_dur:", wer_dur)
                    print("to_be_edited", to_be_edited)
                    print("for_wer_gather", for_wer_gather)
                else:
                    print(wer_dur, to_be_edited, for_wer_gather)
                raise ValueError()
            example = {
                "id": index,
                "source": src_item,
                "target": tgt_item,
                "wer_dur": wer_dur,
                "to_be_edited": to_be_edited,
                "for_wer_gather": for_wer_gather,
                "closest_label": closest_label,
            }
        elif self.nbest_infer:
            # assert not
            src_item_length = int(len(src_item))
            bos = self.tgt_dict.bos() if self.tgt_dict else self.src_dict.bos()
            eos = self.tgt_dict.eos() if self.tgt_dict else self.src_dict.eos()
            # print(src_item_length, src_item)
            if self.append_bos or self.bos_prepended_outside:  # origin 8, append_bos: 9
                # assert (src_item_length - self.src_with_nbest_werdur) % 2 == 1
                # werdur_info = src_item[(src_item_length - self.src_with_nbest_werdur + 1) // 2:][:-1].clone() - 32768  #remove EOS
                # werdur_info = torch.cat([torch.LongTensor([1]), werdur_info], dim=-1)
                src_item = src_item[1:-1] #remove EOS
                #print(src_item, werdur_info)
                # assert len(werdur_info) % self.src_with_nbest_werdur == 0
                assert len(src_item) % self.nbest_infer == 0
                # assert len(werdur_info) / self.src_with_nbest_werdur - len(src_item) / self.src_with_nbest_werdur == 1.0

                # werdur_info = torch.reshape(werdur_info, [self.src_with_nbest_werdur, int(len(werdur_info) / self.src_with_nbest_werdur)]).transpose(0,1)
                src_item = torch.reshape(src_item, [self.nbest_infer, int(len(src_item) / self.nbest_infer)]).transpose(0,1)
                if self.copy_beam1:
                    src_item = src_item[:, 0][:, None].repeat([1, self.nbest_infer])

                # closest_label = werdur_info[-1, :].clone()
                # werdur_info = werdur_info[:-1, :]

                # assert src_item.shape == werdur_info.shape

                # werdur_info = torch.cat([torch.LongTensor([[1 for iter_i in range(self.src_with_nbest_werdur)]]), werdur_info, torch.LongTensor([[1 for iter_i in range(self.src_with_nbest_werdur)]])], dim=0)
                src_item = torch.cat([torch.LongTensor([[bos for iter_i in range(self.nbest_infer)]]), src_item, torch.LongTensor([[eos for iter_i in range(self.nbest_infer)]])], dim=0)


            else:
                # assert (src_item_length - self.src_with_nbest_werdur) % 2 == 0
                # werdur_info = src_item[(src_item_length - self.src_with_nbest_werdur) // 2:][:-1].clone() - 32768  #remove EOS
                # werdur_info = torch.cat([torch.LongTensor([1]), werdur_info], dim=-1)
                src_item = src_item[:-1]  #remove EOS

                # assert len(werdur_info) % self.src_with_nbest_werdur == 0
                assert len(src_item) % self.nbest_infer == 0
                # assert len(werdur_info) / self.src_with_nbest_werdur - len(src_item) / self.src_with_nbest_werdur == 1.0

                # werdur_info = torch.reshape(werdur_info,
                #                             [len(werdur_info) / self.src_with_nbest_werdur, self.src_with_nbest_werdur])
                src_item = torch.reshape(src_item,
                                         [int(len(src_item) / self.nbest_infer), self.nbest_infer]).transpose(0,1)
                if self.copy_beam1:
                    src_item = src_item[:, 0][:, None].repeat([1, self.nbest_infer])

                # closest_label = werdur_info[-1, :].clone()
                # werdur_info = werdur_info[:-1, :]

                # assert closest_label.shape == werdur_info.shape
                #
                # werdur_info = torch.cat(
                #     [werdur_info, torch.LongTensor([[1 for iter_i in range(self.src_with_nbest_werdur)]])], dim=0)
                src_item = torch.cat(
                    [src_item, torch.LongTensor([[eos for iter_i in range(self.nbest_infer)]])], dim=0)

            if self.break_alignment:
                src_item = self.break_beam_alignment_infer(src_item)

            # to_be_edited = werdur_info.clamp(0, 1)
            # wer_dur = torch.abs(werdur_info)
            # for_wer_gather_list = []
            # to_be_edited_list = [1 for _ in range(len(wer_dur))]
            # for k in range(self.src_with_nbest_werdur):
            #     add_to_wer_gather_list = []
            #     for i in range(len(wer_dur)):
            #         # if self.to_be_edited_mask == 'v1':
            #         #     if int(to_be_edited[i]) == 0:
            #         #         to_be_masked = self.random_mask(i, len(wer_dur))
            #         #         for j in to_be_masked:
            #         #             to_be_edited_list[j] = 0
            #         # elif self.to_be_edited_mask == 'v2':
            #         #     if int(to_be_edited[i]) == 0:
            #         #         if int(wer_dur[i]) == 1:
            #         #             to_be_masked = self.random_mask(i, len(wer_dur))
            #         #             for j in to_be_masked:
            #         #                 to_be_edited_list[j] = 0
            #         #         else:
            #         #             to_be_edited_list[i] = 0
            #         for j in range(abs(int(wer_dur[i][k]))):
            #             add_to_wer_gather_list.append(i)
            #     for_wer_gather_list.append(add_to_wer_gather_list)
            # assert to_be_edited_list[0] == 1
            # assert to_be_edited_list[-1] == 1
            '''
            print(wer_dur)
            print(to_be_edited)
            print(src_item)
            print(src_item.shape, tgt_item.shape)
            print(for_wer_gather_list)
            '''
            # for_wer_gather = torch.LongTensor(for_wer_gather_list).transpose(0, 1)
            # if self.to_be_edited_mask == 'v1' or self.to_be_edited_mask == 'v2':
            #     to_be_edited = torch.LongTensor(to_be_edited_list)
            # try:
            #     #if self.src_with_nbest_werdur:
            #     #    assert 2 == 3
            #     assert len(wer_dur) == len(src_item)
            #     assert len(tgt_item) == len(for_wer_gather)
            # except:
            #     print("src string:")
            #     print(self.src_dict.string(src_item))
            #     print("tgt string:")
            #     print(self.tgt_dict.string(tgt_item))
            #     print(src_item, tgt_item)
            #     if self.src_with_nbest_werdur:
            #         print("wer_dur:", wer_dur)
            #         print("to_be_edited", to_be_edited)
            #         print("for_wer_gather", for_wer_gather)
            #     else:
            #         print(wer_dur, to_be_edited, for_wer_gather)
            #     raise ValueError()
            example = {
                "id": index,
                "source": src_item,
                "target": tgt_item,
                # "wer_dur": wer_dur,
                # "to_be_edited": to_be_edited,
                # "for_wer_gather": for_wer_gather,
                # "closest_label": closest_label,
            }
        else:
            example = {
                "id": index,
                "source": src_item,
                "target": tgt_item,
            }
        if self.cal_wer_dur:
            # assert not self.src_with_werdur
            wer_dur, to_be_edited, for_wer_gather = self.calculate_wer_dur(src_item, tgt_item)
            example["wer_dur"] = torch.LongTensor(wer_dur)
            example["to_be_edited"] = torch.LongTensor(to_be_edited)
            example["for_wer_gather"] = torch.LongTensor(for_wer_gather)
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
        if self.src_with_nbest_werdur or self.src_with_werdur:
            if self.src_with_nbest_werdur:
                return max(
                    self.src_sizes[index] // 2 // self.src_with_nbest_werdur,
                    self.tgt_sizes[index] if self.tgt_sizes is not None else 0,
                )
            else:
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
        return data_utils.filter_paired_dataset_indices_by_size(
            self.src_sizes,
            self.tgt_sizes,
            indices,
            max_sizes,
        )
