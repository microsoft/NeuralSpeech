# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import sys
import torch
import argparse
import re
from fairseq.models.transformer import TransformerModel
import os
import os.path
import time

import json
import numpy as np

from fairseq import utils
utils.import_user_module(argparse.Namespace(user_dir='./softcorrect'))
from softcorrect.softcorrect_model import SoftcorrectDetectorModel

from fairseq.tokenizer import tokenize_line
#os.environ["CUDA_VISIBLE_DEVICES"] = '0'

def remove_ch_spaces(input_str):
    return re.sub(r"(?<=[\u4e00-\u9fff])(\s+)(?=[\u4e00-\u9fff])", "", input_str.strip())

def word_to_char(text):
    text = re.sub(r'([\u4e00-\u9fff])', r'\1 ',text)
    text = re.sub('\s+',' ',text)
    return text.strip()

def tn_bpe(text):
    text = re.sub("(?<=[\u4e00-\u9fff])(\s\▁\s*)(?=[\u4e00-\u9fff])"," ",text)
    if text[0] == '▁':
        text = " ".join(text.strip('▁').split())
    assert len(text) == 2 * len(text.split()) - 1
    assert '▁' not in text
    return text


model_name_or_path = sys.argv[2]

eval_data = sys.argv[3]

try:
    test_epoch = int(sys.argv[4])
    checkpoint_file = "checkpoint{}.pt".format(test_epoch)
except:
    test_epoch = 'best'
    checkpoint_file = "checkpoint_best.pt"

print("test {}/{}".format(model_name_or_path, checkpoint_file))
data_name_or_path = "data/detector_dict"
bpe = "sentencepiece"
sentencepiece_model = "./sentence.bpe.model" 

res_dir = os.path.join(model_name_or_path, ( ("results_aishell" )).replace('results', 'results_detector_' + str(test_epoch)))
tmp_dir = os.path.join(model_name_or_path, ( ("tmp_aishell" )).replace('tmp', 'tmp_detector_' + str(test_epoch)))
os.makedirs(res_dir, exist_ok=True)
os.makedirs(tmp_dir, exist_ok=True)


try:
    infile_list = sys.argv[1].split(',')
except:
    raise ValueError()

print("infile_list:", infile_list)

transf_gec = SoftcorrectDetectorModel.from_pretrained(model_name_or_path, checkpoint_file=checkpoint_file, data_name_or_path=data_name_or_path, bpe=bpe, 
                                                        sentencepiece_model=sentencepiece_model, arch="softcorrect_detector", task="softcorrect_task", pad_first_dictionary=True)
transf_gec.eval()
transf_gec.cuda()

for infile in infile_list:
    input_tsv = os.path.join(eval_data, infile, "aligned_nbest_token_raw.data.json")
    all_time = []
    eval_origin_dict = json.load(open(input_tsv, 'r', encoding='utf-8'))
    translate_input_dict = {}
    for k, v in eval_origin_dict["utts"].items():
        if ' # ' in v["output"][0]["rec_text"]:
            translate_input_dict[k] = (" ".join(v["output"][0]["rec_token"].replace(' ||| ', ' ').strip().split()), v["output"][0]["token"], None)
        else:
            raise ValueError("Incorrect format")

    translated_output_dict = {}
    for k, v in translate_input_dict.items():
        text = v[0]
        gt = v[1]
        rec_token_score = v[2]
        
        start_time = time.time()
        time_ok = False
        try:
            text_length = len(text.split())
            assert text_length % 4 == 0
            force_mask_type = "dec_infer"
            text_bin = transf_gec.binarize(text)
            batched_hypos = transf_gec.generate(text_bin, iter_decode_max_iter=0, force_mask_type=force_mask_type, nbest_infer=4)
            if isinstance(batched_hypos, tuple):
                batched_hypos, exm_time = batched_hypos
            else:
                exm_time = 1000000
            detector_probs = [list(map( lambda x: np.exp(float(x)), i)) for i in batched_hypos[0][0]['tokens'][0]][1:-1]
            assert len(detector_probs) == text_length / 4
            all_time.append(exm_time)
            time_ok = True
            #    translated = translated[0]
            translated_output_dict[k] = (text, gt, detector_probs)
            #translated_list.append(translated)
        except Exception as e:
            print(input_tsv + "\t" + text + "\n")
            #fout_ex.write(input_tsv + "\t" + text + "\n")
            translated_list.append(text)
            raise e
            continue

        end_time = time.time()
        if not time_ok:
            all_time.append(end_time - start_time)

        eval_origin_dict["utts"][k]["output"][0]["detector_prob"] = detector_probs  #" ".join(map(str, detector_probs))
    os.makedirs(os.path.join(res_dir, input_tsv.split('/')[-2]), exist_ok=True)
    if all_time:
        with open(os.path.join(res_dir, input_tsv.split('/')[-2], input_tsv.split('/')[-2] + "_time.txt"), 'w') as outfile:
            outfile.write("{}\t{}\t{}\n".format(len(all_time), sum(all_time), sum(all_time)/len(all_time)))
    json.dump(eval_origin_dict, open(os.path.join(res_dir, input_tsv.split('/')[-2], 'data.json'), 'w', encoding='utf-8'), indent=4, sort_keys=True, ensure_ascii=False)
