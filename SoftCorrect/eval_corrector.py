# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import sys
import argparse
import torch
import re
import os
import os.path
import time

import json
import numpy as np

from fairseq import utils
utils.import_user_module(argparse.Namespace(user_dir='./softcorrect'))
from softcorrect.softcorrect_model import SoftcorrectCorrectorModel

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

detector_thre = sys.argv[2]

model_name_or_path = sys.argv[3]



eval_data = sys.argv[4]

try:
    test_epoch = int(sys.argv[5])
    checkpoint_file = "checkpoint{}.pt".format(test_epoch)
except:
    test_epoch = 'best'
    checkpoint_file = "checkpoint_best.pt"

try:
    duptoken_thre = float(sys.argv[6])
except:
    duptoken_thre = -0.43

try:
    phone_thre = float(sys.argv[7])
except:
    phone_thre = -0.09

if duptoken_thre < phone_thre:
    duptoken_first = True
else:
    duptoken_first = False

thre1 = min(phone_thre, duptoken_thre)
thre2 = max(phone_thre, duptoken_thre)

#checkpoint_file = "checkpoint_best.pt"
print("test {}/{}".format(model_name_or_path, checkpoint_file))
data_name_or_path = "data/aishell_corrector"
bpe = "sentencepiece"
sentencepiece_model = "./sentence.bpe.model" 

res_dir = os.path.join(model_name_or_path, ( ("results_aishell_b")).replace('results', 'results_corrector_' + str(test_epoch) + '_p' + str(duptoken_thre) + '_h' + str(phone_thre)))
tmp_dir = os.path.join(model_name_or_path, (  ("tmp_aishell_b")).replace('tmp', 'tmp_corrector_' + str(test_epoch) + '_p' + str(duptoken_thre) + '_h' + str(phone_thre)))
os.makedirs(res_dir, exist_ok=True)
os.makedirs(tmp_dir, exist_ok=True)
#fout_ex = open(os.path.join(tmp_dir, "exception.log"), "w")

try:
    infile_list = sys.argv[1].split(',')
except:
    raise ValueError()

print("infile_list:", infile_list)

transf_gec = SoftcorrectCorrectorModel.from_pretrained(model_name_or_path, checkpoint_file=checkpoint_file, data_name_or_path=data_name_or_path, bpe=bpe, 
                                                        sentencepiece_model=sentencepiece_model, arch="softcorrect_corrector", task="softcorrect_task", pad_first_dictionary=True)
transf_gec.eval()
transf_gec.cuda()

for infile in infile_list:
    input_tsv = os.path.join(eval_data, infile, detector_thre, "data.json")
    all_time = []
    eval_origin_dict = json.load(open(input_tsv, 'r', encoding='utf-8'))
    translate_input_dict = {}
    for k, v in eval_origin_dict["utts"].items():
        translate_input_dict[k] = (v["output"][0]["rec_text"].replace('<eos>', '').strip(), v["output"][0]["token"], v["output"][0]["score"])

    translated_output_dict = {}
    for k, v in translate_input_dict.items():
        text = v[0]
        gt = v[1]
        rec_token_score = v[2]
        rec_token_score = list(map(float, rec_token_score.split(';')))
        
        start_time = time.time()
        time_ok = False
        try:
            text = tn_bpe(transf_gec.apply_bpe(transf_gec.tokenize(word_to_char(text))))
            force_mask_type = []
            for score in rec_token_score:
                if duptoken_first:
                    if score > thre2:
                        force_mask_type.append(0)
                    elif score > thre1:
                        force_mask_type.append(3)
                    else:
                        force_mask_type.append(4)
                else:
                    if score > thre2:
                        force_mask_type.append(0)
                    elif score > thre1:
                        force_mask_type.append(4)
                    else:
                        force_mask_type.append(3)
            if text.split()[0] == '▁':
                assert len(rec_token_score) == len(text.split()) - 1
                force_mask_type = [0, 0] + force_mask_type + [0]
            else:
                assert len(rec_token_score) == len(text.split())
                force_mask_type = [0] + force_mask_type + [0]
            if sum(force_mask_type) == 0:
                translated = "".join(text.split())
                exm_time = 0.0
            else:
                text = transf_gec.binarize(text)
                batched_hypos = transf_gec.generate(text, iter_decode_max_iter=0, force_mask_type=force_mask_type, duptoken_error_distribution=[1.0, 0.0])
                if isinstance(batched_hypos, tuple):
                    batched_hypos, exm_time = batched_hypos
                else:
                    exm_time = 10000
                translated = [transf_gec.decode(hypos[0]['tokens']) for hypos in batched_hypos][0]
            all_time.append(exm_time)
            time_ok = True
            #    translated = translated[0]
            translated_output_dict[k] = (text, gt, translated)
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
        eval_origin_dict["utts"][k]["output"][0]["rec_text"] = " ".join("".join(translated.split()).replace('▁', ' ').strip().split())
        translated_char = [i for i in eval_origin_dict["utts"][k]["output"][0]["rec_text"]]
        eval_origin_dict["utts"][k]["output"][0]["rec_token"] = " ".join(translated_char)
    os.makedirs(os.path.join(res_dir, input_tsv.split('/')[-3], input_tsv.split('/')[-2]), exist_ok=True)
    if all_time:
        with open(os.path.join(res_dir, input_tsv.split('/')[-3], input_tsv.split('/')[-2], input_tsv.split('/')[-3] + "_time.txt"), 'w') as outfile:
            outfile.write("{}\t{}\t{}\n".format(len(all_time), sum(all_time), sum(all_time)/len(all_time)))
    json.dump(eval_origin_dict, open(os.path.join(res_dir, input_tsv.split('/')[-3], input_tsv.split('/')[-2], 'data.json'), 'w', encoding='utf-8'), indent=4, sort_keys=True, ensure_ascii=False)
