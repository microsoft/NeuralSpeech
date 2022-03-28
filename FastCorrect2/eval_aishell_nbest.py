# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import json
import sys
import torch
import argparse
import re
#from fairseq.models.transformer import TransformerModel
import os
import os.path
import time
import json
import numpy as np
#os.environ["CUDA_VISIBLE_DEVICES"] = '1'
from fairseq import utils
utils.import_user_module(argparse.Namespace(user_dir='./FastCorrect'))
from FastCorrect.fastcorrect_model import FastCorrectModel

def remove_ch_spaces(input_str):
    return re.sub(r"(?<=[\u4e00-\u9fff])(\s+)(?=[\u4e00-\u9fff])", "", input_str.strip())

try:
    model_name_or_path = sys.argv[3]
except:
    model_name_or_path = "checkpoints/shared_baseline"

try:
    iter_decode_max_iter = int(sys.argv[4])
except:
    iter_decode_max_iter = -1

try:
    edit_thre = float(sys.argv[5])
except:
    edit_thre = 0

try:
    nbest_infer_type = sys.argv[6]
except:
    nbest_infer_type = "predict"

try:
    test_epoch = int(sys.argv[7])
    checkpoint_file = "checkpoint{}.pt".format(test_epoch)
except:
    test_epoch = 'best'
    checkpoint_file = "checkpoint_best.pt"


#checkpoint_file = "checkpoint_best.pt"
print("test {}/{}".format(model_name_or_path, checkpoint_file))

data_name_or_path =  # <Path-to-AISHELL1-Training-Binary-Data>
bpe = "sentencepiece"
sentencepiece_model =  # <path-to-sentencepiece_model>, you can use arbitrary sentencepiece for our pretrained model since it is a char-level model 

commonset_dir = "./eval_data"
res_dir = os.path.join(model_name_or_path, ("results_aishell" if (iter_decode_max_iter == -1) else ("results_aishell_b" + str(iter_decode_max_iter) + '_t' + str(edit_thre) + '_' + nbest_infer_type)).replace('results', 'results_' + str(test_epoch)))
tmp_dir = os.path.join(model_name_or_path, ("tmp_aishell" if (iter_decode_max_iter == -1) else ("tmp_aishell_b" + str(iter_decode_max_iter) + '_t' + str(edit_thre) + '_' + nbest_infer_type)).replace('tmp', 'tmp_' + str(test_epoch)))
os.makedirs(res_dir, exist_ok=True)
os.makedirs(tmp_dir, exist_ok=True)
#fout_ex = open(os.path.join(tmp_dir, "exception.log"), "w")

try:
    short_set = sys.argv[1].split(',')
except:
    raise ValueError()

print("short_set:", short_set)

transf_gec = FastCorrectModel.from_pretrained(model_name_or_path, checkpoint_file=checkpoint_file, data_name_or_path=data_name_or_path, bpe=bpe, sentencepiece_model=sentencepiece_model)

transf_gec.eval()
transf_gec.cuda()

nbest_num = 4

for input_tsv in [os.path.join(commonset_dir, f, "aligned_nbest_token_raw.data.json") for f in short_set]:
    all_time = []
    eval_origin_dict = json.load(open(input_tsv, 'r', encoding='utf-8'))
    translate_input_dict = {}
    for k, v in eval_origin_dict["utts"].items():
        translate_input_dict[k] = (v["output"][0]["rec_token"].replace('<eos>', '').strip(), v["output"][0]["token"])
    translated_output_dict = {}
    for k, v in translate_input_dict.items():
        #print(v)
        text, gt = v
        assert len(text.split(" ||| ")) == nbest_num and len(text.split(" ||| ")[0]) > 0
        need_skip = False
        if not need_skip:
            binarized = [transf_gec.binarize(text.replace(" ||| ", " "))]
            batched_hypos, exc_time = transf_gec.generate(binarized, nbest_infer=nbest_num, nbest_infer_type=nbest_infer_type, iter_decode_max_iter=iter_decode_max_iter)
            translated = [transf_gec.decode(hypos[0]['tokens']) for hypos in batched_hypos][0]
            translated = " ".join(translated)
        else:
            translated = " ".join(text.split(" ||| ")[0].replace('<void>', '').split())
            exc_time = 0.0
        all_time.append(exc_time)
        eval_origin_dict["utts"][k]["output"][0]["rec_text"] = " ".join("".join(translated.split()).replace('▁', ' ').strip().split())
        #translated_char = [i for i in eval_origin_dict["utts"][k]["output"][0]["rec_text"]]
        eval_origin_dict["utts"][k]["output"][0]["rec_token"] = translated.replace('<void>', '')
        #print(eval_origin_dict["utts"][k]["output"][0]["rec_token"])

    os.makedirs(os.path.join(res_dir, input_tsv.split('/')[-2]), exist_ok=True)
    with open(os.path.join(res_dir, input_tsv.split('/')[-2], input_tsv.split('/')[-2] + "_time.txt"), 'w') as outfile:
        outfile.write("{}\t{}\t{}\n".format(len(all_time), sum(all_time), sum(all_time)/len(all_time)))
    json.dump(eval_origin_dict, open(os.path.join(res_dir, input_tsv.split('/')[-2], 'data.json'), 'w', encoding='utf-8'), indent=4, sort_keys=True, ensure_ascii=False)

