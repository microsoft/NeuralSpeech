# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

#Usage python align_cal_werdur_fast.py <input-tokened-hypo-text-file> <input-tokened-ref-text-file>
#Note:
#  The script will align <input-tokened-hypo-text-file> (text with errors) with <input-tokened-ref-text-file> (ground-truth text) and obtain the number of target token corresponding with each source token
#  The script might skip sentence which takes too much time for alignment.
#  The aligned result of <input-tokened-hypo-text-file> is in <input-tokened-hypo-text-file>.src.werdur.full, which consists of source tokens with their duration.
#  The aligned result of <input-tokened-ref-text-file> is in <input-tokened-hypo-ref-file>.tgt, which consists of target tokens.
#  The sum of all source token durations is equal to the number of target token.
#  <input-tokened-hypo-text-file>.src.werdur.full is used as source file when generating binary dataset; <input-tokened-hypo-ref-file>.tgt is used as target file when generating binary dataset
import argparse
import copy
import gc
import re, sys
import sys


def tn(text):
    return text.strip().split()

py_dict = {}
def get_dict(dict_path):
    for line in open(dict_path, 'r', encoding='utf-8'):
        char, pinyin = line.strip().split('\t')
        py_dict[char] = pinyin

def get_pinyin(text):
    if text == '<void>':
        return ''
    text = re.sub('([\u4e00-\u9fff])', r' \1 ', text)
    text = re.sub('\s+', ' ', text)
    text_list = text.strip().split(' ')
    for i in range(len(text_list)):
        word = text_list[i]
        if word == '▁':
            text_list[i] = '_'
        else:
            if len(word) == 1 and word >= '\u4e00' and word <= '\u9fff':
                if word in py_dict:
                    text_list[i] = py_dict[word]
                else:
                    text_list[i] = 'unk'
    return ''.join(text_list)

def edit_matrix(query, reference):
    matrix = [[0] * (len(query) + 1) for _ in range(len(reference) + 1)]
    j_count = 0
    for j in range(1, len(query) + 1):
        if query[j-1] == '<void>':
            j_count += 1
        matrix[0][j] = j - j_count
    for i in range(len(reference) + 1):
        matrix[i][0] = i
    for i in range(1, len(reference) + 1):
        for j in range(1, len(query) + 1):
            if query[j-1] == '<void>':
                matrix[i][j] = matrix[i][j-1]
            elif query[j-1] == reference[i-1]:
                matrix[i][j] = matrix[i-1][j-1]
            else:
                matrix[i][j] = min(matrix[i-1][j-1]+1, matrix[i-1][j]+1, matrix[i][j-1]+1)
    return matrix

def get_raw_align_script(matrix, query, reference):
    j = len(query)
    i = len(reference)
    edit_script = []
    score = [sys.maxsize for i in range(3)]
    while i>0 or j>0:
        #del
        if i>0 and matrix[i-1][j] == matrix[i][j] - 1:
            score[0] = len(get_pinyin(reference[i-1]))
        #ins
        if j>0 and matrix[i][j-1] == matrix[i][j] - 1:
            score[1] = len(get_pinyin(query[j-1]))
        #sub
        if i>0 and j>0 and matrix[i-1][j-1] == matrix[i][j]-1:
            score[2] = edit_matrix(get_pinyin(query[j-1]), get_pinyin(reference[i-1]))[-1][-1]
        idx = score.index(min(score))

        #identical
        if i>0 and j>0 and matrix[i-1][j-1] == matrix[i][j] and score[idx] == sys.maxsize:
            edit_script.append(('EQUAL', query[j-1], reference[i-1]))
            i -= 1
            j -= 1
        #del
        elif idx ==0:
            edit_script.append(('DEL', '<void>', reference[i-1]))
            i -= 1
        #ins
        elif idx == 1:
            edit_script.append(('INS', query[j-1], '<void>'))
            j -= 1
        #sub
        elif idx == 2:
            edit_script.append(('SUB', query[j-1], reference[i-1]))
            i -= 1
            j -= 1
        else:
            raise Exception('unexpected edit matrix')
        score = [sys.maxsize for i in range(3)]
    edit_script.reverse()
    return edit_script

def werdur(query, reference):
    matrix = edit_matrix(query, reference)
    #print (matrix)
    wer = 0
    j = len(query)
    i = len(reference)
    edit_script = []
    dur_list = []
    ins_edit = 0

    while i>0 or j>0:
        score = [sys.maxsize for i in range(3)]
        #del
        if j>0 and query[j-1] == '<void>':
            dur_list.append(0)
            j -= 1
            continue
        if j>0 and matrix[i][j-1] == matrix[i][j] - 1:
            score[0] = len(get_pinyin(query[j-1]))
        #ins
        if i>0 and matrix[i-1][j] == matrix[i][j] - 1:
            score[1] = len(get_pinyin(reference[i-1]))
        #sub
        if i>0 and j>0 and matrix[i-1][j-1] == matrix[i][j]-1:
            score[2] = edit_matrix(get_pinyin(query[j-1]), get_pinyin(reference[i-1]))[-1][-1]
        idx = score.index(min(score))

        #identical
        if i>0 and j>0 and matrix[i-1][j-1] == matrix[i][j] and score[idx] == sys.maxsize:
            #print ("identical: " + query[j-1]+" "+ reference[i-1])
            if ins_edit == 0:
                dur_list.append(1)
            else:
                dur_list.append(-1+ins_edit)
                ins_edit = 0
            i -= 1
            j -= 1
        #del
        elif idx ==0:
            #print ('del: '+ query[j-1])
            dur_list.append(0)
            wer += 1
            j -= 1
        #ins
        elif idx == 1:
            #print ('ins: '+ reference[i-1])
            ins_edit -= 1
            wer += 1
            i -= 1
        #sub
        elif idx == 2:
            #print ('sub: '+ query[j-1]+" "+ reference[i-1])
            dur_list.append(-1+ins_edit)
            ins_edit = 0
            wer += 1
            i -= 1
            j -= 1
        else:
            raise Exception('unexpected edit matrix')

    dur_list.reverse()
    if ins_edit != 0:
        for i in range(len(dur_list)):
            if dur_list[i] != 0:
                dur_list[i] = -(abs(dur_list[i]) + abs(ins_edit))
                break
    dur_list = map(str, dur_list)
    dur = ' '.join(dur_list)
    return dur, wer

def align(query, reference, align_recog_list):
    matrix = edit_matrix(query, reference)
    raw_align_script = get_raw_align_script(matrix, query, reference)
    align_query = []
    align_reference = []
    for i in range(len(raw_align_script)):
        e = raw_align_script[i]
        align_query.append(e[1])
        align_reference.append(e[2])
        if e[2] == '<void>' and e[0] == 'INS':
            for j in range(len(align_recog_list)):
                if len(align_recog_list[j]) != 0:
                    align_recog_list[j].insert(i, '<void>')
    return align_query, align_reference, align_recog_list

get_dict('pinyin_dict.txt')

def gen_align_dur_nbest(recog_list, trans):
    #obj = Align_Nbest('pinyin_dict.txt')
    #trans = trans_line.strip().split()
    #recog_list = [i.strip().split() for i in recog_line.strip().split(' # ')]
    #-------align nbest candidate------------
    recog_len = [len(recog) for recog in recog_list]
    max_idx = recog_len.index(max(recog_len))
    align_recog_list = [[] for i in range(len(recog_list))]
    align_recog_list[max_idx] = recog_list[max_idx]
    wer_list = []
    dur_list = []
    for i in range(len(recog_list)):
        if i != max_idx:
            align_query, align_reference, align_recog_list = align(recog_list[i], align_recog_list[max_idx], align_recog_list)
            align_recog_list[i] = align_query
    #---------get wer dur and label the best candidate--------
    align_recog_nbest = []
    for recog in align_recog_list:
        dur, wer = werdur(recog, trans)
        wer_list.append(wer)
        dur_list.append(dur)
    min_wer = min(wer_list)
    for i in range(len(wer_list)):
        if wer_list[i] == min_wer:
            align_recog_nbest.append(' '.join(align_recog_list[i]) + ' 1')
        else:
            align_recog_nbest.append(' '.join(align_recog_list[i]) + ' 0')
    align_result = ' ||| '.join(align_recog_nbest) + ' |||| ' + ' ||| '.join(dur_list)

    return align_result, " ".join(trans)

hypo_file = sys.argv[1]
ref_file = sys.argv[2]
all_hypo_line = []
all_ref_line = []

print("Loading: ", hypo_file)
with open(hypo_file, 'r', encoding='utf-8') as infile:
    for line in infile.readlines():
        all_hypo_line.append([i.strip().split() for i in line.strip().split(' # ')])

print("Loading: ", ref_file)
with open(ref_file, 'r', encoding='utf-8') as infile:
    for line in infile.readlines():
        all_ref_line.append(line.strip().split())


import time
start_time = time.time()
count = 0
count_no_skip = 0
with open(ref_file + '.tgt', 'w', encoding='utf-8') as outfile_tgt:
    with open(hypo_file + '.src.werdur.full', 'w', encoding='utf-8') as outfile_full:
        for hypo_list, ref_list in zip(all_hypo_line, all_ref_line):
            skip_this = False
            if not hypo_list:
                skip_this = True
            if not ref_list:
                skip_this = True
            if not skip_this:
                results = gen_align_dur_nbest(hypo_list, ref_list)
                if results:
                    count_no_skip += 1
                    output_src_str, output_tgt_str = results
                    outfile_full.write(output_src_str + "\n")
                    outfile_tgt.write(output_tgt_str + "\n")

            count += 1

            if count % 10000 == 0:
                print(count, "in", time.time() - start_time, "s")
                print(count_no_skip, "not skipped!")

print("Overall: {}/{} finished successful!".format(count_no_skip, count))
