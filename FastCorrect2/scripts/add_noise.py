# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

# Usage python add_noise.py <input-raw-text-file> <output-noised-text-file> <random-seed>

import os
import sys
import random
import numpy as np

sim_dict = {}
vocab_2char = []
with open('sim_prun_char.txt', 'r', encoding='utf-8') as infile:
    for line in infile.readlines():
        line = line.strip()
        first_char = line[0]
        if first_char not in sim_dict.keys():
            sim_dict[first_char] = {1: {}, 2: {}, 3: {}, 4: {}, 5:{}, 6:{}, 7: {}, 8: {}}
        vocab, sim_vocab = line.split('\t')
        sim_vocab = sim_vocab.split()

        vocab_length = len(vocab)
        if vocab_length <= 2:
            vocab_2char.append(vocab)
        #if vocab_length == 4 and len(sim_vocab) <= 4:
        #    print("skip ", line)
        #    continue
        #if vocab_length == 5 and len(sim_vocab) <= 3:
        #    print("skip ", line)
        #    continue
        #if vocab_length == 6 and len(sim_vocab) <= 2:
        #    print("skip ", line)
        #    continue
        if len(sim_vocab) == 1:
            #print("skip ", line)
            continue
        if vocab_length >= 9:
            #print("skip ", line)
            continue
        sim_dict[first_char][vocab_length][vocab] = sim_vocab

noise_ratio = 0.2
beam_size = 4
all_beam = list(range(beam_size))
prob_beam = [0.15, 0.22, 0.28, 0.35]

candidate_logit = [6, 5, 5, 4, 4, 3, 3, 3, 2, 2, 2, 1, 1, 1, 1]

infile = sys.argv[1]
outfile = sys.argv[2]
random.seed(int(sys.argv[3]))
np.random.seed(int(sys.argv[3]))

SUB = 0
INS_L = 1
INS_R = 2
DEL = 3
all_op = [SUB, INS_L, INS_R, DEL]
prob_op = [4/7, 1/14, 1/14, 2/7]

def add_noise(token, op, candidate, meta_ins=None):
    if op == SUB:
        if candidate is None:
            #assert len(token) == 1
            return np.random.choice(vocab_2char)
        else:
            prob_candidate = [i/sum(candidate_logit[:len(candidate)]) for i in candidate_logit[:len(candidate)]]
            return np.random.choice(candidate, p=prob_candidate)
    elif op == DEL:
        return ""
    elif op == INS_L:
        if meta_ins:
            if random.random() < 0.4:
                return meta_ins + token
            else:
                return np.random.choice(vocab_2char) + token
        else:
            random_token = np.random.choice(vocab_2char)
            return random_token + token, random_token
    elif op == INS_R:
        if meta_ins:
            if random.random() < 0.4:
                return token + meta_ins
            else:
                return token + np.random.choice(vocab_2char)
        else:
            random_token = np.random.choice(vocab_2char)
            return token + random_token, random_token
    else:
        raise ValueError("impossible op {}!".format(op))

def noise_meta_beam(token, meta_noise, candidate):
    if meta_noise == INS_L or meta_noise == INS_R:
        return add_noise(token, meta_noise, candidate, None)
    else:
        return add_noise(token, meta_noise, candidate, None), None

def noise_other_beam(token, meta_noise, candidate, meta_ins):
    if random.random() < 0.5:
        if meta_noise == SUB:
            prob_other_beam_op = [4/7, 1/14, 1/14, 2/7]
        elif meta_noise == DEL:
            prob_other_beam_op = [4/7, 0, 0, 3/7]
        elif meta_noise == INS_L:
            prob_other_beam_op = [4/7, 3/7, 0, 0]
        elif meta_noise == INS_R:
            prob_other_beam_op = [4/7, 0, 3/7, 0]
        else:
            raise ValueError("impossible meta_noise {}!".format(meta_noise))
        token = add_noise(token, np.random.choice(all_op, p=prob_other_beam_op), candidate, meta_ins)
        if isinstance(token, tuple):
            token = token[0]
        return token
    else:
        return token

def is_english_letter(c):
    if (65<=ord(c)<=90) or (97<=ord(c)<=122):
        return True
    else:
        return False

import time
begin_time = time.time()

with open(infile, 'r', encoding='utf-8') as infile:
    with open(outfile, 'w', encoding='utf-8') as outfile:
        for count, line in enumerate(infile.readlines()):
            if count % 5000 == 1:
                print("{} finished in {}s".format(count-1, time.time()-begin_time))
            line = line.strip()
            if not line:
                continue
            new_lines = ["" for _ in range(beam_size)]
            sen_length = len(line)
            i = 0
            while i < sen_length:
                tok = line[i]
                if tok == " ":
                    i += 1
                    for j in range(beam_size):
                        new_lines[j] += tok
                    continue

                if is_english_letter(tok):
                    i += 1
                    while i < sen_length and is_english_letter(line[i]):
                        tok += line[i]
                        i += 1
                    if random.random() < 0.2:
                        modify_beam = np.random.choice(all_beam, p=prob_beam)
                        meta_noise = np.random.choice(all_op, p=prob_op)
                        meta_new_token, meta_ins = noise_meta_beam(tok, meta_noise, None)
                        new_lines[modify_beam] += meta_new_token
                        for j in range(beam_size):
                            if j == modify_beam:
                                continue
                            else:
                                other_new_token = noise_other_beam(tok, meta_noise, None, meta_ins)
                                new_lines[j] += other_new_token
                        if tok == 'barbara':
                            print(modify_beam, meta_noise, meta_new_token, meta_ins, new_lines)
                    else:
                        for j in range(beam_size):
                            new_lines[j] += tok
                    continue

                if random.random() < noise_ratio:
                    if tok not in sim_dict.keys():
                        modify_beam = np.random.choice(all_beam, p=prob_beam)
                        meta_noise = np.random.choice(all_op, p=prob_op)
                        meta_new_token, meta_ins = noise_meta_beam(tok, meta_noise, None)
                        new_lines[modify_beam] += meta_new_token
                        for j in range(beam_size):
                            if j == modify_beam:
                                continue
                            else:
                                other_new_token = noise_other_beam(tok, meta_noise, None, meta_ins)
                                new_lines[j] += other_new_token
                        continue
                    if sen_length - i >= 8:
                        if line[i: i+8] in sim_dict[tok][8].keys():
                            tok = line[i: i+8]
                            i += 8
                            modify_beam = np.random.choice(all_beam, p=prob_beam)
                            meta_noise = np.random.choice(all_op, p=prob_op)
                            meta_new_token, meta_ins = noise_meta_beam(tok, meta_noise, (sim_dict[tok[0]][8][tok] if random.random() < 1.1 else None))
                            new_lines[modify_beam] += meta_new_token
                            for j in range(beam_size):
                                if j == modify_beam:
                                    continue
                                else:
                                    other_new_token = noise_other_beam(tok, meta_noise, (sim_dict[tok[0]][8][tok] if random.random() < 1.1 else None), meta_ins)
                                    new_lines[j] += other_new_token
                            continue
                        else:
                            pass
                    if sen_length - i >= 7:
                        if line[i: i+7] in sim_dict[tok][7].keys():
                            tok = line[i: i+7]
                            i += 7
                            modify_beam = np.random.choice(all_beam, p=prob_beam)
                            meta_noise = np.random.choice(all_op, p=prob_op)
                            meta_new_token, meta_ins = noise_meta_beam(tok, meta_noise, (sim_dict[tok[0]][7][tok] if random.random() < 1.1 else None))
                            new_lines[modify_beam] += meta_new_token
                            for j in range(beam_size):
                                if j == modify_beam:
                                    continue
                                else:
                                    other_new_token = noise_other_beam(tok, meta_noise, (sim_dict[tok[0]][7][tok] if random.random() < 1.1 else None), meta_ins)
                                    new_lines[j] += other_new_token
                            continue
                        else:
                            pass
                    if sen_length - i >= 6:
                        if line[i: i+6] in sim_dict[tok][6].keys():
                            tok = line[i: i+6]
                            i += 6
                            modify_beam = np.random.choice(all_beam, p=prob_beam)
                            meta_noise = np.random.choice(all_op, p=prob_op)
                            meta_new_token, meta_ins = noise_meta_beam(tok, meta_noise, (sim_dict[tok[0]][6][tok] if random.random() < 1.1 else None))
                            new_lines[modify_beam] += meta_new_token
                            for j in range(beam_size):
                                if j == modify_beam:
                                    continue
                                else:
                                    other_new_token = noise_other_beam(tok, meta_noise, (sim_dict[tok[0]][6][tok] if random.random() < 1.1 else None), meta_ins)
                                    new_lines[j] += other_new_token
                            continue
                        else:
                            pass
                    if sen_length - i >= 5:
                        if line[i: i+5] in sim_dict[tok][5].keys():
                            tok = line[i: i+5]
                            i += 5
                            modify_beam = np.random.choice(all_beam, p=prob_beam)
                            meta_noise = np.random.choice(all_op, p=prob_op)
                            meta_new_token, meta_ins = noise_meta_beam(tok, meta_noise, (sim_dict[tok[0]][5][tok] if random.random() < 0.998 else None))
                            new_lines[modify_beam] += meta_new_token
                            for j in range(beam_size):
                                if j == modify_beam:
                                    continue
                                else:
                                    other_new_token = noise_other_beam(tok, meta_noise, (sim_dict[tok[0]][5][tok] if random.random() < 0.998 else None), meta_ins)
                                    new_lines[j] += other_new_token
                            continue
                        else:
                            pass
                    if sen_length - i >= 4:
                        if line[i: i+4] in sim_dict[tok][4].keys():
                            tok = line[i: i+4]
                            i += 4
                            modify_beam = np.random.choice(all_beam, p=prob_beam)
                            meta_noise = np.random.choice(all_op, p=prob_op)
                            meta_new_token, meta_ins = noise_meta_beam(tok, meta_noise, (sim_dict[tok[0]][4][tok] if random.random() < 0.95 else None))
                            new_lines[modify_beam] += meta_new_token
                            for j in range(beam_size):
                                if j == modify_beam:
                                    continue
                                else:
                                    other_new_token = noise_other_beam(tok, meta_noise, (sim_dict[tok[0]][4][tok] if random.random() < 0.95 else None), meta_ins)
                                    new_lines[j] += other_new_token
                            continue
                        else:
                            pass
                    if sen_length - i >= 3:
                        if line[i: i+3] in sim_dict[tok][3].keys():
                            tok = line[i: i+3]
                            i += 3
                            modify_beam = np.random.choice(all_beam, p=prob_beam)
                            meta_noise = np.random.choice(all_op, p=prob_op)
                            meta_new_token, meta_ins = noise_meta_beam(tok, meta_noise, (sim_dict[tok[0]][3][tok] if random.random() < 0.93 else None))
                            new_lines[modify_beam] += meta_new_token
                            for j in range(beam_size):
                                if j == modify_beam:
                                    continue
                                else:
                                    other_new_token = noise_other_beam(tok, meta_noise, (sim_dict[tok[0]][3][tok] if random.random() < 0.93 else None), meta_ins)
                                    new_lines[j] += other_new_token
                            continue
                        else:
                            pass
                    if sen_length - i >= 2:
                        if line[i: i+2] in sim_dict[tok][2].keys():
                            tok = line[i: i+2]
                            i += 2
                            modify_beam = np.random.choice(all_beam, p=prob_beam)
                            meta_noise = np.random.choice(all_op, p=prob_op)
                            meta_new_token, meta_ins = noise_meta_beam(tok, meta_noise, (sim_dict[tok[0]][2][tok] if random.random() < 0.9 else None))
                            new_lines[modify_beam] += meta_new_token
                            for j in range(beam_size):
                                if j == modify_beam:
                                    continue
                                else:
                                    other_new_token = noise_other_beam(tok, meta_noise, (sim_dict[tok[0]][2][tok] if random.random() < 0.9 else None), meta_ins)
                                    new_lines[j] += other_new_token
                            continue
                        else:
                            pass
                    if sen_length - i >= 1:
                        if line[i: i+1] in sim_dict[tok][1].keys():
                            tok = line[i: i+1]
                            i += 1
                            modify_beam = np.random.choice(all_beam, p=prob_beam)
                            meta_noise = np.random.choice(all_op, p=prob_op)
                            meta_new_token, meta_ins = noise_meta_beam(tok, meta_noise, (sim_dict[tok[0]][1][tok] if random.random() < 0.9 else None))
                            new_lines[modify_beam] += meta_new_token
                            for j in range(beam_size):
                                if j == modify_beam:
                                    continue
                                else:
                                    other_new_token = noise_other_beam(tok, meta_noise, (sim_dict[tok[0]][1][tok] if random.random() < 0.9 else None), meta_ins)
                                    new_lines[j] += other_new_token
                            continue
                        else:
                            pass
                    else:
                        raise ValueError("Impossible condition!")
                else:
                    i += 1
                    for j in range(beam_size):
                        new_lines[j] += tok

            need_skip = False
            for iter_line in new_lines:
                if not iter_line.strip():
                    need_skip = True
                    break
            if need_skip:
                continue

            final_line = "\t".join([line] + new_lines)
            outfile.write(final_line + '\n')