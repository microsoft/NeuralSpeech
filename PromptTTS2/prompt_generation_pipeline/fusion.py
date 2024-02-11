# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import math
import random
import pandas as pd
from tqdm import tqdm

from prompt_pl import prompt_word_path, use_placeholder

def gen_num_categories(count_categories):
    nums = list()
    counts = math.prod(count_categories)
    cur_list = [0] * len(count_categories)
    for _ in range(counts):
        nums.append(cur_list[:])
        add = 0
        for i in range(len(cur_list)):
            if i == 0:
                cur_list[i] += 1 + add
            else:
                cur_list[i] += add
            add = 0
            if cur_list[i] == count_categories[i]:
                cur_list[i] = 0
                add = 1
    assert add == 1
    return nums

def fusion(categories, csv_lambda, fu_data_path):
    root_path = os.getcwd()  # 'ChatGPTAUG'
    prompt_word_rpath = os.path.join(root_path, prompt_word_path)
    normal_categories = [list(categories[key])[0] for key in categories.keys()]
    count_categories = [len(categories[key].keys()) for key in categories.keys()]
    columns = list(categories.keys())
    columns.extend(['english', 'prompt_class_num'])
    for category in categories.keys():
        for sub_category in categories[category].keys():
            while True:
                try:
                    with open(os.path.join(prompt_word_rpath, sub_category + '.txt'), 'r') as f:
                        lines = f.readlines()
                    if len(lines) == 0:
                        continue
                    for line in lines:
                        word = line.strip().lower()
                        if len(word) != 0:
                            categories[category][sub_category].add(word)
                    break
                except:
                    continue

    num_categories = gen_num_categories(count_categories)
    answer_list = list()
    for num_category in tqdm(num_categories):
        for key in csv_lambda.keys():
            sample_num = csv_lambda[key]
            aug_csv = pd.read_csv(key)
            prompt_class_num = '-'.join([str(num_category_i) for num_category_i in num_category])
            if 'pl' == key.split('.')[0].split('_')[-1]:
                english_list = list()
                while len(english_list) < sample_num:
                    prompt_class_num_place = '-'.join(['U' if num_category_i == 0 and random.random() < use_placeholder else 'P' for num_category_i in num_category])
                    if prompt_class_num_place not in aug_csv['prompt_class_num_place'].unique():
                        continue
                    sentence = aug_csv[aug_csv['prompt_class_num_place'] == prompt_class_num_place]['english'].sample(1).item()
                    for i, place_i in enumerate(prompt_class_num_place.split('-')):
                        if place_i == 'P':
                            category = list(categories.keys())[i]
                            sub_category = list(categories[category].keys())[num_category[i]]
                            word = random.choice(list(categories[category][sub_category]))
                            sentence = sentence.replace(f'[{category}]', word)
                    english_list.append(sentence)
            else:
                english_list = list(aug_csv[aug_csv['prompt_class_num'] == prompt_class_num]['english'].sample(sample_num))
            for english in english_list:
                answer = num_category[:]
                answer.append(english)
                answer.append(prompt_class_num)
                answer_list.append(answer)
    pd.DataFrame(answer_list, columns=columns).to_csv(fu_data_path, index=None)