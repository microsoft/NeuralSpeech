# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import sys
import math
import openai
import random
import argparse
import itertools
import pandas as pd
from tqdm import tqdm
import config as cfg


prompt_placeholder_path = 'prompt_pl.csv'
prompt_word_path = 'prompt_word'
normal_threshold = 0.5 
use_placeholder = 0.5 
ask_error_time = 10
# The first key is the normal value
prefix_sentence = f"I have some sentences, can you combine these sentences into one sentence to describe the style of speech with the same meaning? You can add different elements to make the description not boring and use different sentence structure in {sentence_num} different sentences. The sentences are: "
prefix_sentence_word = f"I have some sentences, can you combine these sentences into one phrase with fewer than [word_num] words to describe the style of speech with the same meaning? You can add different elements to make the description not boring and use different sentence structure in {sentence_num} different sentences.  Please note that the sentences you generate must include [class_id]. The sentences are: "
prefix_sp = f"I have some sentences, can you combine these sentences into one sentence and keep all of the '[]' to describe the style of speech with the same meaning? You can add different elements to make the description not boring and use different sentence structure in {sentence_num} different sentences. The sentences are:  "
prefix_sp_word = f"I have some sentences, can you combine these sentences into one phrase with fewer than [word_num] and keep all of the '[]' to describe the style of speech with the same meaning? You can add different elements to make the description not boring and use different sentence structure in {sentence_num} different sentences.  Please note that the sentences you generate must include [class_id]. The sentences are: "

sentence_num = cfg.sentence_num  
# Stage one
prefix_keyword = cfg.prefix_keyword
# Stage two
prefix_template = cfg.prefix_template
# Stage three
prefix_sentence_sentences = cfg.prefix_sentence_sentences
prefix_sentence_phrases = cfg.prefix_sentence_phrases
prefix_sentence_words = cfg.prefix_sentence_words

prefix_sp_sentences = cfg.prefix_sp_sentences
prefix_sp_phrases = cfg.prefix_sp_phrases
prefix_sp_words = cfg.prefix_sp_words

def process(csv_path, thread_num):
    root_path = os.getcwd()
    csv_rpath = os.path.join(root_path, csv_path)
    output_df = pd.read_csv(csv_rpath) if os.path.exists(csv_rpath) else pd.DataFrame()
    for thread_num_i in range(thread_num):
        csv_i_path = str(thread_num_i) + '_' + csv_path
        csv_i_rpath = os.path.join(root_path, 'nohup_out', csv_i_path)
        if not os.path.exists(csv_i_rpath):
            print('Not Exists: ', csv_i_rpath)
            continue
        while True:
            try:
                csv_i = pd.read_csv(csv_i_rpath)
                output_df = output_df.append(csv_i).drop_duplicates(subset='english')
                output_df.to_csv(csv_rpath, index=None)
                break
            except:
                continue

def gen_combinations(categories_list):
    output = list()
    for i in range(1, len(categories_list) + 1):
        for combination in list(itertools.combinations(categories_list, i)):
            output.append(list(combination))
    return output

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

def ask(question, count_num=0, use_api=True, direct_return=False):
    if use_api:
        text = str()
        for _ in range(ask_error_time):
            try:
                completion = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "user",
                         "content": question},
                    ]
                )
                text = completion.choices[0].message['content']
                break
            except:
                continue
    print("Question: ", question)
    print("Answer: ", text)
    if direct_return:
        print(text)
        return text
    text = text.split('\n')
    answers = list()
    search_id = 0
    while search_id < count_num:
        search_id += 1
        for line in text:
            line = line.strip()
            if line.startswith("{}. ".format(search_id)):
                answers.append(line[line.find('.')+2:].strip())
    print(answers)
    return answers

def stage1(categories, keywords_num, use_api=True):
    root_path = os.getcwd()  # 'ChatGPTAUG'
    prompt_word_rpath = os.path.join(root_path, prompt_word_path)
    os.makedirs(prompt_word_rpath, exist_ok=True)
    for category in categories.keys():
        for i, sub_category in enumerate(categories[category].keys()):
            sub_category_path = os.path.join(prompt_word_rpath, sub_category + '.txt')
            if os.path.exists(sub_category_path):
                with open(sub_category_path, 'r') as f:
                    lines = f.readlines()
                for line in lines:
                    word = line.strip().lower()
                    if len(word) != 0:
                        categories[category][sub_category].add(word)
            keywords_num_i = keywords_num - len(categories[category][sub_category]) if keywords_num - len(categories[category][sub_category]) > 0 else 0
            while keywords_num_i > 0:
                prompt = prefix_keyword.replace("[keywords_num]", str(keywords_num_i)).replace("[sub_category]", ' '.join([sub.lower() for sub in sub_category.split('_')[::-1]]))
                answers = ask(prompt, count_num=keywords_num_i, use_api=use_api)
                for answer in answers:
                    if len(answer) != 0:
                        categories[category][sub_category].add(answer.lower())
                keywords_num_i = keywords_num_i - len(categories[category][sub_category]) if keywords_num_i - len(categories[category][sub_category]) > 0 else 0
            if i == 0:
                repeat_words = categories[category][sub_category]
            else:
                repeat_words = repeat_words & categories[category][sub_category]
        for i, sub_category in enumerate(categories[category].keys()):
            for repeat_word in repeat_words:
                categories[category][sub_category].remove(repeat_word)
            with open(os.path.join(prompt_word_rpath, sub_category + '.txt'), 'w') as f:
                f.write('\n'.join(list(categories[category][sub_category])))

def stage2(categories, template_num, use_api=True):
    root_path = os.getcwd()  # 'ChatGPTAUG'
    prompt_placeholder_rpath = os.path.join(root_path, prompt_placeholder_path)
    if os.path.exists(prompt_placeholder_rpath):
        prompt_placeholder = pd.read_csv(prompt_placeholder_rpath).drop_duplicates()
    else:
        prompt_placeholder = pd.DataFrame(columns=['class', 'prompt'])
    for category in categories.keys():
        template_num_i = template_num - len(prompt_placeholder[prompt_placeholder['class']==category]) if template_num - len(prompt_placeholder[prompt_placeholder['class']==category]) > 0 else 0
        while template_num_i != 0:
            prompt = random.choice(prefix_template).replace("[category]", category.lower())
            answers = ask(prompt, count_num=sentence_num, use_api=use_api)
            for answer in answers:
                if len(answer) != 0 and answer.count('[placeholder]') == 1:
                    prompt_placeholder = prompt_placeholder.append(pd.DataFrame([[category, answer]], columns=['class', 'prompt'])).drop_duplicates()
            template_num_i = template_num_i - len(prompt_placeholder[prompt_placeholder['class'] == category]) if template_num_i - len(prompt_placeholder[prompt_placeholder['class'] == category]) > 0 else 0
    prompt_placeholder.to_csv(prompt_placeholder_rpath, index=None)


def stage3(categories, api_key, word_num, sp, turns, csv_path='prompt.csv', prompt=str(), thread_num_i=0, thread_num=1, use_api=True):
    if use_api:
        import openai
        openai.api_key = api_key
        print("USE API!")

    if len(prompt) != 0:
        ask(prompt, use_api=use_api, direct_return=True)
        sys.exit()

    print("Word Num: ", word_num)
    root_path = os.getcwd()  # 'ChatGPTAUG'
    prompt_placeholder_rpath = os.path.join(root_path, prompt_placeholder_path)
    prompt_word_rpath = os.path.join(root_path, prompt_word_path)
    normal_categories = [list(categories[key])[0] for key in categories.keys()]
    for category in categories.keys():
        for sub_category in categories[category].keys():
            while True:
                try:
                    if thread_num_i == 0:
                        words = set()
                    with open(os.path.join(prompt_word_rpath, sub_category + '.txt'), 'r') as f:
                        lines = f.readlines()
                    if len(lines) == 0:
                        continue
                    for line in lines:
                        word = line.strip().lower()
                        if len(word) != 0:
                            categories[category][sub_category].add(word)
                            if thread_num_i == 0:
                                words.add(word)
                    if thread_num_i == 0:
                        with open(os.path.join(prompt_word_rpath, sub_category + '.txt'), 'w') as f:
                            f.write('\n'.join(list(words)))
                    break
                except:
                    continue

    assert len(categories) == len(normal_categories)
    count_categories = [len(categories[key].keys()) for key in categories.keys()]
    columns = list(categories.keys())
    columns.extend(['english', 'prompt_class_num', 'prompt_class_num_normal', 'prompt_class_num_place', 'question'])
    while True:
        try:
            prompt_placeholder = pd.read_csv(prompt_placeholder_rpath).drop_duplicates()
            break
        except:
            continue
    for i in range(len(prompt_placeholder)):
        try:
            if '[placeholder]' not in prompt_placeholder.iloc[i]['prompt']:
                prompt_placeholder = prompt_placeholder.drop(i)
        except:
            continue
    if thread_num_i == 0:
        while True:
            try:
                prompt_placeholder.to_csv(prompt_placeholder_rpath, index=None)
                break
            except:
                continue
    prompt_placeholder = prompt_placeholder.set_index('class')

    sub_csv_path = str(thread_num_i) + '_' + csv_path
    os.makedirs('nohup_out', exist_ok=True)
    sub_csv_rpath = os.path.join(root_path, 'nohup_out', sub_csv_path)
    if os.path.exists(sub_csv_rpath):
        os.remove(sub_csv_rpath)
    csv_rpath = os.path.join(root_path, csv_path)
    answer_list = list()
    # aug_csv = pd.read_csv(sub_csv_rpath) if os.path.exists(sub_csv_rpath) else pd.DataFrame(columns=columns)
    # aug_csv = pd.DataFrame(columns=columns)
    aug_csv = pd.read_csv(csv_rpath) if os.path.exists(csv_rpath) else pd.DataFrame(columns=columns)

    if sp:
        print('USE PLACEHOLDER!')
        aug_csv_dict = aug_csv.set_index('prompt_class_num_place')
        num_categories = gen_combinations(list(categories.keys()))
        random.shuffle(num_categories)
        for turn in tqdm(range(turns)):
            for num_category in tqdm(num_categories):
                answers = list()
                while len(answers) == 0:
                    prompt_list = list()
                    while len(prompt_list) == 0:
                        index_list = list()
                        class_list = ['U'] * len(categories)
                        for num_category_i in num_category:
                            class_list[list(categories.keys()).index(num_category_i)] = 'P'
                            index_list.append(list(categories.keys()).index(num_category_i))
                        while len(index_list) != 0:
                            sample_index = random.sample(index_list, random.randint(1, len(index_list)))
                            prompt_class_num = '-'.join(['P' if i in sample_index else 'U' for i in range(len(category))])
                            if prompt_class_num not in aug_csv_dict.index or len(sample_index) == 1 and random.random() > use_placeholder:
                                sample_index_i = random.sample(index_list, 1)[0]
                                sentence = prompt_placeholder.loc[list(categories.keys())[sample_index_i]].sample(1)['prompt'].item()
                                sentence_re = sentence.replace('[placeholder]', f'[{list(categories.keys())[sample_index_i]}]')
                                if f'[{list(categories.keys())[sample_index_i]}]' not in sentence_re:
                                    prompt_list = list()
                                    break
                                prompt_list.append(sentence_re)
                                index_list.remove(sample_index_i)
                            else:
                                try:
                                    sentence = aug_csv_dict.loc[prompt_class_num].sample(1)['english'].item()
                                except:
                                    prompt_list = list()
                                    break
                                check_flag = False
                                for sample_index_i in sample_index:
                                    if f'[{list(categories.keys())[sample_index_i]}]' not in sentence:
                                        prompt_list = list()
                                        check_flag = True
                                        break
                                if check_flag:
                                    break
                                prompt_list.append(sentence)
                                for sample_index_i in sample_index:
                                    index_list.remove(sample_index_i)
                    random.shuffle(prompt_list)
                    if word_num == -3:
                        prefix = prefix_sp_words
                    elif word_num == -2:
                        prefix = prefix_sp_phrases
                    elif word_num == -1:
                        prefix = prefix_sp_sentences
                    elif word_num == 0:
                        prefix = prefix_sp
                    elif word_num > 0:
                        class_use = num_category[:]
                        random.shuffle(class_use)
                        prompt_word_num = word_num * len(prompt_list)
                        prefix = prefix_sp_word.replace('[word_num]', str(prompt_word_num)).replace('[class_id]', ', '.join(class_use))
                    prompt = ' '.join(prompt_list)
                    answers_check = ask(prefix + prompt, count_num=sentence_num,use_api=use_api)
                    answers = list()
                    for answer in answers_check:
                        check_flag = False
                        for num_category_i in num_category:
                            if f'[{num_category_i}]' not in answer or answer.count(f'[{num_category_i}]') != 1:
                                check_flag = True
                                break
                            answer = answer.replace(f'[{num_category_i}]', f'<{num_category_i}>')
                        if check_flag:
                            continue
                        answer = answer.replace('[', '').replace(']', '')
                        for num_category_i in num_category:
                            for num_category_i in num_category:
                                answer = answer.replace(f'<{num_category_i}>', f'[{num_category_i}]')
                        answers.append(answer)
                num_list = '-'.join([str(num) for num in class_list])
                for answer in answers:
                    class_list_copy = class_list[:]
                    class_list_copy.append(answer)
                    class_list_copy.append(None)
                    class_list_copy.append(None)
                    class_list_copy.append(num_list)
                    class_list_copy.append(prompt)
                    answer_list.append(class_list_copy)
                aug_csv = aug_csv.append(pd.DataFrame(answer_list, columns=columns)).drop_duplicates(subset='english')

                while True:
                    try:
                        aug_csv.to_csv(sub_csv_rpath, index=None)
                        output_df = pd.read_csv(csv_rpath) if os.path.exists(csv_rpath) else pd.DataFrame()
                        break
                    except:
                        continue
                process(csv_path, thread_num)
                aug_csv = aug_csv.append(output_df).drop_duplicates(subset='english')
                aug_csv_dict = aug_csv.set_index('prompt_class_num_place')
            aug_csv.to_csv(sub_csv_rpath, index=None)
            process(csv_path, thread_num)
    else:
        print('NO USE PLACEHOLDER!')
        aug_csv_dict = aug_csv.set_index('prompt_class_num_normal')
        num_categories = gen_num_categories(count_categories)
        random.shuffle(num_categories)
        for turn in tqdm(range(turns)):
            for num_category in tqdm(num_categories):
                answers = list()
                while len(answers) == 0:
                    prompt_list = list()
                    while len(prompt_list) == 0:
                        index_list = [i for i in range(len(num_category))]
                        class_list = [list(categories[list(categories.keys())[i]].keys())[num_category[i]] for i in range(len(num_category))]
                        normal_num_category = ['N' if num_category[i] == 0 and random.random() > normal_threshold else str(num_category[i]) for i in range(len(num_category))]
                        while len(index_list) != 0:
                            sample_index = random.sample(index_list, random.randint(1, len(index_list)))
                            prompt_class_num = '-'.join([normal_num_category[i] if i in sample_index else 'N' for i in range(len(normal_num_category))])
                            if prompt_class_num not in aug_csv_dict.index or len(sample_index) == 1 and random.random() > use_placeholder:
                                sample_index_i = random.sample(index_list, 1)[0]
                                if normal_num_category[sample_index_i] == 'N':
                                    index_list.remove(sample_index_i)
                                    continue
                                word = random.choice(list(categories[list(categories.keys())[sample_index_i]][list(categories[list(categories.keys())[sample_index_i]].keys())[num_category[sample_index_i]]]))
                                sentence = prompt_placeholder.loc[list(categories.keys())[sample_index_i]].sample(1)['prompt'].item()
                                sentence_re = sentence.replace('[placeholder]', word)
                                if '[placeholder]' in sentence_re:
                                    prompt_list = list()
                                    break
                                prompt_list.append(sentence_re)
                                index_list.remove(sample_index_i)
                            else:
                                try:
                                    sentence = aug_csv_dict.loc[prompt_class_num].sample(1)['english'].item()
                                except:
                                    prompt_list = list()
                                    break
                                prompt_list.append(sentence)
                                for sample_index_i in sample_index:
                                    index_list.remove(sample_index_i)
                    random.shuffle(prompt_list)
                    if word_num == -3:
                        prefix = prefix_sentence_words
                    elif word_num == -2:
                        prefix = prefix_sentence_phrases
                    elif word_num == -1:
                        prefix = prefix_sentence_sentences
                    elif word_num == 0:
                        prefix = prefix_sentence
                    elif word_num > 0:
                        class_use = list()
                        for i in range(len(normal_num_category)):
                            if normal_num_category[i] != 'N':
                                class_use.append(list(categories.keys())[i])
                        random.shuffle(class_use)
                        prompt_word_num = word_num * len(prompt_list)
                        prefix = prefix_sentence_word.replace('[word_num]', str(prompt_word_num)).replace('[class_id]', ', '.join(class_use))

                    prompt = ' '.join(prompt_list)
                    answers = ask(prefix + prompt, count_num=sentence_num, use_api=use_api)

                num_list = '-'.join([str(num) for num in num_category])
                normal_num_list = '-'.join(normal_num_category)
                for answer in answers:
                    class_list_copy = class_list[:]
                    class_list_copy.append(answer)
                    class_list_copy.append(num_list)
                    class_list_copy.append(normal_num_list)
                    class_list_copy.append(None)
                    class_list_copy.append(prompt)
                    answer_list.append(class_list_copy)
                aug_csv = aug_csv.append(pd.DataFrame(answer_list, columns=columns)).drop_duplicates(subset='english')

                while True:
                    try:
                        aug_csv.to_csv(sub_csv_rpath, index=None)
                        output_df = pd.read_csv(csv_rpath) if os.path.exists(csv_rpath) else pd.DataFrame()
                        break
                    except:
                        continue
                process(csv_path, thread_num)
                aug_csv = aug_csv.append(output_df).drop_duplicates(subset='english')
                aug_csv_dict = aug_csv.set_index('prompt_class_num_normal')
            aug_csv.to_csv(sub_csv_rpath, index=None)
            process(csv_path, thread_num)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='ChatGPT DataAug')
    parser.add_argument('--thread_num', type=int)
    parser.add_argument('--thread_num_i', type=int, default=0)
    parser.add_argument('--output', type=str, default='aug_data.csv')
    parser.add_argument('--prompt', type=str, default='')
    parser.add_argument('--word_num', type=int, default=-1)
    parser.add_argument('--sp', default=False, action="store_true")
    parser.add_argument('--use_api', default=False, action="store_true")
    parser.add_argument('--turns', type=int)
    args = parser.parse_args()
    csv_path = args.output
    thread_num_i = args.thread_num_i
    prompt = args.prompt
    word_num = args.word_num
    thread_num = args.thread_num
    sp = args.sp
    use_api = args.use_api
    turns = args.turns

    gender = {'gender_male': set(), 'gender_female': set()}
    speed = {'speed_normal': set(), 'speed_slow': set(), 'speed_fast': set()}
    volume = {'volume_normal': set(), 'volume_low': set(), 'volume_high': set()}
    pitch = {'pitch_normal': set(), 'pitch_low': set(), 'pitch_high': set()}
    emotion = {'emotion_neutral': set(), 'emotion_happy': set(), 'emotion_angry': set(), 'emotion_sad': set()}
    age = {'age_adult': set(), 'age_old': set()}
    accent = {'accent_british': set(), 'accent_american': set()}
    categories = {'gender': gender, 'speed': speed, 'volume': volume, 'pitch': pitch, 'emotion': emotion, 'age': age, 'accent': accent}

    stage3(categories, str(), -1, False, 1, csv_path='prompt_wopl.csv', prompt=str(), thread_num_i=0, thread_num=1, use_api=True)
    stage3(categories, str(), -1, True, 1, csv_path='prompt_pl.csv', prompt=str(), thread_num_i=0, thread_num=1, use_api=True)


