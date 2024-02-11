# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import json
import openai
import argparse
import pandas as pd

from prompt_pl import stage1, stage2, stage3
from fusion import fusion

def input_line(question, config_dict, config_key, sep=',', type=list):
    if config_key not in config_dict.keys():
        config_list = input(question).strip().split(sep)
        if not isinstance(config_list, type):
            config_dict[config_key] = type(config_list[0])
        else:
            config_dict[config_key] = config_list
    return config_dict[config_key], config_dict

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Prompt Generation')
    parser.add_argument('--config_path', type=str, default=str())
    args = parser.parse_args()

    config_path = args.config_path
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
    else:
        config_dict = dict()

    api_key, config_dict = input_line(f"Please enter api key of openai: ", config_dict, "api_key", sep=None, type=str)
    openai.api_key = api_key

    categories, config_dict = input_line(f"Please enter all attributes separated by commas: ", config_dict, "attributes")
    categories = {category: dict() for category in categories}
    for attribute in config_dict["attributes"]:
        attribute_classes, config_dict = input_line(f"Please enter the class of attribute {attribute} separated by commas and the first one is the default class: ", config_dict, attribute)
        categories[attribute] = {f'{attribute}_{attribute_class}': set() for attribute_class in attribute_classes}

    template_num, config_dict = input_line(f"Please enter the minimum number of sentences for each attribute: ", config_dict, "template_num", sep=None, type=int)
    keywords_num, config_dict = input_line(f"Please enter the minimum number of keywords for each class: ", config_dict, "keywords_num", sep=None, type=int)
    turns, config_dict = input_line(f"Please enter how many turns you want to generate: ", config_dict,  "turns", sep=None, type=int)
    use_placeholder, config_dict = input_line(f"Please enter whether you want to use placeholder [yes/no]: ", config_dict, "use_placeholder", sep=None, type=str)
    if use_placeholder == 'yes':
        use_placeholder = True
        csv_path = 'process_pl.csv'
        use_placeholder_num, config_dict = input_line(f"Please enter the minimum number of sentences using placeholder for each class: ", config_dict, "use_placeholder_num", sep=None, type=int)
    elif use_placeholder == 'no':
        use_placeholder = False
        csv_path = 'process_npl.csv'
    else:
        raise ValueError("You must choose from yes and no.")
    sentences_type, config_dict = input_line(f"Please enter what the type of sentences you want to generate [sentence/phrase/word]: ", config_dict, "sentences_type", sep=None, type=str)
    if sentences_type == 'sentence':
        sentences_type = -1
    elif sentences_type == 'phrase':
        sentences_type = -2
    elif sentences_type == 'word':
        sentences_type = -3
    else:
        raise ValueError("You must choose from sentence, phrase and word.") # [mulword] 0[mulsentence] -1[sentenc] -2[phrase] -3[word]

    if os.path.exists(config_path):
        with open(config_path, 'w') as f:
            json.dump(config_dict, f, indent=4)
        print("The configuration is saved to: ", config_path)
    else:
        with open('config.json', 'w') as f:
            json.dump(config_dict, f, indent=4)
        print("The configuration is saved to: ", 'config.json')

    print("Stage #1: Keyword Generation")
    stage1(categories, keywords_num)
    print("Stage #2: Sentence Generation")
    stage2(categories, template_num)
    print("Stage #3: Sentence Combination")
    stage3(categories, api_key, sentences_type, use_placeholder, turns, csv_path=csv_path)
    print("Stage #4: Dataset Construction")
    if use_placeholder:
        output_csv_path = 'output_pl.csv'
        fusion(categories, {csv_path:fused_num}, output_csv_path)
    else:
        output_csv_path = 'output_npl.csv'
        pd.read_csv(csv_path).to_csv(output_csv_path, index=None)
    print("Output dataset: ", output_csv_path)




