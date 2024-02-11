<h2 align="center">
<p>PromptGen
</h2>

<h3 align="center">
<p>Pipeline of Prompt Generation with ChatGPT
</h3>

 <!-- - [Introduction](#Introduction)
 - [Prepare ](#Prepare)  
 - [Run](#Run)  
 - [Detail](#Detail)   -->

## Introduction

 - In Stage 1, ChatGPT generates keywords for each style attribute. 
 - In Stage 2, ChatGPT composes sentences for each style attribute, incorporating placeholders for the corresponding attributes. 
 - In Stage 3, ChatGPT combines the sentences from Stage 2 to create a sentence describing multiple attributes simultaneously. 
 - In Stage 4, the dataset is utilized by first sampling a sentence and subsequently sampling keywords to replace the placeholders within the sentence.

## Prepare 

Clone or download this repository and install the dependencies. Just run:

```bash
pip install -r requirements.txt 
```

## Run

You can set up your run configuration and generate prompt by the answering the questions asked. Just run:

```bash
python main.py
```

from terminal to generate `config.json`. Here is an example:

```diff
  python main.py
  Please enter api key of openai: [api key]
  Please enter all attributes separated by commas: gender,speed,volume,pitch
  Please enter the class of attribute gender separated by commas and the first one is the default class: male,female
  Please enter the class of attribute speed separated by commas and the first one is the default class: normal,slow,fast
  Please enter the class of attribute volume separated by commas and the first one is the default class: normal,low,high
  Please enter the class of attribute pitch separated by commas and the first one is the default class: normal,low,high
  Please enter the minimum number of sentences for each attribute: 10
  Please enter the minimum number of keywords for each class: 10
  Please enter how many turns you want to generate: 1
  Please enter whether you want to use placeholder [yes/no]: yes
  Please enter the minimum number of sentences using placeholder for each class: 10
  Please enter what the type of sentences you want to generate [sentence/phrase/word]: sentence
```

As you can see in this example, You can apply for `[api key]` from [OPENAI](https://openai.com/). 

You can also use `config.json` to directly run. Just run:

```bash
python main.py --config_path config.json
```

Here is an example of `config.json`:

```json
{
    "api_key": "[api key]",
    "attributes": [
        "gender",
        "speed",
        "volume",
        "pitch"
    ],
    "gender": [
        "male",
        "female"
    ],
    "speed": [
        "normal",
        "slow",
        "fast"
    ],
    "volume": [
        "normal",
        "low",
        "high"
    ],
    "pitch": [
        "normal",
        "low",
        "high"
    ],
    "template_num": 10,
    "keywords_num": 10,
    "turns": 1,
    "use_placeholder": "yes",
    "use_placeholder_num": 10,
    "sentences_type": "sentence"
}
```

## Details and Datasets

 - The keywords is saved in `prompt_word` and the sentence is saved to `prompt_pl.csv`. We have provided some keywords and sentences, and if you need to regenerate them, you can delete them.
 - If placeholder is used, the output is `output_pl.csv` and if placeholder is not used, the output is `output_npl.csv`.
 - Due to the iteration of ChatGPT, you may need to make some adjustments to the inquiry sentences in `config.py`.

