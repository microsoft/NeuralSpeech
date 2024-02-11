# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

# Due to the iteration of ChatGPT, you may need to make some adjustments to the inquiry sentences.
sentence_num = 5
# Stage one
prefix_keyword = f"Can you list [keywords_num] words or phrases that is the synonyms for [sub_category] in lower case? Please generate one word per line and identify it with a serial number."
# Stage two
prefix_template = [
    f"Please write {sentence_num} templates for me to describe the [category] of a speech. The templates should use [placeholder] to indicate where a specific [category] would be inserted. This template should be simple enough to only have 10 words.",
    f"Please generate {sentence_num} templates to to ask for generating a voice. These templates can only describe the [category] of the voice and use [placeholder] to indicate where a word to describe [category] would be inserted. This template should be simple enough to only have a few words."
]
# Stage three
prefix_sentence_sentences = f"I have some sentences, can you combine these sentences into one sentence to describe the style of speech with the same meaning? You can generate {sentence_num} different sentences and callout number. The sentences are: "
prefix_sentence_phrases = f"I have some sentences, can you combine these sentences into one phrase to describe the style of speech with the same meaning? You can generate {sentence_num} different phrases and callout number. Please reduce to phrases. The sentences are: "
prefix_sentence_words = f"I have some sentences, can you combine these sentences into one phrase to describe the style of speech with the same meaning? You can generate {sentence_num} different phrases and callout number. Please reduce to words. The sentences are: "

prefix_sp_sentences = f"I have some sentences, can you combine these sentences into one sentence and keep all of the '[]' to describe the style of speech with the same meaning? You can generate {sentence_num} different sentences and callout number. The sentences are: "
prefix_sp_phrases = f"I have some sentences, can you combine these sentences into one phrase and keep all of the '[]' to describe the style of speech with the same meaning? You can generate {sentence_num} different phrases and callout number. Please reduce to phrases. The sentences are: "
prefix_sp_words = f"I have some sentences, can you combine these sentences into one phrase and keep all of the '[]' to describe the style of speech with the same meaning? You can generate {sentence_num} different phrases and callout number. Please reduce to words. The sentences are: "
