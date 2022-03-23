# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import json
import os

from tts_utils.text_encoder import TokenTextEncoder


def build_phone_encoder(data_dir):
    phone_list_file = os.path.join(data_dir, 'phone_set.json')
    phone_list = json.load(open(phone_list_file))
    return TokenTextEncoder(None, vocab_list=phone_list)
