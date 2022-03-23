# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os

os.environ["OMP_NUM_THREADS"] = "1"

import json
import os
import re
import subprocess
from multiprocessing.pool import Pool
import pandas as pd
from g2p_en import G2p
from tqdm import tqdm

basedir = 'data/raw/LJSpeech-1.1'

g2p = G2p()


def g2p_job(idx, fn, txt):
    spk = idx // 100
    phs = [p.replace(" ", "|") for p in g2p(txt)]
    phs_str = " ".join(phs)
    os.makedirs(f'{basedir}/mfa_input/{spk}', exist_ok=True)
    with open(f'{basedir}/mfa_input/{spk}/{fn}.ph', 'w') as f_txt:
        phs_str = re.sub("([\!\'\,\-\.\?])+", r"\1", phs_str)
        f_txt.write(phs_str)
    with open(f'{basedir}/mfa_input/{spk}/{fn}.lab', 'w') as f_txt:
        phs_str_mfa = re.sub("[\!\'\,\-\.\?]+", "PUNC", phs_str)
        phs_str_mfa = re.sub("\|", "SEP", phs_str_mfa)
        f_txt.write(phs_str_mfa)
    subprocess.check_call(f'cp "{basedir}/wavs/{fn}.wav" "{basedir}/mfa_input/{spk}/"', shell=True)
    return phs_str.split(" "), phs_str_mfa.split(" ")


if __name__ == "__main__":
    # build mfa_input for forced alignment
    p = Pool(os.cpu_count())
    subprocess.check_call(f'rm -rf {basedir}/mfa_input', shell=True)
    futures = []

    f = open('ljspeech_text.txt', 'w')
    for idx, l in enumerate(open(f'{basedir}/metadata.csv').readlines()):
        fn, _, txt = l.strip().split("|")
        futures.append(p.apply_async(g2p_job, args=[idx, fn, txt]))
    p.close()
    mfa_dict = set()
    phone_set = set()
    for f in tqdm(futures):
        phs, phs_mfa = f.get()
        for ph in phs:
            phone_set.add(ph)
        for ph_mfa in phs_mfa:
            mfa_dict.add(ph_mfa)
    mfa_dict = sorted(mfa_dict)
    phone_set = sorted(phone_set)
    print("| mfa dict: ", mfa_dict)
    print("| phone set: ", phone_set)
    with open(f'{basedir}/dict_mfa.txt', 'w') as f:
        for ph in mfa_dict:
            f.write(f'{ph} {ph}\n')
    with open(f'{basedir}/dict.txt', 'w') as f:
        for ph in phone_set:
            f.write(f'{ph} {ph}\n')
    phone_set = ["<pad>", "<EOS>", "<UNK>"] + phone_set
    json.dump(phone_set, open(f'{basedir}/phone_set.json', 'w'))
    p.join()

    # build metadata_phone
    meta_ori_df = pd.read_csv(os.path.join(basedir, 'metadata.csv'), delimiter='|', names=['wav', 'txt1', 'txt2'])
    subprocess.check_call(f"mkdir -p {basedir}/mfa_input_txt; "
                          f"cp {basedir}/mfa_input/*/*.ph {basedir}/mfa_input_txt",
                          shell=True)

    meta_ori_df['phone2'] = meta_ori_df.apply(
        lambda r: open(f"{basedir}/mfa_input_txt/{r['wav']}.ph").readlines()[0].strip(), 1)
    meta_ori_df.to_csv(f"{basedir}/metadata_phone.csv")
