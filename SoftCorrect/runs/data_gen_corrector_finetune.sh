# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

# TEXT variable: saving raw data, containing train.zh_CN train.zh_CN_tgt valid.zh_CN valid.zh_CN_tgt
# DATA_PATH variable: saving the output data bin

EXP_HOME=$(cd `dirname $0`; pwd)/..
cd $EXP_HOME

DATA_PATH=data/aishell_corrector

export PYTHONPATH=${EXP_HOME}/sc_utils:$PYTHONPATH

TEXT=raw_data/aishell_corrector  # saving raw data, containing train.zh_CN train.zh_CN_tgt valid.zh_CN valid.zh_CN_tgt

python $EXP_HOME/sc_utils/preprocess_fc.py --source-lang zh_CN --target-lang zh_CN_tgt \
    --task translation \
    --trainpref $TEXT/train --validpref $TEXT/valid \
    --padding-factor 1 \
    --src-with-werdur \
    --destdir ${DATA_PATH} \
    --srcdict './data/dict.txt' --tgtdict './data/dict.txt' \
    --workers 20
