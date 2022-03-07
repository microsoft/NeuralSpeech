# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

EXP_HOME=$(cd `dirname $0`; pwd)/..
cd $EXP_HOME

cd $EXP_HOME

DATA_PATH=   #<Path-to-Binary-Data>
export PYTHONPATH=$EXP_HOME/FC_utils:$PYTHONPATH

TEXT=data/werdur_data_aishell   #<Path-to-Data-with-duration>
# contains train.zh_CN, train.zh_CN_tgt, valid.zh_CN, valid.zh_CN_tgt
# *.zh_CN is indeed the *.src.werdur.full in alignment result
# *.zh_CH_tgt is indeed the *.tgt in alignment result
dict_path=data/werdur_data_aishell/dict.CN_char.txt  #<path-to-dictionary>

#We use shared dictionary extracted from training corpus

$EXP_HOME/FC_utils/preprocess_fc.py --source-lang zh_CN --target-lang zh_CN_tgt \
    --task translation \
    --trainpref $TEXT/train --validpref $TEXT/valid \
    --padding-factor 8 \
    --src-with-nbest-werdur 4 \
    --destdir ${DATA_PATH} \
    --srcdict ${dict_path} --tgtdict ${dict_path} \
    --workers 20
