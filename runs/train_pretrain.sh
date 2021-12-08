# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

EXP_HOME=$(cd `dirname $0`; pwd)/..
cd $EXP_HOME

cd FastCorrect
pip uninstall uuid
pip install --editable .
cd $EXP_HOME

DATA_PATH=<Path-to-pretrain-Binary-Data>
export PYTHONPATH=$EXP_HOME/FastCorrect:$PYTHONPATH

SAVE_DIR=<PATH-to-Pretrain-Save-Dir>
python $EXP_HOME/FastCorrect/fairseq_cli/train.py $DATA_PATH --task translation_lev \
        --arch nonautoregressive_transformer --lr 5e-4 --lr-scheduler inverse_sqrt \
        --length-loss-factor 0.5 \
        --noise full_mask \
        --src-with-werdur \
        --dur-predictor-type "v2" \
        --dropout 0.3 --weight-decay 0.0001 \
        --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
        --criterion nat_loss --label-smoothing 0.1 \
        --max-tokens 9000 \
        --werdur-max-predict 3 \
        --assist-edit-loss \
        --save-dir $SAVE_DIR \
        --left-pad-target False \
        --left-pad-source False \
        --encoder-layers 6 --decoder-layers 6 \
        --max-epoch 30 --update-freq 4 --fp16 --num-workers 8 \
        --share-all-embeddings --encoder-embed-dim=512 --decoder-embed-dim=512
