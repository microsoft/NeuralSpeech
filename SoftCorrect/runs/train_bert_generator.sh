# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

EXP_HOME=$(cd `dirname $0`; pwd)/..
cd $EXP_HOME

DATA_PATH=${EXP_HOME}/pretrain_400M_tgtonly
SAVE_DIR=$EXP_HOME/checkpoints/bert_generator

export PYTHONPATH=${EXP_HOME}/sc_utils:$PYTHONPATH
export MKL_THREADING_LAYER=GNU
#export CUDA_VISIBLE_DEVICES=0
nvidia-smi
mkdir -p $SAVE_DIR
# train on 8 card
$(which fairseq-train) $DATA_PATH --task softcorrect_task \
    --arch softcorrect_corrector --lr 5e-4 --lr-scheduler inverse_sqrt \
    --length-loss-factor 0.5 \
    --noise full_mask \
    --dropout 0.3 --weight-decay 0.0001 \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
    --criterion softcorrect_loss --label-smoothing 0.0 \
    --max-tokens 4000 \
    --save-dir $SAVE_DIR \
    --mask-ratio 0.20 \
    --detector-mask-ratio 0.50 \
    --untouch-token-loss 0.2 \
    --error-distribution "0.1,0.4,0.4,0.0,0.1" \
    --homophone-dict-path "${EXP_HOME}/data/homophone.dict.txt" \
    --left-pad-target False \
    --left-pad-source False \
    --encoder-layers 12 --decoder-layers 0 \
    --user-dir $EXP_HOME/softcorrect \
    --pad-first-dictionary \
    --max-epoch 30 --update-freq 2 --fp16 --num-workers 2 \
    --share-all-embeddings --encoder-embed-dim=512 --decoder-embed-dim=512 2>&1 | tee -a ${SAVE_DIR}/train.log

