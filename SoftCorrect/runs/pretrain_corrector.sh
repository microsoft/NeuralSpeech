# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

EXP_HOME=$(cd `dirname $0`; pwd)/..
cd $EXP_HOME

DATA_PATH=${EXP_HOME}/pretrain_400M_tgtonly
SAVE_DIR=${EXP_HOME}/checkpoints/pretrained_corrector

export PYTHONPATH=${EXP_HOME}/sc_utils:$PYTHONPATH
export MKL_THREADING_LAYER=GNU
nvidia-smi
mkdir -p $SAVE_DIR

# Trained on 8 32G V100
$(which fairseq-train) $DATA_PATH --task softcorrect_task \
    --arch softcorrect_corrector --lr 5e-4 --lr-scheduler inverse_sqrt \
    --length-loss-factor 0.5 \
    --noise full_mask \
    --dropout 0.3 --weight-decay 0.0001 \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
    --criterion softcorrect_loss --label-smoothing 0.0 \
    --max-tokens 5000 \
    --save-dir $SAVE_DIR \
    --mask-ratio 0.20 \
    --untouch-token-loss 0.2 \
    --error-distribution "0.1,0.3,0.3,0.3" \
    --duptoken-error-distribution "0.8,0.2" \
    --homophone-dict-path "${EXP_HOME}/data/homophone.dict.txt" \
    --left-pad-target False \
    --left-pad-source False \
    --encoder-layers 6 --decoder-layers 0 \
    --user-dir $EXP_HOME/softcorrect \
    --pad-first-dictionary \
    --max-epoch 50 --update-freq 2 --num-workers 2 \
    --share-all-embeddings --encoder-embed-dim=512 --decoder-embed-dim=512 2>&1 | tee -a $SAVE_DIR/train.log

