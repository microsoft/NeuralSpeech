# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

EXP_HOME=$(cd `dirname $0`; pwd)/..
cd $EXP_HOME

DATA_PATH=$EXP_HOME/data/aishell_corrector
SAVE_DIR=$EXP_HOME/checkpoints/corrector_finetune
PRETRAINED=$EXP_HOME/checkpoints/corrector_pretrain/checkpoint6.pt

export PYTHONPATH=${EXP_HOME}/sc_utils:$PYTHONPATH
export MKL_THREADING_LAYER=GNU
#Trained on 4 cards
nvidia-smi
mkdir -p $SAVE_DIR
$(which fairseq-train) $DATA_PATH --task softcorrect_task \
    --arch softcorrect_corrector --lr 5e-4 --lr-scheduler inverse_sqrt \
    --length-loss-factor 0.5 \
    --noise full_mask \
    --src-with-werdur \
    --dropout 0.3 --weight-decay 0.0001 \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
    --criterion softcorrect_loss --label-smoothing 0.1 \
    --max-tokens 5000 \
    --save-dir $SAVE_DIR \
    --mask-ratio 0.0 \
    --untouch-token-loss 8.0 \
    --ft-error-distribution "0.05,0.5" \
    --duptoken-error-distribution "0.8,0.2" \
    --homophone-dict-path "${EXP_HOME}/data/homophone.dict.txt" \
    --left-pad-target False \
    --left-pad-source False \
    --encoder-layers 6 --decoder-layers 0 \
    --max-epoch 50 --update-freq 4 --num-workers 4 \
    --restore-file $PRETRAINED --reset-optimizer \
    --user-dir $EXP_HOME/softcorrect \
    --pad-first-dictionary \
    --share-all-embeddings --encoder-embed-dim=512 --decoder-embed-dim=512 2>&1 | tee -a $SAVE_DIR/train.log
