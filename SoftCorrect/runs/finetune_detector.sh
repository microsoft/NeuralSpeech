# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

EXP_HOME=$(cd `dirname $0`; pwd)/..
cd $EXP_HOME

DATA_PATH=${EXP_HOME}/pretrain_400M_tgtonly
SAVE_DIR=${EXP_HOME}/checkpoints/detector_finetune

export PYTHONPATH=${EXP_HOME}/sc_utils:$PYTHONPATH
export MKL_THREADING_LAYER=GNU
#export CUDA_VISIBLE_DEVICES=0
nvidia-smi
mkdir -p $SAVE_DIR
# train on 8 card
python ${EXP_HOME}/sc_utils/train_sc.py $DATA_PATH --task softcorrect_task \
    --arch softcorrect_detector --lr 5e-4 --lr-scheduler inverse_sqrt \
    --length-loss-factor 0.5 \
    --noise full_mask \
    --dropout 0.3 --weight-decay 0.0001 \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
    --criterion softcorrect_loss --label-smoothing 0.0 \
    --max-tokens 10000 \
    --save-dir $SAVE_DIR \
    --mask-ratio 0.25 \
    --detector-mask-ratio 0.0 \
    --untouch-token-loss 0.2 \
    --error-distribution "0.0,0.598,0.299,0.0,0.1,0.003" \
    --homophone-dict-path "${EXP_HOME}/data/homophone.dict.txt" \
    --left-pad-target False \
    --left-pad-source False \
    --encoder-layers 12 --decoder-layers 0 \
    --label-leak-prob 0.30 --candidate-size -1 \
    --nbest-void-insert-ratio 0.005 --nbest-input-num 4 --nbest-input-sample-temp 1.2 --nbest-input-sample-untouch-temp 0.0 \
    --encoder-training-type "detector" \
    --emb-dropout 0.0 \
    --force-same-ratio 0.3 \
    --same-also-sample 1.2 \
    --user-dir $EXP_HOME/softcorrect \
    --layernorm-embedding \
    --pad-first-dictionary \
    --bert-generator-encoder-model-path "${EXP_HOME}/checkpoints/bert_generator/checkpoint30.pt" \
    --main-encoder-warmup-path "${EXP_HOME}/checkpoints/detector_pretrain/checkpoint5.pt" \
    --max-epoch 30 --update-freq 4 --fp16 --num-workers 2 \
    --share-all-embeddings --encoder-embed-dim=512 --decoder-embed-dim=512 2>&1 | tee -a $SAVE_DIR/train.log
