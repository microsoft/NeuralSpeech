# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

EXP_HOME=$(cd `dirname $0`; pwd)/..
cd $EXP_HOME


DATA_PATH=${EXP_HOME}/pretrain_400M_tgtonly
SAVE_DIR=$EXP_HOME/checkpoints/detector_pretrain

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
        --mask-ratio 0.20 \
        --detector-mask-ratio 0.1 \
        --candidate-size -1 \
        --untouch-token-loss 0.2 \
        --error-distribution "0.1,0.397,0.397,0.0,0.1,0.006" \
        --homophone-dict-path "${EXP_HOME}/data/homophone.dict.txt" \
        --left-pad-target False \
        --left-pad-source False \
        --encoder-layers 12 --decoder-layers 0 \
        --nbest-void-insert-ratio 0.01 --nbest-input-num 4 --nbest-input-sample-temp 2.0 \
        --encoder-training-type "bert" \
        --pad-first-dictionary \
        --user-dir $EXP_HOME/softcorrect \
        --bert-generator-encoder-model-path "${EXP_HOME}/checkpoints/bert_generator/checkpoint30.pt" \
        --max-epoch 30 --update-freq 2 --fp16 --num-workers 2 \
        --share-all-embeddings --encoder-embed-dim=512 --decoder-embed-dim=512 2>&1 | tee -a $SAVE_DIR/train.log

