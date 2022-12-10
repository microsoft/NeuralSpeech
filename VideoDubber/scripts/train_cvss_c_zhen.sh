nvidia-smi
export CUDA_VISIBLE_DEVICES=0

HOME=fairseq

MODEL=transformer_speech_length
PROBLEM=cvss_c_zhen
DATA=nmt_data/cvss_c_zhen/data-bin
MODEL_DIR=nmt_models/$PROBLEM-$MODEL
TASK=translation_speech_length
# set1 with initialization
export PYTHONPATH=$HOME:$PYTHONPATH

rm -rf $MODEL_DIR
mkdir -p $MODEL_DIR

python $HOME/train.py \
    $DATA \
    --task $TASK \
    --arch $MODEL --share-decoder-input-output-embed \
    --represent-length-by-lrpe --quant_N 5 --ordinary-sinpos --dur-sinpos --use-dur-predictor \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
    --lr 5e-4 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
    --dropout 0.3 --weight-decay 0.0001 --length-loss-factor 0.05 --restore-file $MODEL_DIR/checkpoint_last.pt \
    --reset-dataloader --reset-lr-scheduler --reset-meters --reset-optimizer \
    --criterion label_smoothed_cross_entropy_speech_length --label-smoothing 0.1 \
    --max-tokens 8192 --no-progress-bar --log-format 'simple' --log-interval 100 --skip-invalid-size-inputs-valid-test \
    --max-update 100000 --keep-interval-updates 5 --keep-last-epochs 5 --keep-best-checkpoints 5 --save-dir $MODEL_DIR 
    
