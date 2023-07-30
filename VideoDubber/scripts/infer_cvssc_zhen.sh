nvidia-smi
export CUDA_VISIBLE_DEVICES=0

HOME=fairseq

MODEL=videodubber
PROBLEM=zhen
DATA=nmt_data/data-bin
MODEL_DIR=nmt_models/zhen
TASK=translation_speech_length

SCRIPTS=mosesdecoder/scripts
DETC=${SCRIPTS}/recaser/detruecase.perl
MULTI_BLEU=${SCRIPTS}/generic/multi-bleu.perl
DETOKENIZER=${SCRIPTS}/tokenizer/detokenizer.perl

BEAM=5
BATCHSIZE=64

export PYTHONPATH=$HOME:$PYTHONPATH


mkdir -p $MODEL_DIR

TEST_RESULT_FILE=$MODEL_DIR/result/bestbeam$BEAM.txt
mkdir -p $MODEL_DIR/result
python $HOME/fairseq_cli/generate.py $DATA \
	  --task $TASK\
	  --path $MODEL_DIR/checkpoint_best.pt \
	  --batch-size $BATCHSIZE --beam $BEAM \
	  2>&1 \
	| tee $TEST_RESULT_FILE

grep ^H $TEST_RESULT_FILE | cut -f3- > $MODEL_DIR/result/predict.tok.true.bpe.en
grep ^T $TEST_RESULT_FILE | cut -f2- > $MODEL_DIR/result/answer.tok.true.bpe.en

sed -r 's/(@@ )| (@@ ?$)//g' < $MODEL_DIR/result/predict.tok.true.bpe.en  > $MODEL_DIR/result/predict.tok.true.en
sed -r 's/(@@ )| (@@ ?$)//g' < $MODEL_DIR/result/answer.tok.true.bpe.en  > $MODEL_DIR/result/answer.tok.true.en

${DETC} < $MODEL_DIR/result/predict.tok.true.en > $MODEL_DIR/result/predict.tok.en
${DETC} < $MODEL_DIR/result/answer.tok.true.en > $MODEL_DIR/result/answer.tok.en
${MULTI_BLEU} -lc $MODEL_DIR/result/answer.tok.en < $MODEL_DIR/result/predict.tok.en

# de <brk>
sed -r 's/(<brk> )| (<brk> ?$)//g' < $MODEL_DIR/result/predict.tok.en  > $MODEL_DIR/result/predict.tok.debrk.en
sed -r 's/(<brk> )| (<brk> ?$)//g' < $MODEL_DIR/result/answer.tok.en  > $MODEL_DIR/result/answer.tok.debrk.en
${MULTI_BLEU} -lc $MODEL_DIR/result/answer.tok.debrk.en < $MODEL_DIR/result/predict.tok.debrk.en
