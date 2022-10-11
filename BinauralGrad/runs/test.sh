# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

EXP_HOME=$(cd `dirname $0`; pwd)/..
cd $EXP_HOME

TEST_DATA_DIR=$EXP_HOME/data/testset
STAGE_ONE_SAVE_DIR=$EXP_HOME/checkpoints/stage_one
STAGE_TWO_SAVE_DIR=$EXP_HOME/checkpoints/stage_two
mkdir -p ${STAGE_ONE_SAVE_DIR} ${STAGE_TWO_SAVE_DIR}
export MKL_THREADING_LAYER=GNU

export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=${EXP_HOME}/src:$PYTHONPATH
#Inference stage one model
OUTPUT_DIR_S1=${STAGE_ONE_SAVE_DIR}/output_s1
mkdir -p ${OUTPUT_DIR_S1}
for i in `seq 1 8`; do
  python src/binauralgrad/inference.py --fast ${STAGE_ONE_SAVE_DIR}/pretrained_ckpt.s1.pt \
    --dsp_path ${TEST_DATA_DIR}/subject${i} \
    --binaural_type leftright \
    -o ${OUTPUT_DIR_S1}/subject${i}.wav \
    --params params_stage_one
done

python src/binauralgrad/inference.py --fast ${STAGE_ONE_SAVE_DIR}/pretrained_ckpt.s1.pt \
  --dsp_path ${TEST_DATA_DIR}/validation_sequence \
  --binaural_type leftright \
  -o ${OUTPUT_DIR_S1}/validation_sequence.wav \
  --params params_stage_one

#Inference stage two model based on the results of stage one.
OUTPUT_DIR_S2=${STAGE_TWO_SAVE_DIR}/output_s2
mkdir -p ${OUTPUT_DIR_S2}

for i in `seq 1 8`; do
  python src/binauralgrad/inference.py --fast ${STAGE_TWO_SAVE_DIR}/pretrained_ckpt.s2.pt \
    --dsp_path ${TEST_DATA_DIR}/subject${i} \
    --binaural_type leftright \
    -o ${OUTPUT_DIR_S2}/subject${i}.wav \
    --mean-condition-folder ${OUTPUT_DIR_S1} \
    --params params_stage_two
done

python src/binauralgrad/inference.py --fast ${STAGE_TWO_SAVE_DIR}/pretrained_ckpt.s2.pt \
  --dsp_path ${TEST_DATA_DIR}/validation_sequence \
  --binaural_type leftright \
  -o ${OUTPUT_DIR_S2}/validation_sequence.wav \
  --mean-condition-folder ${OUTPUT_DIR_S1} \
  --params params_stage_two

# Calculate objective metric
python metric.py ${OUTPUT_DIR_S2} ${TEST_DATA_DIR}
