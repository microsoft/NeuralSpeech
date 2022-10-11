# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

EXP_HOME=$(cd `dirname $0`; pwd)/..
cd $EXP_HOME

DATA_DIR=$EXP_HOME/data/trainset
SAVE_DIR=$EXP_HOME/checkpoints/stage_two
mkdir -p $SAVE_DIR
export MKL_THREADING_LAYER=GNU
export PYTHONPATH=${EXP_HOME}/src:$PYTHONPATH

python src/binauralgrad/train.py $SAVE_DIR $DATA_DIR --binaural-type leftright --params params_stage_two
