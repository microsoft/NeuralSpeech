# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

EXP_HOME=$(cd `dirname $0`; pwd)/..

cd $EXP_HOME
export PYTHONPATH=${EXP_HOME}/src:$PYTHONPATH
DATA_DIR=${EXP_HOME}/data/trainset

echo "Geo warping for training set"
for i in `seq 1 8`; do
  python geowarp.py $DATA_DIR/subject${i}/mono.wav $DATA_DIR/subject${i}/tx_positions.txt $DATA_DIR/subject${i}/binaural_geowarp.wav
done

DATA_DIR=${EXP_HOME}/data/testset

echo "Geo warping for test set"
for i in `seq 1 8`; do
  python geowarp.py $DATA_DIR/subject${i}/mono.wav $DATA_DIR/subject${i}/tx_positions.txt $DATA_DIR/subject${i}/binaural_geowarp.wav
done

python geowarp.py $DATA_DIR/validation_sequence/mono.wav $DATA_DIR/validation_sequence/tx_positions.txt $DATA_DIR/validation_sequence/binaural_geowarp.wav
