# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

EXP_HOME=$(cd `dirname $0`; pwd)/..
cd $EXP_HOME


SAVE_DIR=    #<PATH-to-AISHELL1-Save-Dir>
export PYTHONPATH=$EXP_HOME/FC_utils:$PYTHONPATH

test_epochs=45  # Should be decided by dev set performance

for test_epoch in ${test_epochs}; do
echo "test epoch: $test_epoch"

mkdir -p ${SAVE_DIR}/log_aishell_e${test_epoch}
edit_thre=-1.0

export CUDA_VISIBLE_DEVICES=0
nohup python -u eval_aishell.py "dev" "" ${SAVE_DIR} 0 0 ${test_epoch} >> ${SAVE_DIR}/log_aishell_e${test_epoch}/nohup.b0ttest00.log 2>&1 &
export CUDA_VISIBLE_DEVICES=1
nohup python -u eval_aishell.py "test" "" ${SAVE_DIR} 0 0 ${test_epoch} >> ${SAVE_DIR}/log_aishell_e${test_epoch}/nohup.b0ttest01.log 2>&1 &
wait
done
