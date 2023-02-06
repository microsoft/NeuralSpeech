# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

EXP_HOME=$(cd `dirname $0`; pwd)/..
cd $EXP_HOME

SAVE_DIR=checkpoints/corrector_finetune
export PYTHONPATH=${EXP_HOME}/sc_utils:$PYTHONPATH
mkdir -p ${SAVE_DIR}/log_aishell_trans

test_epoch=17
detector_thre=1.3
eval_data=detect_results/detector_finetune_ep16

export CUDA_VISIBLE_DEVICES=0
nohup python eval_corrector.py "dev" "Thre${detector_thre}" ${SAVE_DIR} "${eval_data}" ${test_epoch} -7.9 -6.2 >> ${SAVE_DIR}/log_aishell_trans/nohup.b0t${edit_thre}p-7.9h-6.2e${test_epoch}test00.log 2>&1 &
export CUDA_VISIBLE_DEVICES=1
nohup python eval_corrector.py "test" "Thre${detector_thre}" ${SAVE_DIR} "${eval_data}" ${test_epoch} -7.9 -6.2 >> ${SAVE_DIR}/log_aishell_trans/nohup.b0t${edit_thre}p-7.9h-6.2e${test_epoch}test11.log 2>&1 &
wait
