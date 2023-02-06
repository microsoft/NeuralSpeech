# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

EXP_HOME=$(cd `dirname $0`; pwd)/..
cd $EXP_HOME

SAVE_DIR=checkpoints/detector_finetune
#export PYTHONPATH=${EXP_HOME}/sc_utils:$PYTHONPATH
export PYTHONPATH=${EXP_HOME}/sc_utils:${EXP_HOME}/softcorrect:$PYTHONPATH
mkdir -p ${SAVE_DIR}/log_aishell_trans
eval_data=eval_data/aishell_nbest_eval_detector
test_epoch=16

edit_thre=-1.0
export CUDA_VISIBLE_DEVICES=0
nohup python eval_detector.py "dev" ${SAVE_DIR} "${eval_data}"  ${test_epoch} >> ${SAVE_DIR}/log_aishell_trans/nohup.b0e${test_epoch}test00.log 2>&1 &
export CUDA_VISIBLE_DEVICES=1
nohup python eval_detector.py "test" ${SAVE_DIR} "${eval_data}" ${test_epoch} >> ${SAVE_DIR}/log_aishell_trans/nohup.b0e${test_epoch}test11.log 2>&1 &
wait
