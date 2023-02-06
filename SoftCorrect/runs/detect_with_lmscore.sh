# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

exp_home=$(cd `dirname $0`; pwd)/..

cd $exp_home

mkdir -p detect_results; 
subsets="test dev";
epoch=16;
for subset in $subsets; do
  ckpt=detector_finetune
  python detect_error_token.py $exp_home/checkpoints/${ckpt}/results_detector_${epoch}_aishell/${subset}/data.json $exp_home/eval_data/aishell_nbest_eval_tokenscore/${subset}/data.json detect_results/detector_finetune_ep${epoch}/${subset} 2>&1 >> /dev/null &
done
