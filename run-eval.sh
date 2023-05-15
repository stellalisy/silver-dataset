#!/usr/bin/env bash
#$ -wd /home/sli136/silver-data-creation
#$ -V
#$ -N vi-exp0-eval
#$ -j y -o $JOB_NAME-$JOB_ID.out
#$ -M sli136@jhu.edu
#$ -m e
#$ -l ram_free=40G,mem_free=10G,gpu=1,hostname=octopod|c*
#$ -q g.q
#$ -t 1
#$ -o /home/sli136/silver-data-creation/job-output/new/ar/

source /home/sli136/scripts/acquire_gpu
conda activate l2mt
cd /home/sli136/silver-data-creation

tgt=${1}
exp_num=${2}
model_dir=/export/c11/sli136/silver-dataset/models/en-ft/checkpoint-last

echo "Evaluating ${tgt} experiment ${exp_num} on ${model_dir}"
CUDA_LAUNCH_BLOCKING=1 python3 /home/sli136/silver-data-creation/run.py \
    --tgt_lang ${tgt} \
    --model_init_dir ${model_dir} \
    --predict_only \
    --eval_output test_results_v1_baseline.json 