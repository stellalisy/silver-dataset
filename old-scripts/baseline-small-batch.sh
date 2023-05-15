#!/usr/bin/env bash
#$ -wd /home/sli136/silver-data-creation
#$ -V
#$ -N small-baseline
#$ -j y -o $JOB_NAME-$JOB_ID.out
#$ -M sli136@jhu.edu
#$ -m e
#$ -l ram_free=40G,mem_free=10G,gpu=1,hostname=octopod|c*
#$ -q g.q
#$ -t 1
#$ -o /home/sli136/silver-data-creation/job-output

source /home/sli136/scripts/acquire_gpu

# Activate dev environments and call programs
conda activate l2mt
cd /home/sli136/silver-data-creation

xlmr_langs='zh,ja,zh_yue'
missing='en'

for src in ${missing//,/ }; do
echo training baseline for ${src}
model_dir=/export/c11/sli136/silver-dataset/baseline-models/${src}/
mkdir -p ${model_dir}
python3 /home/sli136/silver-data-creation/baseline.py \
    -m xlm-roberta-base \
    -b 4 \
    -t $src \
    -w ${src}-baseline \
    -md ${model_dir} \
    -o ${model_dir}/${src}_results.json
done