#!/usr/bin/env bash
#$ -wd /home/sli136/silver-data-creation/
#$ -N es-exp8
#$ -j y -o $JOB_NAME-$JOB_ID.out
#$ -M sli136@jhu.edu
#$ -m e
#$ -l ram_free=20G,mem_free=10G,gpu=1,hostname=octopod|c*
#$ -q g.q
#$ -t 1
#$ -o /home/sli136/silver-data-creation/job-output/phrase/es/

source /home/sli136/scripts/acquire_gpu

# Activate dev environments and call programs
conda activate l2mt
cd /home/sli136/silver-data-creation/

necessary_langs='en,ar,de,es,fr,hi,ru,vi,zh'

tgt=${1}
exp_num=${2}

# tgt=ar
if [ ${tgt} == "ar" ]; then
same_script=he,fa,ur,ku,ps,tg
sim_langs=am,mt,he,fa,ps,ku,ur
fi

# tgt=de
if [ ${tgt} == "de" ]; then
same_script=en,nl,nn
sim_langs=en,nl,da,sv,nn,no,fo,af
fi

# tgt=es
if [ ${tgt} == "es" ]; then
same_script=pt,ca,it
sim_langs=pt,ca,it,fr,ro
fi

# tgt=fr
if [ ${tgt} == "fr" ]; then
same_script=it,ca,es
sim_langs=pt,ca,it,es,ro
fi

# tgt=hi
if [ ${tgt} == "hi" ]; then
same_script=mr,sa,ne,sd
sim_langs=ur,ne,bn,pa,mr,gu,sd,sa,as,or,bh,fa,ps,ku,tg,en
fi 

# tgt=ru
if [ ${tgt} == "ru" ]; then
same_script=be,bg,kk,ky,mk,sr,tg,tk,uk,uz
sim_langs=uk,be,pl,cs,sk,bg,sr,hr,bs,sl
fi

# tgt=vi
if [ ${tgt} == "vi" ]; then
same_script=yo,ig,gn,it,sl,cy
sim_langs=km,zh,th,en,zh-yue
fi

# tgt=zh
if [ ${tgt} == "zh" ]; then
same_script=zh-yue,zh-classical,zh-min-nan,ja,bo
sim_langs=zh-yue,ja,bo,vi,en
fi

if [ ${exp_num} == "1" ]; then
    inputs=1:en:none:none   # exp_num:s1_oro_langs:s2_oro_langs:s2_silver_langs
fi
if [ ${exp_num} == "2" ]; then
    inputs=2:ru:none:none
fi
if [ ${exp_num} == "3" ]; then
    inputs=3:none:none:en
fi
if [ ${exp_num} == "4" ]; then
    inputs=4:none:en:en
fi
if [ ${exp_num} == "5" ]; then
    inputs=5:en:none:en
fi
if [ ${exp_num} == "6" ]; then
    inputs=6:en:en:en
fi
if [ ${exp_num} == "7" ]; then
    inputs=7:none:none:${necessary_langs}
fi
if [ ${exp_num} == "8" ]; then
    inputs=8:none:en:${necessary_langs}
fi
if [ ${exp_num} == "9" ]; then
    inputs=9:en:none:${necessary_langs}
fi
if [ ${exp_num} == "10" ]; then
    inputs=10:en:en:${necessary_langs}
fi
if [ ${exp_num} == "11" ]; then
    inputs=11:none:none:${sim_langs}
fi
if [ ${exp_num} == "12" ]; then
    inputs=12:none:en:${sim_langs}
fi
if [ ${exp_num} == "13" ]; then
    inputs=13:en:none:${sim_langs}
fi
if [ ${exp_num} == "14" ]; then
    inputs=14:en:en:${sim_langs}
fi
if [ ${exp_num} == "15" ]; then
    inputs=15:none:${same_script}:${sim_langs}
fi
if [ ${exp_num} == "16" ]; then
    inputs=16:en:${same_script}:${sim_langs}
fi
if [ ${exp_num} == "17" ]; then
    inputs=17:${same_script}:none:${sim_langs}
fi
if [ ${exp_num} == "18" ]; then
    inputs=18:${same_script}:${same_script}:${sim_langs}
fi

exp_num=$(echo ${inputs} | cut -d':' -f1)
s1_oro_langs=$(echo ${inputs} | cut -d':' -f2)
s2_oro_langs=$(echo ${inputs} | cut -d':' -f3)
s2_silver_langs=$(echo ${inputs} | cut -d':' -f4)

if [ ${s1_oro_langs} == "none" ]; then
    model_init_dir=xlm-roberta-base
    model_dir_s1_key="none"
fi
if [ ${s1_oro_langs} == "en" ]; then
    model_init_dir=/export/c11/sli136/silver-dataset/models/en-ft/checkpoint-last
    s1_oro_langs="none"
    model_dir_s1_key="en"
fi
if [ ${s1_oro_langs} == ${same_script} ]; then
    if [ -d "/export/c11/sli136/silver-dataset/models/${tgt}/s1-script/runs" ]; then
        model_init_dir=/export/c11/sli136/silver-dataset/models/${tgt}/s1-script/checkpoint-50000
        s1_oro_langs="none"
    else
        model_init_dir=xlm-roberta-base
    fi
    model_dir_s1_key="script"
fi

echo running combined fine-tune for ${tgt} - Exp \#${exp_num}
echo "inputs: ${inputs}"
echo s1 oro langs: ${s1_oro_langs}
echo s2 oro langs: ${s2_oro_langs}
echo s2 silver langs: ${s2_silver_langs}
echo model init dir: ${model_init_dir}
model_dir_s1=/export/c11/sli136/silver-dataset/models/${tgt}/s1-${model_dir_s1_key}
model_dir_s2=/export/c11/sli136/silver-dataset/models/${tgt}.phrase/${tgt}-exp${exp_num}
[ ! -d ${model_dir_s1} ] && mkdir -p ${model_dir_s1}
[ ! -d ${model_dir_s2} ] && mkdir -p ${model_dir_s2}

CUDA_LAUNCH_BLOCKING=1 python3 /home/sli136/silver-data-creation/run.py \
    --batch_size 4 \
    --phrase \
    --stage1_oro_langs ${s1_oro_langs} --stage1_oro_size 20000 \
    --stage2_oro_langs ${s2_oro_langs} --stage2_oro_size 20000 \
    --stage2_silver_langs ${s2_silver_langs} --stage2_silver_size -1 \
    --tgt_lang ${tgt} \
    --eval_output test_results.json \
    --model_save_dir_s1 ${model_dir_s1} \
    --model_save_dir_s2 ${model_dir_s2} \
    --model_init_dir ${model_init_dir} \
    --silver_dir /export/c11/sli136/silver-dataset/translations

# CUDA_LAUNCH_BLOCKING=1 python3 /home/sli136/silver-data-creation/run.py \
#     --tgt_lang ${tgt} \
#     --model_init_dir /export/c11/sli136/silver-dataset/models/de/de-exp12/checkpoint-47500 \
#     --predict_only \
#     --eval_output test_results.json 