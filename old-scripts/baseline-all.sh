#!/usr/bin/env bash
#$ -wd /home/sli136/silver-data-creation
#$ -V
#$ -N all-baseline
#$ -j y -o $JOB_NAME-$JOB_ID.out
#$ -M sli136@jhu.edu
#$ -m e
#$ -l ram_free=40G,mem_free=10G,gpu=2,hostname=octopod
#$ -q g.q
#$ -t 1
#$ -o /home/sli136/silver-data-creation/job-output

source /home/sli136/scripts/acquire_gpus 2

# Activate dev environments and call programs
conda activate l2mt
cd /home/sli136/silver-data-creation

xlmr_langs='es,fr,de,zh,ru,pt,it,ar,ja,id,tr,nl,pl,simple,fa,vi,sv,ko,he,ro,no,hi,uk,cs,fi,hu,th,da,ca,el,bg,sr,ms,bn,hr,sl,zh_yue,az,sk,eo,ta,sh,lt,et,ml,la,bs,sq,arz,af,ka,mr,eu,tl,ang,gl,nn,ur,kk,be,hy,te,lv,mk,zh_classical,als,is,wuu,my,sco,mn,ceb,ast,cy,kn,br,an,gu,bar,uz,lb,ne,si,war,jv,ga,zh_min_nan,oc,ku,sw,nds,ckb,ia,yi,fy,scn,gan,tt,am'

necessary_langs='ar,de,en,es,fr,hi,ru,vi,zh'

for src in ${xlmr_langs//,/ }; do
echo training baseline for ${src}
model_dir=/export/c11/sli136/silver-dataset/baseline-models/${src}/
mkdir -p ${model_dir}
python3 /home/sli136/silver-data-creation/baseline.py \
    -m xlm-roberta-base \
    -b 8 \
    -t $src \
    -w ${src}-baseline \
    -md ${model_dir} \
    -o ${model_dir}/${src}_results.json
done