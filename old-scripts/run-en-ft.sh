#!/usr/bin/env bash
#$ -wd /home/sli136/silver-data-creation
#$ -V
#$ -N en-baseline-all
#$ -j y -o $JOB_NAME-$JOB_ID.out
#$ -M sli136@jhu.edu
#$ -m e
#$ -l ram_free=40G,mem_free=10G,gpu=1,hostname=octopod|c2*
#$ -q g.q
#$ -t 1
#$ -o /home/sli136/silver-data-creation/job-output

source /home/sli136/scripts/acquire_gpu

# Activate dev environments and call programs
conda activate l2mt
cd /home/sli136/silver-data-creation

xlmr_langs='es,fr,de,zh,ru,pt,it,ar,ja,id,tr,nl,pl,simple,fa,vi,sv,ko,he,ro,no,hi,uk,cs,fi,hu,th,da,ca,el,bg,sr,ms,bn,hr,sl,zh_yue,az,sk,eo,ta,sh,lt,et,ml,la,bs,sq,arz,af,ka,mr,eu,tl,ang,gl,nn,ur,kk,be,hy,te,lv,mk,zh_classical,als,is,wuu,my,sco,mn,ceb,ast,cy,kn,br,an,gu,bar,uz,lb,ne,si,war,jv,ga,zh_min_nan,oc,ku,sw,nds,ckb,ia,yi,fy,scn,gan,tt,am'

necessary_langs='ar,de,es,fr,hi,ru,vi,zh'
src=en
for tgt in ${necessary_langs//,/ }; do
echo "running baseline (fine-tune on en) for "${tgt}
# trans_dir=/export/c11/sli136/silver-dataset/translations/${src}-${tgt}/
model_dir=/export/c11/sli136/silver-dataset/model/${tgt}-baseline
mkdir -p ${model_dir}
CUDA_LAUNCH_BLOCKING=1 python3 /home/sli136/silver-data-creation/run.py \
    -b 16 \
    -s ${src} \
    -t ${tgt} \
    -po \
    -md ${model_dir} \
    -r /export/c11/sli136/silver-dataset/model/en-ft/checkpoint-last \
    -o ${model_dir}/${src}_results.json
# break
done