#!/usr/bin/env bash
#$ -wd /home/sli136/silver-data-creation
#$ -V
#$ -N sil-mark-ne
#$ -j y -o $JOB_NAME-$JOB_ID.out
#$ -M sli136@jhu.edu
#$ -m e
#$ -l ram_free=40G,hostname=octopod|c*|b*
#$ -q all.q
#$ -t 1
#$ -o /home/sli136/silver-data-creation/job-output/trans

# source /home/sli136/scripts/acquire_gpu

# Activate dev environments and call programs
conda activate l2mt
cd /home/sli136/silver-data-creation

phrase=.phrase  # if token level masking, phrase=''

all_src=en,ar,de,es,fr,hi,ru,vi,zh,bo,ur,gu,fo,uk,km,sr,fa,ku,tg,th,bh,pl,ro,bg,zh-classical,cs,it,sa,nn,bn,af,sd,or,pt,pa,zh-yue,mr,ne,be,sl,da,bs,sk,hr,ca,zh-min-nan,nl,sv,ja,as,ps,no,mt,he,am
group_src=ar,de,es,fr,hi,ru,vi,zh                 # group 1
group_src=bo,ur,gu,fo,uk,km,sr,fa,ku              # group 2
group_src=tg,th,bh,pl,ro,bg,zh-classical,cs,it,sa # group 3
group_src=nn,bn,af,sd,or,pt,pa,zh-yue,mr          # group 4
group_src=ne,be,sl,da,bs,sk,hr,ca,zh-min-nan      # group 5
group_src=nl,sv,ja,as,ps,no,mt,he,am              # group 6

src=ne
# for src in ${group_src//,/ }; do
echo ========marking source lang for ${src}========
save_dir=/export/c11/sli136/silver-dataset/translations/marked/${src}.marked${phrase}
# if [ -d ${save_dir} ]; then
#     echo "${save_dir} already exists, skipping"
#     continue
# fi

if [ -d ${save_dir} ]; then
    rm -r ${save_dir}
fi

[ ! -d ${save_dir} ] && mkdir -p ${save_dir}

echo saving marked sentences to ${save_dir}
# trans_dir=/export/c11/sli136/silver-dataset/translations/${src}-${tgt}/
# mkdir -p ${trans_dir}
python3 /home/sli136/silver-data-creation/run.py \
    -b 16 \
    -s ${src} \
    --mark_only --phrase \
    -td /export/c11/sli136/silver-dataset/translations
# done