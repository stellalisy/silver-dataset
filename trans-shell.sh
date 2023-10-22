#!/usr/bin/env bash
#$ -wd /home/sli136/silver-data-creation
#$ -V
#$ -N trans-de
#$ -j y -o $JOB_NAME-$JOB_ID.out
#$ -M sli136@jhu.edu
#$ -m e
#$ -l ram_free=40G,mem_free=10G,hostname=octopod|c*|b*
#$ -t 1
#$ -o /home/sli136/silver-data-creation/job-output/trans

conda activate l2mt
cd /home/sli136/silver-data-creation

# split=${1}
# src=${2}
# tgt=${3}

src=${1}
tgt=${2}

# tgt=ar
# same_script=he,fa,ur,ku,ps,tg
# sim_langs=am,mt,he,fa,ps,ku,ur,an

# tgt=de
# same_script=en,nl,nn
# sim_langs=en,nl,da,sv,nn,no,fo,af

# tgt=es
# same_script=pt,ca,it
# sim_langs=pt,ca,it,fr,ro

# tgt=fr
# same_script=it,ca,es
# sim_langs=pt,ca,it,es,ro

# tgt=hi
# same_script=mr,sa,ne,sd
# sim_langs=ur,ne,bn,pa,mr,gu,sd,sa,as,or,bh,fa,ps,ku,tg,en

# tgt=ru
# same_script=be,bg,kk,ky,mk,sr,tg,tk,uk,uz
# sim_langs=uk,be,pl,cs,sk,bg,sr,hr,bs,sl

# tgt=vi
# same_script=yo,ig,gn,it,sl,cy
# sim_langs=km,zh,th,en

# tgt=zh
# same_script=zh,zh-yue,zh-classical,zh-min-nan,ja,bo
# sim_langs=zh-yue,zh-classical,zh-min-nan,ja,bo

splits=train,test,validation
phrase=.phrase  # if token level masking, 
phrase=''

for split in ${splits//,/ }; do
    echo running ${split} translation for ${src}-${tgt}
    source_dir='/export/c11/sli136/silver-dataset/translations/marked/'${src}'.marked'${phrase}
    source_file=${source_dir}/${src}.${split}.marked${phrase}

    trans_dir=/export/c11/sli136/silver-dataset/translations/${tgt}/${src}-${tgt}${phrase}
    trans_file=${trans_dir}/${src}-${tgt}.${split}.trans

    [ ! -d ${trans_dir} ] && mkdir -p ${trans_dir}
    if [ -f ${trans_file} ]; then
        echo "${trans_file} exists, removing"
        rm ${trans_file}
        # continue
    fi

    mkdir -p ${trans_dir}
    echo "command: trans ${src}:${tgt} file://${source_file} > ${trans_file}"
    trans ${src}:${tgt} file://${source_file} > ${trans_file}
done
