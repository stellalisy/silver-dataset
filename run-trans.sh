base_langs='en,ar,de,es,fr,hi,ru,vi,zh'

tgt=ar
same_script=he,fa,ur,ku,ps,tg
sim_langs=am,mt,he,fa,ps,ku,ur

# tgt=de
# same_script=en,nl,nn
# sim_langs=nl,da,sv,nn,no,fo,af  #en

# tgt=es
# same_script=pt,ca,it
# sim_langs=pt,ca,it,ro  #fr

# tgt=fr
# same_script=it,ca,es
# sim_langs=pt,ca,it,ro  #es

# tgt=hi
# same_script=mr,sa,ne,sd
# sim_langs=ur,ne,bn,pa,mr,gu,sd,sa,as,or,bh,fa,ps,ku,tg  #en

# tgt=ru
# same_script=be,bg,kk,ky,mk,sr,tg,tk,uk,uz
# sim_langs=uk,be,pl,cs,sk,bg,sr,hr,bs,sl

# tgt=vi
# same_script=yo,ig,gn,it,sl,cy
# sim_langs=km,th,zh-yue  #zh,en

# tgt=zh
# same_script=zh,zh-yue,zh-classical,zh-min-nan,ja,bo
# sim_langs=zh-yue,ja,bo

for src in ${base_langs//,/ }; do
if [[ ${src} == ${tgt} ]]; then
    continue
fi
qsub /home/sli136/silver-data-creation/trans-shell.sh ${src} ${tgt}
done

for src in ${sim_langs//,/ }; do
if [[ ${src} == ${tgt} ]]; then
    continue
fi
qsub /home/sli136/silver-data-creation/trans-shell.sh ${src} ${tgt}
done