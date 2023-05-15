cd /home/sli136/silver-data-creation/stanford-segmenter

src_langs=ar,bo,de,en,es,fr,hi,ja,ru,vi,zh-classical,zh-min-nan,zh-yue
splits=train,validation,test

for src in ${src_langs//,/ }; do
    folder=/export/c11/sli136/silver-dataset/translations/zh/${src}-zh
    test_seg=${folder}/${src}-zh.test.seg

    if [ -f ${test_seg} ]; then
        echo "segmened files exist in ${folder}, skipping"
        continue
    fi
    for split in ${splits//,/ }; do
        echo "segmenting for ${split} split in ${folder}"
        trans_file=${folder}/${src}-zh.${split}.trans
        seg_file=${folder}/${src}-zh.${split}.seg
        ./segment.sh pku ${trans_file} UTF-8 0 > ${seg_file}
    done
done