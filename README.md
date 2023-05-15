# mark-then-translate project

Silver dataset creation project using Wikiann data for Named Entity Recognition (NER).

### Step-by-step Instruction

```
Example:
src_lang=ca
tgt_lang=es
trans_dir=[path/to/translation/directory]
```

1. First, insert tags to source oro data:  `qsub run-mark.sh`
    ```
    python3 run.py \
        -b 16 \
        -s ${src_lang} \
        --mark_only --phrase \
        -td ${trans_dir}
    ```
2. Second, translate marked data with [Google Translate Shell](https://github.com/soimort/translate-shell): `qsub run-trans.sh`
    ```
    trans ${src_lang}:${tgt_lang} file://${source_file} > ${trans_file}
    ```
    2.1. If ${tgt_lang} is Chinese, run [segmenter](https://nlp.stanford.edu/software/segmenter.shtml) before next step.

3. Extract target sentences and tag information and train NER model (fine-tune on XLMR): `qsub run-main-phrase.sh ${tgt_lang} ${exp_num}`
    ```
    python3 run.py \
        --batch_size 4 \
        --phrase \
        --stage1_oro_langs pt,ca,it --stage1_oro_size 20000 \
        --stage2_oro_langs pt,ca,it,fr,ro --stage2_oro_size 20000 \
        --stage2_silver_langs pt,ca,it,fr,ro --stage2_silver_size -1 \
        --tgt_lang ${tgt_lang} \
        --eval_output test_results.json \
        --model_save_dir_s1 ${model_dir_s1} \
        --model_save_dir_s2 ${model_dir_s2} \
        --model_init_dir roberta-xlmr-base \
        --silver_dir ${trans_dir}
    ```

