ORO_LANGS = {
    "ar": "he,fa,ur,ku,ps,tg",
    "de": "en,nl,nn",
    "es": "pt,ca,it",
    "fr": "it,ca,es",
    "hi": "mr,sa,ne,sd",
    "ru": "be,bg,kk,ky,mk,sr,tg,tk,uk,uz",
    "vi": "yo,ig,gn,it,sl,cy",
    "zh": "zh-yue,zh-classical,zh-min-nan,ja,bo"
}

SILVER_LANGS = {
    "ar": "am,mt,he,fa,ps,ku,ur",
    "de": "en,nl,da,sv,nn,no,fo,af",
    "es": "pt,ca,it,fr,ro",
    "fr": "pt,ca,it,es,ro",
    "hi": "ur,ne,bn,pa,mr,gu,sd,sa,as,or,bh,fa,ps,ku,tg,en",
    "ru": "uk,be,pl,cs,sk,bg,sr,hr,bs,sl",
    "vi": "km,zh,th,en,zh-yue",
    "zh": "zh-yue,ja,bo,vi,en"
}

TOKENIZER = "xlm-roberta-base"
# SRC_LANG='en'  # new change 8/19/2023
TGT_LANG='mg'
CLASSIFICATION_MODEL = "xlm-roberta-base"  #"bert-base-multilingual-cased"
TRANSLATION_MODEL = "facebook/m2m100_418M" # or use command line trans