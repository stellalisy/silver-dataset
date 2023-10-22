import os
import argparse
# import wandb
import time
from pathlib import Path

from classifier import classifier_train
from translator import translate, mark
from constants import *

os.environ["WANDB_DISABLED"] = "true"
# os.environ["WANDB_API_KEY"]="a84285031fcd2e0955fd1d015249882145a057ff"
# os.environ["WANDB_ENTITY"]="mark-translate"
# os.environ["WANDB_PROJECT"]="translate"

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    ############# Training related arguments #############
    parser.add_argument(
        "-b",
        "--batch_size",
        type=int,
        default=16,
        help="batch size"
    )
    parser.add_argument(
        "--stage1_oro_langs", type=str, default="",
        help="untranslated data used in stage 1 fine-tuning"
    )
    parser.add_argument(
        "--stage2_oro_langs", type=str, default="",
        help="untranslated data used in stage 2 fine-tuning"
    )
    parser.add_argument(
        "--stage2_silver_langs", type=str, default="",
        help="translated data used in stage 2 fine-tuning"
    )
    # parser.add_argument("-s", "--src_lang", type=str, default=SRC_LANG, help="source language")   # new change 8/19/2023
    parser.add_argument(
        "-t",
        "--tgt_langs",
        type=str,
        default=TGT_LANG,
        help="target language"
    )
    parser.add_argument(
        "--stage1_oro_size", type=int, default=-1,
        help="untranslated data used in stage 1 fine-tuning"
    )
    parser.add_argument(
        "--stage2_oro_size", type=int, default=-1,
        help="untranslated data used in stage 2 fine-tuning"
    )
    parser.add_argument(
        "--stage2_silver_size", type=int, default=-1,
        help="translated data used in stage 2 fine-tuning"
    )
    parser.add_argument(
        "--stage3_gold_size", type=int, default=0,
        help="few shot fine-tuning on gold data"
    )
    parser.add_argument(
        "-eo",
        "--eval_output",
        type=str,
        default="",
        help="eval output file name"
    )
    parser.add_argument(
        "-md1",
        "--model_save_dir_s1",
        type=str,
        default="/export/c11/sli136/silver-dataset/model",
        help="model save directory for stage 1 training"
    )
    parser.add_argument(
        "-md2",
        "--model_save_dir_s2",
        type=str,
        default="/export/c11/sli136/silver-dataset/model",
        help="model save directory for stage 2 training"
    )
    parser.add_argument(
        "-md3",
        "--model_save_dir_s3",
        type=str,
        default=None,
        help="model save directory for stage 3 training"
    )
    parser.add_argument(
        "-mi",
        "--model_init_dir",
        type=str,
        default=CLASSIFICATION_MODEL,
        help="name of the classification model, \
            if neither this or stage1_oro_lang provided, fine-tune XLMR from scratch, \
            if only stage1_oro_lang provided, find the fine-tuned xlmr model in model_dir"
    )
    parser.add_argument(
        "--only_load_silver",
        action=argparse.BooleanOptionalAction,
        type=bool,
        default=False,
        help="exit the program after silver datasets are loaded (stats are printed)"
    )

    parser.add_argument(
        "-td",
        "--silver_dir",
        type=str,
        default="/export/c11/sli136/silver-dataset/translations",
        help="directory to save/load the translated silver data"
    )

    ############# Prediction related arguments #############
    parser.add_argument(
        "-po",
        "--predict_only",
        action=argparse.BooleanOptionalAction,
        type=bool,
        default=False,
        help="only predict the target language from provided model"
    )
    ############# Mark related arguments #############
    parser.add_argument(
        "-mo",
        "--mark_only",
        action=argparse.BooleanOptionalAction,
        type=bool,
        default=False,
        help="only mark the source language data"
    )
    parser.add_argument(
        "--seg_level",
        type=str,
        default="token",   # or phrase
        help="only mark the source language data"
    )
    parser.add_argument(
        "--dry_run",
        action=argparse.BooleanOptionalAction,
        type=bool,
        default=False,
        help="only mark the source language data"
    )
    ############# Translation related arguments #############
    parser.add_argument(
        "-to",
        "--translate_only",
        action=argparse.BooleanOptionalAction,
        type=bool,
        default=False,
        help="only translate"
    )
    parser.add_argument(
        "-tm",
        "--translation_model",
        type=str,
        default=TRANSLATION_MODEL,
        help="model name"
    )
    parser.add_argument(
        "-mf",
        "--marked-file",
        type=Path,
        default=None,
        help="marked source data (only needed for translations)"
    )
    return parser.parse_args()

args = parse_args()
args.tgt_langs = args.tgt_langs.replace(' ', '').split(',')
if args.stage1_oro_langs == "none": args.stage1_oro_langs = ""
if args.stage2_oro_langs == "none": args.stage2_oro_langs = ""
if args.stage2_silver_langs == "none": args.stage2_silver_langs = ""
if args.model_save_dir_s3 == None: args.model_save_dir_s3 = os.path.join(args.model_save_dir_s2, 'gold')
args.phrase = True if args.seg_level == "phrase" else False

for k, v in vars(args).items():
    print(f'[{time.strftime("%Y-%m-%d %H:%M:%S")}] {k}: {v}')

if args.translate_only: translate(args)

elif args.mark_only: 
    if args.phrase:
        mark(args, phrase=True)
    else:
        mark(args)

else: classifier_train(args)

# wandb.finish()