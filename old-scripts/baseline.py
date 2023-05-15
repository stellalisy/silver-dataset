import os
# from os import listdir
# from os.path import isfile, join
import argparse
# import sys
# import csv
from string import punctuation
# import re
# import subprocess
# import itertools
from itertools import compress

import numpy as np
# from sklearn.model_selection import train_test_split
# from sklearn import metrics
import torch
import torch.nn as nn
# from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
from transformers import XLMRobertaForTokenClassification, XLMRobertaTokenizerFast
from transformers import TrainingArguments, Trainer
import numpy as np
from datasets import load_metric, load_dataset
import numpy as np
# import sentencepiece

import json
from transformers import DataCollatorForTokenClassification

import wandb

SRC_LANG='en'
TGT_LANG='mg'
MODEL_NAME = "xlm-roberta-base"  #"bert-base-multilingual-cased"
DEV_SIZE = 3000
TEST_SIZE = 3000

# INFERENCE_DATA_DIR = "/export/c07/sli136/switchboard/processed/pw"
# DATA_DIR = "/export/c11/kenton/miniscale23/iwslt-tunisian-bitext/LDC2022E01-stm"

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-w",
        "--wandb_name",
        type=str,
        default="fine-tune",
        help="wandb project name"
    )
    parser.add_argument(
        "-m",
        "--model_name",
        type=str,
        default=MODEL_NAME,
        help="model name"
    )
    parser.add_argument(
        "-b",
        "--batch_size",
        type=int,
        default=16,
        help="batch size"
    )
    parser.add_argument(
        "-o",
        "--output_file",
        type=str,
        default="/export/c11/sli136/silver-dataset/model/results.json",
        help="batch size"
    )
    parser.add_argument(
        "-s",
        "--src_lang",
        type=str,
        default=SRC_LANG,
        help="source language"
    )
    parser.add_argument(
        "-t",
        "--tgt_lang",
        type=str,
        default=TGT_LANG,
        help="target language"
    )
    parser.add_argument(
        "-md",
        "--model_dir",
        type=str,
        default="/export/c11/sli136/silver-dataset/model",
        help="model save directory"
    )
    parser.add_argument(
        "-r",
        "--resume_saved",
        type=str,
        default="",
        help="resume training from saved model"
    )
    parser.add_argument(
        "-i",
        "--inference_only",
        action=argparse.BooleanOptionalAction,
        type=bool,
        default=False,
        help="inference only"
    )
    return parser.parse_args()

args = parse_args()

os.environ["WANDB_API_KEY"]="a84285031fcd2e0955fd1d015249882145a057ff"
os.environ["WANDB_ENTITY"]="mark-translate"
os.environ["WANDB_PROJECT"]=args.wandb_name

if 'xlm' in args.model_name:
    max_length = 514
else:
    max_length = 512

dataset = load_dataset("wikiann", args.tgt_lang.replace('_', '-'))
label_names = dataset["train"].features["ner_tags"].feature.names
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

tokenizer = XLMRobertaTokenizerFast.from_pretrained(args.model_name, model_max_length=max_length, max_length=max_length)
model = XLMRobertaForTokenClassification.from_pretrained(args.model_name, num_labels=len(label_names), max_position_embeddings=max_length)

def tokenize_adjust_labels(all_samples_per_split):
    tokenized_samples = tokenizer.batch_encode_plus(all_samples_per_split["tokens"], is_split_into_words=True, max_length=max_length-2, truncation=True)
    #tokenized_samples is not a datasets object so this alone won't work with Trainer API, hence map is used 
    #so the new keys [input_ids, labels (after adjustment)]
    #can be added to the datasets dict for each train test validation split
    total_adjusted_labels = []
    # print("max tokenized_samples['input_ids']:", max([len(tokenized_samples["input_ids"][i]) for i in range(len(tokenized_samples["input_ids"]))]))
    for k in range(0, len(tokenized_samples["input_ids"])):
        prev_wid = -1
        word_ids_list = tokenized_samples.word_ids(batch_index=k)
        existing_label_ids = all_samples_per_split["ner_tags"][k]
        i = -1
        adjusted_label_ids = []
    
        for wid in word_ids_list:
            if(wid is None):
                adjusted_label_ids.append(-100)
            elif(wid!=prev_wid):
                i = i + 1
                adjusted_label_ids.append(existing_label_ids[i])
                prev_wid = wid
            else:
                label_name = label_names[existing_label_ids[i]]
                adjusted_label_ids.append(existing_label_ids[i])
            
        total_adjusted_labels.append(adjusted_label_ids)
    tokenized_samples["labels"] = total_adjusted_labels
    return tokenized_samples

tokenized_dataset = dataset.map(tokenize_adjust_labels, batched=True)

data_collator = DataCollatorForTokenClassification(tokenizer)

metric = load_metric("seqeval")

def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    # Remove ignored index (special tokens)
    true_predictions = [
        [label_names[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [label_names[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    results = metric.compute(predictions=true_predictions, references=true_labels)
    flattened_results = {
        "overall_precision": results["overall_precision"],
        "overall_recall": results["overall_recall"],
        "overall_f1": results["overall_f1"],
        "overall_accuracy": results["overall_accuracy"],
    }
    for k in results.keys():
      if(k not in flattened_results.keys()):
        flattened_results[k+"_f1"]=results[k]["f1"]

    return flattened_results

training_args = TrainingArguments(
    output_dir=args.model_dir,
    learning_rate=2e-5,
    per_device_train_batch_size=args.batch_size,
    per_device_eval_batch_size=args.batch_size,
    num_train_epochs=10,
    weight_decay=0.01,
    logging_steps = 1000,
    report_to="wandb",
    run_name = args.wandb_name + "_" + args.tgt_lang,
    save_total_limit = 2,
    save_strategy = "epoch",
    evaluation_strategy = "epoch",
    load_best_model_at_end=True
)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["validation"],
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)



trainer.train()
test_results = trainer.evaluate(tokenized_dataset["test"])
results_obj = json.dumps(test_results, indent=4)
with open(args.eval_output, 'w+') as f:
    f.write(results_obj)
trainer.save_model(args.model_dir + args.tgt_lang + "_best.model")
wandb.finish()