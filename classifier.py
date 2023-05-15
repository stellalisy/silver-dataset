import os
import torch
from transformers import XLMRobertaForTokenClassification, XLMRobertaTokenizerFast
from transformers import TrainingArguments, Trainer
import numpy as np
from transformers import DataCollatorForTokenClassification
from transformers import EarlyStoppingCallback, IntervalStrategy
from datasets import load_dataset, concatenate_datasets
from datasets.dataset_dict import DatasetDict, Dataset
from evaluate import load as load_metric
from collections import defaultdict
import json
from projector import read_segmented, read_segmented_filter
import random

def load_silver_dataset(translate_dir: str, src_langs: str, tgt_lang: str, train_size: int, eval_ratio: float = 0.2, phrase: bool = False):
    silver_langs = src_langs.split(',')
    if tgt_lang in silver_langs:
        silver_langs.remove(tgt_lang)
    lang_split_ratio = defaultdict(dict)
    split_dict = defaultdict(lambda: defaultdict(list))
    phrase_suffix = '.phrase' if phrase else ''
    for src in silver_langs:
        data_dir = os.path.join(translate_dir, tgt_lang, src+'-'+tgt_lang+phrase_suffix)
        for split in ['train', 'test', 'validation']: 
            if tgt_lang == 'zh':
                file_name = os.path.join(data_dir, src+'-'+tgt_lang+'.'+split+'.seg')
            else:
                file_name = os.path.join(data_dir, src+'-'+tgt_lang+'.'+split+'.trans')
            print("classifier.py line 28 - reading file:", file_name)
            res_dict, ratio = read_segmented_filter(file_name, src, tgt_lang)
            lang_split_ratio[src][split] = ratio
            for k, v in res_dict.items():
                split_dict[split][k].extend(v)
        print("silver dataset size:", len(split_dict['train']['tokens']))
    print("lang_split_ratio:", lang_split_ratio)
    if train_size == -1:
        train_size = len(split_dict['train']['tokens'])
    eval_size = int(train_size * eval_ratio)
    ret_dataset_dict = {}
    ret_dataset_dict['train'] = sample(split_dict['train'], train_size, 'train')
    ret_dataset_dict['validation'] = sample(split_dict['validation'], eval_size, 'validation')
    ret_dataset_dict['test'] = sample(split_dict['test'], eval_size, 'test')
    print("silver dataset size [train: {}, validation:{}, test:{}]:".format(len(split_dict['train']['tokens']),len(split_dict['validation']['tokens']),len(split_dict['test']['tokens'])))
    return DatasetDict(ret_dataset_dict)

def load_oro_dataset(src_langs: str, tgt_lang: str, train_size: int, eval_ratio: float = 0.2):
    oro_langs = src_langs.split(',')
    if tgt_lang in oro_langs:
        oro_langs.remove(tgt_lang)
    collect_datasets = {'train': [], 'validation': [], 'test': []}  # collect all datasets
    for src in oro_langs:
        src_dataset = load_dataset("wikiann", src)
        for split in ['train', 'validation', 'test']:
            collect_datasets[split].append(src_dataset[split])
    if train_size == -1:
        train_size = sum([len(dataset) for dataset in collect_datasets['train']])
    eval_size = int(train_size * eval_ratio)
    for split in ['train', 'validation', 'test']:
        dataset = concatenate_datasets(collect_datasets[split])
        dataset = dataset.remove_columns('spans')
        print("dataset.features:", type(dataset.features))
        collect_datasets[split] = sample(dataset_to_dict(dataset), train_size, split) if split == 'train' else sample(dataset_to_dict(dataset), eval_size, split)
    return DatasetDict(collect_datasets)

def sample(datadict: dict, train_size: int, split: str = 'train'):
    array_len = len(datadict['tokens'])
    if train_size <= array_len:
        idxs = random.sample(range(array_len), train_size)
    else:
        idxs = list(range(array_len)) * (train_size//array_len) + random.sample(list(range(array_len)), (train_size%array_len))
        random.shuffle(idxs)
    for feature in datadict.keys():
        datadict[feature] = [datadict[feature][i] for i in idxs]
    print("randomly selected {} samples from combined silver {} set".format(train_size, split))
    return Dataset.from_dict(datadict)

def dataset_to_dict(dataset: Dataset):
    dataset_dict = {}
    for feature in dataset.features.keys():
        dataset_dict[feature] = dataset[feature]
    return dataset_dict

def combine_ds(oro_dataset_dict: DatasetDict, silver_dataset_dict: DatasetDict):
    combined_dataset_dict = {}
    for split in ['train', 'validation', 'test']: 
        silver_dataset_dict[split].features['ner_tags'].feature = oro_dataset_dict[split].features['ner_tags'].feature
        combined_dataset_dict[split] = concatenate_datasets([oro_dataset_dict[split], silver_dataset_dict[split]])
    return DatasetDict(combined_dataset_dict)

def train_stage1(args, model, tokenizer, label_names, metric):
    oro_dataset = load_oro_dataset(args.stage1_oro_langs, args.tgt_lang, args.stage1_oro_size)
    print("stage 1 oro_dataset: {}".format(oro_dataset))

    tokenized_oro_dataset = tokenize_data(oro_dataset, tokenizer, label_names)

    training_args_s1 = TrainingArguments(
        output_dir=args.model_save_dir_s1,
        evaluation_strategy=IntervalStrategy.STEPS,
        learning_rate=1e-4,
        num_train_epochs=20,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        weight_decay=0.01,
        logging_steps = 2500,
        save_strategy=IntervalStrategy.STEPS,
        save_steps = 2500,
        save_total_limit = 2,
        # metric_for_best_model = "f1",
        # report_to="wandb",
        # run_name = args.wandb_name,
        load_best_model_at_end=False
    )
    trainer = get_trainer(training_args_s1, tokenized_oro_dataset['train'], tokenized_oro_dataset['validation'], model, tokenizer, label_names, metric)
    trainer.train()
    return trainer

def train_stage2(args, model, tokenizer, label_names, metric):
    if args.stage2_oro_langs == "":
        combined_dataset = load_silver_dataset(args.silver_dir, args.stage2_silver_langs, args.tgt_lang, args.stage2_silver_size, phrase=args.phrase)
        print("stage 2 silver_dataset: {}".format(combined_dataset))
        
    else:
        silver_dataset = load_silver_dataset(args.silver_dir, args.stage2_silver_langs, args.tgt_lang, args.stage2_silver_size, phrase=args.phrase)
        print("stage 2 silver_dataset: {}".format(silver_dataset))
        
        oro_dataset = load_oro_dataset(args.stage2_oro_langs, args.tgt_lang, args.stage2_oro_size)
        print("stage 2 oro_dataset: {}".format(oro_dataset))
        
        combined_dataset = combine_ds(oro_dataset, silver_dataset)
        print("combined_dataset: {}".format(combined_dataset))

    tokenized_combined_dataset = tokenize_data(combined_dataset, tokenizer, label_names)
          
    training_args_s2 = TrainingArguments(
        output_dir=args.model_save_dir_s2,
        evaluation_strategy=IntervalStrategy.STEPS,
        learning_rate=5e-6,
        max_steps=60000,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        weight_decay=0.01,
        logging_steps = 2500,
        save_strategy=IntervalStrategy.STEPS,
        save_steps = 2500,
        save_total_limit = 2,
        # metric_for_best_model = "f1",
        # report_to="wandb",
        # run_name = args.wandb_name,
        load_best_model_at_end=True
    )
    trainer = get_trainer(training_args_s2, tokenized_combined_dataset['train'], tokenized_combined_dataset['validation'], model, tokenizer, label_names, metric, patience=None)
    print("training stage 2 started")
    trainer.train()
    return trainer

def initialize_trainer(model, tokenizer, label_names, metric):
    def compute_metrics(p):
        predictions, labels = p
        predictions = np.argmax(predictions, axis=2)
        # Remove ignored index (special tokens)
        true_predictions = [ [label_names[p] for (p, l) in zip(prediction, label) if l != -100] for prediction, label in zip(predictions, labels) ]
        true_labels = [ [label_names[l] for (p, l) in zip(prediction, label) if l != -100] for prediction, label in zip(predictions, labels) ]

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
    
    data_collator = DataCollatorForTokenClassification(tokenizer)
    trainer = Trainer(
            model=model,
            data_collator=data_collator,
            tokenizer=tokenizer,
            compute_metrics=compute_metrics
        )
    return trainer
    
def get_trainer(training_args, train_dataset, eval_dataset, model, tokenizer, label_names, metric, patience=None):
    def compute_metrics(p):
        predictions, labels = p
        predictions = np.argmax(predictions, axis=2)
        # Remove ignored index (special tokens)
        true_predictions = [ [label_names[p] for (p, l) in zip(prediction, label) if l != -100] for prediction, label in zip(predictions, labels) ]
        true_labels = [ [label_names[l] for (p, l) in zip(prediction, label) if l != -100] for prediction, label in zip(predictions, labels) ]

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
     
    data_collator = DataCollatorForTokenClassification(tokenizer)
    trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            tokenizer=tokenizer,
            compute_metrics=compute_metrics,
            callbacks = [EarlyStoppingCallback(early_stopping_patience=patience)] if patience else None
        )
    return trainer

def tokenize_data(combined_dataset, tokenizer, label_names):
    max_length = 514
    def tokenize_adjust_labels(all_samples_per_split):
        tokenized_samples = tokenizer(all_samples_per_split["tokens"], is_split_into_words=True, max_length=max_length-2, truncation=True, return_length=True)
        #tokenized_samples is not a datasets object so this alone won't work with Trainer API, hence map is used 
        #so the new keys [input_ids, labels (after adjustment)]
        #can be added to the datasets dict for each train test validation split
        total_adjusted_labels = []
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
    return combined_dataset.map(tokenize_adjust_labels, batched=True)

def classifier_eval(args, trainer, testdataset, stage=2, write=False):
    test_results = trainer.evaluate(testdataset, metric_key_prefix='test')
    print("test_results:", test_results)
    if (not write) or (args.eval_output == ""):
        return test_results
    if os.path.isabs(args.eval_output):
        out_file = args.eval_output
    else:
        if stage == 2:
            out_file = os.path.join(args.model_save_dir_s2, args.eval_output)
        else:
            out_file = os.path.join(args.model_save_dir_s1, args.eval_output)
    results_obj = json.dumps(test_results, indent=4)
    with open(out_file, 'w+') as f:
        f.write(results_obj)

def classifier_train(args):
    if args.only_load_silver:
        combined_dataset = load_silver_dataset(args.silver_dir, args.stage2_silver_langs, args.tgt_lang, args.stage2_silver_size, phrase=args.phrase)
        print("silver_dataset: {}".format(combined_dataset))
        exit()
    
    max_length = 514
    tgt_dataset = load_dataset("wikiann", args.tgt_lang)
    label_names = tgt_dataset["train"].features["ner_tags"].feature.names
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    tokenizer = XLMRobertaTokenizerFast.from_pretrained(args.model_init_dir, model_max_length=max_length, max_length=max_length-2, truncation=True)
    print("tokenizer.vocab_size:", tokenizer.vocab_size)
    model = XLMRobertaForTokenClassification.from_pretrained(args.model_init_dir, num_labels=len(label_names), max_position_embeddings=max_length)
    print("loaded model from", args.model_init_dir)
    model.to(device)

    metric = load_metric("seqeval")
    tokenized_tgt_dataset = tokenize_data(tgt_dataset, tokenizer, label_names)

    trainer = initialize_trainer(model, tokenizer, label_names, metric)
    if args.predict_only:
        print("\n====================predicting loaded model: {}====================".format(args.model_init_dir))
        classifier_eval(args, trainer, tokenized_tgt_dataset["test"], write=False)
        return

    if args.stage1_oro_langs != "":
        print("\n====================training stage 1====================")
        trainer = train_stage1(args, trainer.model, tokenizer, label_names, metric)
        print("\n====================predicting stage 1====================")
        classifier_eval(args, trainer, tokenized_tgt_dataset["test"], stage=1, write=False)
        trainer.save_model(os.path.join(args.model_save_dir_s2, args.tgt_lang + "_best.model"))
    if args.stage2_silver_langs != "":
        print("\n====================training stage 2====================")
        trainer = train_stage2(args, trainer.model, tokenizer, label_names, metric)
        print("\n====================predicting stage 2====================")
        classifier_eval(args, trainer, tokenized_tgt_dataset["test"], stage=2, write=True)
        trainer.save_model(os.path.join(args.model_save_dir_s2, args.tgt_lang + "_best.model"))