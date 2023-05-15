import os
import torch
import numpy as np
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer
from datasets import load_metric, load_dataset
from pathlib import Path

def mark(args, phrase=False):
    mark_suffix = 'marked' if not phrase else 'marked.phrase'
    file_path = os.path.join(args.silver_dir, 'marked', f'{args.src_lang}.{mark_suffix}', f'{args.src_lang}.train.{mark_suffix}')
    if os.path.exists(file_path):
        print("{} exists, skipping marking".format(file_path))
        return
    dataset_src = load_dataset("wikiann", args.src_lang.replace('_', '-'))
    # print("dataset_src['train'].features['ner_tags'].feature.names:", dataset_src['train'].features['ner_tags'].feature.names)
    tag_names = dataset_src['train'].features['ner_tags'].feature.names
    marked_dataset = dataset_src.map(insert_marks_map, batched=True) if not phrase else dataset_src.map(insert_marks_map_phrase, batched=True)
    for split, v in marked_dataset.items():
        print("marked {} sentences from {} in {} split".format(len(v['marked']), args.src_lang, args.tgt_lang, split))
        # print("v['labels']:", v['labels'])
        if not args.dry_run:
            save_marked(v['marked'], args, split)
            save_tags(v['labels'], args, split, tag_names)

    
def translate(args):
    if 'm2m' in args.translation_model:
        max_length = 1024
    else:
        max_length = 512

    def tokenize_adjust_labels(all_samples_per_split):
        # print("all_samples_per_split:", all_samples_per_split, ", type:", type(all_samples_per_split))
        # marked_sents = insert_marks(all_samples_per_split)
        tokenized_samples = tokenizer.batch_encode_plus(all_samples_per_split['tokens'], is_split_into_words=True, return_tensors="pt", 
                                                            max_length=max_length, padding='max_length', truncation=True).to(device)
        #tokenized_samples is not a datasets object so this alone won't work with Trainer API, hence map is used 
        #so the new keys [input_ids, labels (after adjustment)]
        #can be added to the datasets dict for each train test validation split
        # tokenized_samples['marked'] = marked_sents
        tokenized_samples['original'] = all_samples_per_split["tokens"]
        tokenized_samples['labels'] = all_samples_per_split["ner_tags"]
        return tokenized_samples

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    model = M2M100ForConditionalGeneration.from_pretrained(args.translation_model, max_position_embeddings=max_length)
    tokenizer = M2M100Tokenizer.from_pretrained(args.translation_model, src_lang=args.src_lang, tgt_lang=args.tgt_lang, max_length=max_length)
    model.to(device)
    print("loaded model")

    if os.path.isfile(args.marked_file):
        print("processing file:", args.marked_file)
        with open(args.marked_file, 'r') as f:
            lines = f.readlines()
        batches = []
        batch = []
        for i, line in enumerate(lines):
            if i != 0 and i % args.batch_size == 0:
                batches.append(batch)
                batch = []
            batch.append(line)
        translations = []
        for batch in batches:
            encoded_source = tokenizer(batch, is_split_into_words=True, return_tensors="pt", 
                                                    max_length=max_length, padding='max_length', truncation=True).to(device)
            generated_tokens = model.generate(**encoded_source, forced_bos_token_id=tokenizer.get_lang_id(args.tgt_lang))
            translated_sents = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
            print(translated_sents[0])
            translations.append(translated_sents)
        print("translated {} sentences from {} to {}".format(len(translated_sents), args.src_lang, args.tgt_lang))
        save_translation(translated_sents, args, str(args.marked_file).split('.')[-2])
        return
    # if os.path.isdir(arg)
    return

def insert_marks(batch):
    marked_sents = []
    tokens = batch['tokens']
    tags = batch['ner_tags']
    for sample, tag in zip(tokens, tags):
        padded_sent = []
        for i, token, in enumerate(sample):
            if tag[i] == 0: padded_sent.append(token)
            else: padded_sent.extend(['<'+str(tag[i])+'>', token, '</'+str(tag[i])+'>'])
        marked_sents.append(' '.join(padded_sent))
    return marked_sents

def insert_marks_phrase(batch):
    marked_sents, tokens, tags = [], batch['tokens'], batch['ner_tags']
    for sample, tag in zip(tokens, tags):
        start_count, end_count = 0, 0
        # print("before:", ' '.join(sample))
        if 'R E D I R E C T' in ' '.join(sample[:8]) or 'r e d i r e c t' in ' '.join(sample[:8]):
            sample = sample[8:]
            tag = tag[8:]
        start, end = process_tags(tag)
        padded_sent = []
        for i, token, in enumerate(sample):
            if start[i] and end[i]: 
                padded_sent.extend(['<'+str(start[i])+'>', token, '</'+str(end[i])+'>'])
                start_count += 1
                end_count += 1
            elif start[i]: 
                padded_sent.extend(['<'+str(start[i])+'>', token])
                start_count += 1
            elif end[i]: 
                padded_sent.extend([token, '</'+str(end[i])+'>'])
                end_count += 1
            else: padded_sent.append(token)
        marked_sents.append(' '.join(padded_sent))
        # print("after:", marked_sents[-1])
        if start_count != end_count:
            print("start count {} != end count {}".format(start_count, end_count))
            print("sample:", sample)
            print("tag:", tag)
            print("start:", start)
            print("end:", end)
            print("padded_sent:", padded_sent)
            print("marked_sents[-1]:", marked_sents[-1])
            exit()
        # assert start_count == end_count, "start count {} != end count {}".format(start_count, end_count)
    return marked_sents


def process_tags(tags):
    start, end = [], []
    curr, phr_len, curr_tag = 0, 0, 0
    while curr < len(tags):
        if tags[curr] % 2 == 1:
            if phr_len > 0:
                end[-1] = curr_tag
                phr_len = 0
            curr_tag = tags[curr]
            start.append(curr_tag)
            end.append(0)
            phr_len += 1
            curr += 1
        elif tags[curr] == 0:
            if phr_len > 0: # end of an NER phrase
                end[-1] = curr_tag
                phr_len = 0
            start.append(0)
            end.append(0)
            curr += 1
        elif phr_len == 0:
            curr_tag = tags[curr] - 1
            start.append(curr_tag)
            end.append(0)
            phr_len += 1
            curr += 1
        else:
            start.append(0)
            end.append(0)
            phr_len += 1
            curr += 1
    if phr_len > 0:
        end[-1] = curr_tag
    return start, end


def insert_marks_map(all_samples_per_split):
    marked_sents = insert_marks(all_samples_per_split)
    untokenized_samples = {}
    untokenized_samples['marked'] = marked_sents
    untokenized_samples['original'] = all_samples_per_split["tokens"]
    untokenized_samples['labels'] = all_samples_per_split["ner_tags"]
    return untokenized_samples

def insert_marks_map_phrase(all_samples_per_split):
    marked_sents = insert_marks_phrase(all_samples_per_split)
    untokenized_samples = {}
    untokenized_samples['marked'] = marked_sents
    untokenized_samples['original'] = all_samples_per_split["tokens"]
    untokenized_samples['labels'] = all_samples_per_split["ner_tags"]
    return untokenized_samples

def save_translation(out, args, split):
    if args.translation_model == 'mark':
        affix = '.marked'
    else: affix = '.pred'
    data_dir = os.path.join(args.silver_dir, args.src_lang+'-'+args.tgt_lang)
    with open(os.path.join(data_dir, args.src_lang+'-'+args.tgt_lang+'.'+split+affix), 'w+') as f:
        f.writelines('\n'.join(out))

def save_marked(out, args, split):
    mark_suffix = '.marked' if not args.phrase else '.marked.phrase'
    data_dir = os.path.join(args.silver_dir, 'marked', args.src_lang+mark_suffix)
    with open(os.path.join(data_dir, args.src_lang+'.'+split+mark_suffix), 'w+') as f:
        f.writelines('\n'.join(out))

def save_tags(out, args, split, tag_names):
    mark_suffix = '.marked' if not args.phrase else '.marked.phrase'
    out = ['\t'.join([tag_names[i] for i in s]) for s in out]
    affix = '.tags'
    data_dir = os.path.join(args.silver_dir, 'marked', args.src_lang+mark_suffix)
    with open(os.path.join(data_dir, args.src_lang+'.'+split+affix), 'w+') as f:
        f.writelines('\n'.join(out))
    with open(os.path.join(data_dir, args.src_lang+'.'+split+".lookup"), 'w+') as f:
        f.writelines('\n'.join(tag_names))