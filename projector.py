from datasets import Dataset
from collections import defaultdict
from typing import Tuple

def read_segmented(filename, tgt_lang) -> "Dataset":
    silver_dict = defaultdict(list)
    
    with open(filename, 'r') as f:
        for line in f.readlines():
            tokens, ner_tags, langs, spans = [], [], [], []
            label = 0
            segmented = line.replace('><', '> <').replace(' >', '>').replace('< ', '<').replace('</ ', '</')
            if tgt_lang == 'ar':
                segmented = segmented.replace('<1 />', '</1>').replace('<2 />', '</2>').replace('<3 />', '</3>').replace('<4 />', '</4>').replace('<5 />', '</5>').replace('<6 />', '</6>')
            segmented = segmented.split()
            for j in range(len(segmented)):
                if '<' in segmented[j] or '>' in segmented[j]:
                    # print("'<' or '>' in segmented[j]")
                    for ch in segmented[j]:
                        if ch.isdigit():
                            try:
                                label = int(ch)
                            except:
                                try:
                                    label = int(ch.translate('%7d   45678923'%10*999))
                                except:
                                    label = 0
                            # print("label is: {}".format(label))
                            if label >= 7: # if label > num_classes
                                continue
                            break
                    if '/' in segmented[j]:
                        label = 0
                else:
                    tokens.append(segmented[j])
                    # ner_tags.append(tag_names[int(label)])
                    ner_tags.append(label)
                    langs.append(tgt_lang)
            silver_dict['tokens'].append(tokens)
            silver_dict['ner_tags'].append(ner_tags)
            silver_dict['langs'].append(langs)
            # silver_dict['spans'].append(spans)
    return Dataset.from_dict(silver_dict)


def read_segmented_filter(filename, src_lang, tgt_lang) -> Tuple[dict, float]:
    silver_dict = defaultdict(list)
    total, good = 0,0
    print("reading file:", filename)
    with open(filename, 'r') as f:
        for line in f.readlines():
            total += 1
            res = seg_sent(line, tgt_lang)
            if not res:
                continue
            tokens, ner_tags, langs = res
            silver_dict['tokens'].append(tokens)
            silver_dict['ner_tags'].append(ner_tags)
            silver_dict['langs'].append(langs)
            good += 1
            # silver_dict['spans'].append(spans)
    print("{}: {}-{}: {}".format(filename.split('/')[-1], src_lang, tgt_lang, good/total))
    # return Dataset.from_dict(silver_dict), good/total
    return silver_dict, good/total

def seg_sent(line, tgt_lang):
    # print('[{}]: {}'.format(tgt_lang, line))
    matched, between = True, False
    temp = []
    tokens, ner_tags, langs = [], [], []
    label = 0
    line = ' '.join(line.split()).strip()
    segmented = line.replace('><', '> <').replace(' >', '> ').replace('< ', ' <').replace('</ ', ' </').replace('>', '> ').replace('<', ' <').replace('</', ' </')
    if tgt_lang == 'ar':
        segmented = segmented.replace('<1 />', '</1>').replace('<2 />', '</2>').replace('<3 />', '</3>').replace('<4 />', '</4>').replace('<5 />', '</5>').replace('<6 />', '</6>')
    segmented = segmented.split()
    for j in range(len(segmented)):
        if '<' in segmented[j] and '>' in segmented[j]  \
                and (len(segmented[j])==3 or len(segmented[j])==4) \
                and segmented[j][-2].isdigit():
            if len(segmented[j]) == 3:
                if matched == False:
                    return None
                label = int(segmented[j][-2])
                matched = False
                between = True
            elif len(segmented[j]) == 4:
                if '/' in segmented[j] and int(segmented[j][-2]) == label:
                    matched = True
                    between = False
                else:
                    return None
        elif not matched and between:
            temp.append(segmented[j])
        if matched and not between:
            if len(temp) > 0:
                langs.extend([tgt_lang]*len(temp))
                if label % 2 == 0 or len(temp) == 1:
                    tokens.extend(temp)
                    ner_tags.extend([label]*len(temp))
                else:
                    tokens.extend(temp)
                    ner_tags.extend([label] + [label+1]*(len(temp)-1))
                temp = []
            else:
                tokens.append(segmented[j])
                ner_tags.append(0)
                langs.append(tgt_lang)
    if matched == False:
        return None
    return tokens, ner_tags, langs
