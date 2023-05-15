import json
import csv
import pandas as pd
from sacremoses import MosesTokenizer

mt = MosesTokenizer(lang='en')

f = open("/exp/mmohammadi/better/data/basic_event/gold/clean_sent/basic.eng-provided-72.0pct.train-70.0pct.d.bp.json")
#f = open("/exp/mmohammadi/better/data/basic_event/gold/clean_sent/basic.eng-provided-72.0pct.analysis-15.0pct.ref.d.bp.json")
data = json.load(f)
entries = data['entries']

#src_trans_csv = open("better/data/data.analysis.brac.csv","r", encoding="utf-8-sig")
src_trans_csv = open("better/data/data.train.brac.csv","r", encoding="utf-8-sig")
csv_analysis = csv.DictReader(src_trans_csv, delimiter=',')

def tokenize_str(text):
    text = f"DUMMYWORD {text} DUMMYWORD"
    text_tok_dir = mt.tokenize(text, escape=False)
    text_tok = []
    for c in text_tok_dir:
      text_tok.append(c)
    return text_tok[1:-1]


def sta_end_char_offsets(tgt_brac_tok, sta_tok_tgt, end_tok_tgt):
  print("** tgt_brac_tok, sta_tok_tgt, end_tok_tgt", tgt_brac_tok, sta_tok_tgt, end_tok_tgt)
  tgt_span = []
  sta_tgt = 0
  end_tgt = 0
  leng = 0
  for tok_id in range(sta_tok_tgt, end_tok_tgt+1):
     tgt_span.append(tgt_brac_tok[tok_id])
     leng += len(tgt_brac_tok[tok_id]) + 1
  leng -= 1
  for l in range(sta_tok_tgt):
    #if l > 0:
      sta_tgt += len(tgt_brac_tok[l]) + 1
  end_tgt = sta_tgt + leng
  print("tgt_span, leng", tgt_span, leng)
  print("sta_tgt, end_tgt", sta_tgt, end_tgt)
  return sta_tgt, end_tgt

def find_tgt_span(src_brac_tok, tgt_brac_tok):
  #print("tgt_brac_tok", tgt_brac_tok)
  sta_tok_tgt = -1
  end_tok_tgt = -1
  sta_tgt = -1
  end_tgt = -1
  if '{' in tgt_brac_tok and '}' in tgt_brac_tok:
    if tgt_brac_tok.index('{') < tgt_brac_tok.index('}'):
      print("BOTH")
      sta_tok_tgt = tgt_brac_tok.index('{')
      end_tok_tgt = tgt_brac_tok.index('}') - 2
      tgt_brac_tok.pop(tgt_brac_tok.index('}'))
      tgt_brac_tok.pop(tgt_brac_tok.index('{'))
      sta_tgt, end_tgt = sta_end_char_offsets(tgt_brac_tok, sta_tok_tgt, end_tok_tgt)
    else:
      print("swapped }{")
  elif '{' in tgt_brac_tok:
    print("ONLY {")
    # get one token after {
    #print("tgt_brac_tok.index('{') , len(tgt_brac_tok)", tgt_brac_tok.index('{') , len(tgt_brac_tok)-1)
    if tgt_brac_tok.index('{') < len(tgt_brac_tok)-1: # { is not the last token
      sta_tok_tgt = tgt_brac_tok.index('{')
      end_tok_tgt = sta_tok_tgt
      tgt_brac_tok.pop(tgt_brac_tok.index('{'))
      sta_tgt, end_tgt = sta_end_char_offsets(tgt_brac_tok, sta_tok_tgt, end_tok_tgt)
  elif '}' in tgt_brac_tok:
    print("ONLY }")
    # get one token before }
    sta_tok_tgt = tgt_brac_tok.index('}') - 1
    end_tok_tgt = sta_tok_tgt
    tgt_brac_tok.pop(tgt_brac_tok.index('}'))
    sta_tgt, end_tgt = sta_end_char_offsets(tgt_brac_tok, sta_tok_tgt, end_tok_tgt)
  #else:
  #  if '-' in tgt_brac_tok and '-' not in src_brac_tok:
  #    print("- in tgt")
  #  elif '"' in tgt_brac_tok and '"' not in src_brac_tok:
  #    print("\" in tgt")
  #  else:
  return tgt_brac_tok, sta_tok_tgt, end_tok_tgt, sta_tgt, end_tgt

#dict = {'sta': [], 'end': [], 'src': []}
new_entries = {}
for entry_id, entry in entries.items():
  print('\n', entry_id)
  seg_txt_tok = entry['segment-text-tok'][:]
  #print("seg_txt_tok", seg_txt_tok)
  events = entry['annotation-sets']['basic-events']
  starts_ends = set()
  for span_set_id, span_set in events['span-sets'].items():
    #print(span_set_id, span_set['spans'], len(span_set['spans']))
    spans = span_set['spans']
    for span in spans:
      #print("span", span)
      sta = span['start-token']
      end = span['end-token']
      starts_ends.add(str(sta) + " "+ str(end))
  #print("starts_ends", starts_ends)
  
  i = 0
  for sta_end in sorted(starts_ends):
    #print("span_set_id", span_set_id)
    sta = int(sta_end.split(" ")[0])
    end = int(sta_end.split(" ")[1])
    seg_txt_tok.insert(end+1, '}')
    seg_txt_tok.insert(sta, '{')
    #dict['sta'].append(sta)
    #dict['end'].append(end)
    #dict['src'].append(' '.join(seg_txt_tok))
    print(sta, end, seg_txt_tok)
    next_csv_analysis = next(csv_analysis)
    print("next_csv_analysis['src'], ' '.join(seg_txt_tok)", next_csv_analysis['src'], ' '.join(seg_txt_tok))
    assert(next_csv_analysis['src'] == ' '.join(seg_txt_tok))
    seg_txt_tok = entry['segment-text-tok'][:]
    print("next_csv_analysis['tgt']", next_csv_analysis['tgt'])
    tok_tgt = tokenize_str(next_csv_analysis['tgt'])
    tok_src = tokenize_str(next_csv_analysis['src'])
    tgt_brac_tok, sta_tok_tgt, end_tok_tgt, sta_tgt, end_tgt = find_tgt_span(tok_src, tok_tgt)
    print("tgt_brac_tok, sta_tok_tgt, end_tok_tgt", tgt_brac_tok, sta_tok_tgt, end_tok_tgt)
    for span_set_id, span_set in events['span-sets'].items():
#      new_spans = []
      spans = span_set['spans']
      for span in spans:
        if span['start-token'] == sta and span['end-token'] == end:
         if sta_tok_tgt != -1 and end_tok_tgt != -1:
          new_span_sets = {}
          new_span = {}
          new_entry = {"annotation-sets": {"basic-events": {"events": {}, "span-sets": {}}}, "doc-id": '', "entry-id": '', "segment-sections": [], "segment-text": '', "segment-type": ''}
          new_entry['annotation-sets']['basic-events']['events'] = events['events']
          #print("span" ,span)
          #span['start'] = 
          new_span['string'] = ' '.join(tgt_brac_tok[sta_tok_tgt:end_tok_tgt+1])
          new_span['string-tok'] = tgt_brac_tok[sta_tok_tgt:end_tok_tgt+1]
          new_span['hstring'] = new_span['string']
          new_span['hstring-tok'] = new_span['string-tok']
          print("span['string']", new_span['string'])
          new_span['start-token'] = sta_tok_tgt
          new_span['end-token'] = end_tok_tgt
          new_span['start'] = sta_tgt
          new_span['end'] = end_tgt
          #if 'start' in span:
          #  del span['start']
          #if 'hstart' in span:
          #  del span['hstart']
          #if 'end' in span:
          #  del span['end']
          #if 'hend' in span:
          #  del span['hend']
 #     new_spans.append(new_span)
          new_entry['segment-text-tok'] = tgt_brac_tok
          new_entry['segment-text'] = ' '.join(tgt_brac_tok)
          new_span_sets[span_set_id] = {"spans": [], "ssid": ''}
          print("new_span, span_set_id", new_span, span_set_id)
          new_span_sets[span_set_id]['spans'].append(new_span) 
          new_span_sets[span_set_id]['ssid'] = span_set_id
          new_entry['annotation-sets']['basic-events']['span-sets'] = new_span_sets
          new_entry['entry-id'] = entry_id +"+"+ str(i)
          new_entry['doc-id'] = entry['doc-id']
          seg_sec = {"end": len(new_entry['segment-text']), "start": 0, "structural-element": "Sentence"}
          new_entry["segment-sections"].append(seg_sec)
          # check if the same segment-text already exists
          added = False
          for ee_id, existing_entry in new_entries.items():
            if new_entry['segment-text'] == existing_entry['segment-text']:
              print("segment text exists in "+ existing_entry['entry-id'])
              added = True
              existing_entry['annotation-sets']['basic-events']['span-sets'].update(new_entry['annotation-sets']['basic-events']['span-sets'])
              break
          if added == False:
            new_entries[entry_id +"+"+ str(i)] = new_entry
            i += 1

output = {"corpus-id": data["corpus-id"], "entries": {}}
output["entries"] = new_entries

with open("tmpout.bp.json", 'w', encoding='utf8') as f:
  json.dump(output, f, indent=2, ensure_ascii=False)

