from transformers import AutoTokenizer

from tqdm.notebook import tqdm
import torch
import pandas as pd
import json
from tqdm import tqdm
from safetensors.torch import load_model

from src.data.processing.pipline import split_sentence, process_data, load_data
from models.classify import ClaimVerification
from models.qact import QACT
from eval import three_class_classify, binary_classify, find_nei_evi, select_sentance_text
import argparse

def check_evidence(claim, context, model_evidence_QA, tokenizer_QA, device, evidence_tf):
    lines = split_sentence(context)
    tokens = context.split(' ')
    
    if len(tokens) <= 400: 
        evi = find_nei_evi(claim = claim, context = context, model_evidence_QA = model_evidence_QA, tokenizer_QA = tokenizer_QA, device = device)
        if evi == -1:   
            return -1, evidence_tf
        return 0 , evi
    
        
    token_line = [l.split(' ') for l in lines]
    
    tmp_context_token = []
    tmp_context = []

    evidence_list = []
    for idx in range(len(lines)):
        check = True
        if len(token_line[idx] + tmp_context_token) <=400:
            tmp_context_token += token_line[idx]
            tmp_context.append(lines[idx])
            check = False
        
        if len(token_line[idx] + tmp_context_token) > 400 or idx == len(lines) - 1:
            context_sub = '. '.join(tmp_context)
            if len(context_sub)== 0: 
                continue
            evidence = find_nei_evi(claim = claim, context = context_sub, model_evidence_QA = model_evidence_QA, tokenizer_QA = tokenizer_QA, device = device)

               
            if evidence != -1: # If appear evidence
                
                evidence_list.append(evidence)
            if check:
                tmp_context_token = token_line[idx] 
                tmp_context = [lines[idx]]
            else:
                tmp_context_token = []
                tmp_context = []

    if len(evidence_list) == 1: 
        return 0, evidence_list[0]
    else :
        if len(evidence_list) == 0: 
            return -1, evidence_tf
        else: 
            return 0, evidence_tf


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if "csv" in args.data_path:
        data = pd.read_csv(args.data_path)
    else:
        data = pd.read_json(args.data_path).T
    
    print('Load data')

    # process data
    tqdm.pandas()
    data["id"] = data.index
    data['context'] = data['context'].progress_apply(lambda x: process_data(x))
    cag = ['NEI', 'SUPPORTED', 'REFUTED']

    test_data = load_data(data)
    print(f'Test data: {len(test_data)}')

    ##### Load model #####
    # load QACT
    model_evidence_QA = QACT(name_model= args.model_evidence_QA)
    if "safetensors" in args.weight_evidence_QA:
        load_model(model_evidence_QA, args.weight_evidence_QA)
    else:
        model_evidence_QA.load_state_dict(torch.load(args.weight_evidence_QA)['model_state_dict'])
    tokenizer_QA = AutoTokenizer.from_pretrained(args.model_evidence_QA)

    ##### Load classify #####
    # load 2 class classify
    model_2_class = ClaimVerification(n_classes=2, name_model= args.model_2_class)
    if "safetensors" in args.weight_2_class:
        load_model(model_2_class, args.weight_2_class)
    else:
        model_2_class.load_state_dict(torch.load(args.weight_2_class))
    tokenizer_2_class = AutoTokenizer.from_pretrained(args.model_2_class)

    # load 3 class classify
    model_3_class = ClaimVerification(n_classes=3, name_model= args.model_3_class)
    if "safetensors" in args.weight_3_class:
        load_model(model_3_class, args.weight_3_class)
    else:
        model_3_class.load_state_dict(torch.load(args.weight_3_class))
    tokenizer_3_class = AutoTokenizer.from_pretrained(args.model_3_class)

    print('Start predict')

    submit = {}
    cnt_not_use = 0
    id_nei_by_evi = []
    for i in tqdm(test_data.keys()):
        idx = str(i)
        context= test_data[i][0]['context']
        claim = test_data[i][0]['claim']

        top_tfidf = select_sentance_text(context, claim, top_k=1)[0]
        ##### evidence #####
        if top_tfidf[0] > args.thres_evidence:
            submit[idx] = {
                    'verdict': '1',
                    'evidence': top_tfidf[1]
            }
        else:
            not_nei, evidence = check_evidence(claim = claim, context = context, model_evidence_QA = model_evidence_QA, tokenizer_QA = tokenizer_QA, device = device, evidence_tf=top_tfidf[1])
            if not_nei == -1:
                id_nei_by_evi.append(i)
        
            submit[idx] = {
                        'verdict': '1',
                        'evidence': evidence
                }
            cnt_not_use+=1
        
        ##### classify #####
        context_sub = submit[idx]['evidence']
        claim_sub = claim

        try:
            prob3class, pred_3_class = three_class_classify(claim = claim_sub, context = context_sub, model_classify_3_class = model_3_class, tokenizer_3_class = tokenizer_3_class, device = device)
        except:
            print(idx)
            print(context_sub)
            break
        
        if pred_3_class == 0:
            submit[idx]['verdict'] ='NEI'
            submit[idx]['evidence'] =''
        else:
            prob2class, pred_rs = binary_classify(claim = claim_sub, evidence = context_sub, model_classify_binary = model_2_class, tokenizer_binary = tokenizer_2_class, device = device)
            label_3class = cag[pred_3_class]
            label_2class = ""
            if pred_rs == 0:
                label_2class = 'SUPPORTED'
            else: 
                label_2class = 'REFUTED'
                
            submit[idx]['verdict'] = label_2class
            
            if label_3class != label_2class:
        
                if prob2class > prob3class:
                    submit[idx]['verdict'] = label_2class
                else: submit[idx]['verdict'] = label_3class

    # Save file
    with open(args.output_path, "w", encoding="utf-8") as json_file:
        json.dump(submit, json_file, ensure_ascii=False, indent=4)
        
    check_data = pd.DataFrame(submit).T
    print(check_data.verdict.value_counts())
    print(data.verdict.value_counts())


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="data/test.json", help="Path to data")
    parser.add_argument("--output_path", type=str, default="output.json", help="Path to output")
    parser.add_argument("--model_evidence_QA", type=str, default="QACT", help="Model evidence QA")
    parser.add_argument("--weight_evidence_QA", type=str, default="weights/QACT.pth", help="Weight evidence QA")
    parser.add_argument("--model_2_class", type=str, default="2_class", help="Model 2 class")
    parser.add_argument("--weight_2_class", type=str, default="weights/2_class.pth", help="Weight 2 class")
    parser.add_argument("--model_3_class", type=str, default="3_class", help="Model 3 class")
    parser.add_argument("--weight_3_class", type=str, default="weights/3_class.pth", help="Weight 3 class")
    parser.add_argument("--thres_evidence", type=float, default=0.5, help="Threshold evidence")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
 
