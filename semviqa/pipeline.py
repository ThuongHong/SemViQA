from transformers import AutoTokenizer
from tqdm.notebook import tqdm
import torch
import pandas as pd
import json
from tqdm import tqdm

from .data_processing.pipline import process_data, load_data
from .ser.ser_eval import *
from .ser.qatc_model import QATCForQuestionAnswering
from .tvc.tvc_eval import *
from .tvc.model import ClaimModelForClassification
import argparse

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
    # data['context'] = data['context'].progress_apply(lambda x: process_data(x))
    cag = ['NEI', 'SUPPORTED', 'REFUTED']

    test_data = load_data(data)
    print(f'Test data: {len(test_data)}')

    ##### Load model #####
    # load QATC
    tokenizer_QA = AutoTokenizer.from_pretrained(args.model_evidence_QA)
    model_evidence_QA = QATCForQuestionAnswering.from_pretrained(args.model_evidence_QA).to(device) 

    ##### Load classify #####
    tokenizer_classify = AutoTokenizer.from_pretrained(args.model_3_class)
    model_3_class = ClaimModelForClassification.from_pretrained(args.model_3_class).to(device)
    model_2_class = ClaimModelForClassification.from_pretrained(args.model_2_class).to(device)
    print('Start predict')

    submit = {} 
    for i in tqdm(test_data.keys()):
        idx = str(i)
        context= test_data[i][0]['context']
        claim = test_data[i][0]['claim']

        evidence = extract_evidence_tfidf_qatc(claim, context, model_evidence_QA, tokenizer_QA, device, thres=args.thres_evidence)
        
        submit[idx] = {
                    'verdict': '1',
                    'evidence': evidence
            } 
        
        ##### classify #####
        context_sub = submit[idx]['evidence']
        claim_sub = claim

        try:
            prob3class, pred_3_class = classify_claim(claim_sub, context_sub, model_3_class, tokenizer_classify, device)
        except Exception as e:
            print(f"Error at index {idx} with context: {context_sub}")
            raise e  

        if pred_3_class == 0:
            submit[idx].update({'verdict': 'NEI'})
        else:
            prob2class, pred_2_class = classify_claim(claim_sub, context_sub, model_2_class, tokenizer_classify, device)

            label_2class = 'SUPPORTED' if pred_2_class == 0 else 'REFUTED'
            label_3class = cag[pred_3_class]
            
            submit[idx]['verdict'] = label_2class if prob2class > prob3class else label_3class


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
    parser.add_argument("--model_evidence_QA", type=str, default="QATC", help="Model evidence QA")  
    parser.add_argument("--model_2_class", type=str, default="2_class", help="Model 2 class") 
    parser.add_argument("--model_3_class", type=str, default="3_class", help="Model 3 class") 
    parser.add_argument("--thres_evidence", type=float, default=0.5, help="Threshold evidence")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
 
