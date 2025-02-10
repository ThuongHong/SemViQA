import numpy as np
import torch
import torch.nn.functional as F
from pyvi import ViTokenizer, ViPosTagger
from src.data.processing.pipline import split_sentence, preprocess_text
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

###### Evidence ######
def find_nei_evi(claim, context, model_evidence_QA, tokenizer_QA, device):
    # model_evidence_QA.to(device)
    model_evidence_QA.eval()
    inputs = tokenizer_QA(
        claim, 
        context, 
        max_length = 512, 
        return_tensors="pt",
        truncation="only_second", 
        padding="max_length"
    )
    input_ids = inputs['input_ids']
    attention_mask = inputs['attention_mask']

    pt, start_logits, end_logits = model_evidence_QA(input_ids = input_ids, attention_mask = attention_mask)

    start_logits = start_logits.detach().cpu().numpy()
    end_logits = end_logits.detach().cpu().numpy()
     
    start_index = np.argmax(start_logits)
    end_index = np.argmax(end_logits)
 
    answer_tokens = inputs['input_ids'][0][start_index:end_index + 1] 
    evidence = tokenizer_QA.decode(answer_tokens)

    ### check characte species
    if '</s></s>' in evidence :
        evidence = evidence.split('</s></s>')[1]
    # check number of sentence in evidence predict
    cntx = 0
    for p in split_sentence(evidence):
        if p.strip()!='':
            cntx+=1

    evidence = evidence.replace('<s>', '')
    evidence = evidence.replace('</s>', '')

    
    if evidence =='<s>' or len(evidence) == 0 or cntx > 1:
        return -1
    else:
        lines = split_sentence(context)
        for line in lines:
            if preprocess_text(evidence) in preprocess_text(line):
                return line
        print('error: not find evi in context')
        
        print(lines)
        print('==========')
        print(evidence)
        return evidence
    

def select_sentance_text(context, claim, thres= 0.6, top_k=1):
    tfidf_vectorizer = TfidfVectorizer()
    corpus = split_sentence(context)
    answer = corpus.copy()
    claim = preprocess_text(claim)
    claim = ViTokenizer.tokenize(claim).lower()
    
    len_claim = len(claim.split(' '))
    
    corpus_pro = []
    
    for i in range(len(corpus)):
        corpus[i] = preprocess_text(corpus[i])
        corpus[i] = ViTokenizer.tokenize(corpus[i]).lower()
        
        sentence = corpus[i]
        
        l = len(sentence.split(' '))
        
        p  = l/len_claim
        
        if i != 0 and p < thres and l > 1 :
            sentence = f'{corpus[i-1]}. {sentence}'
        corpus_pro.append(sentence)
    corpus_pro.append(claim)
    tfidf_matrix = tfidf_vectorizer.fit_transform(corpus_pro)
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    
    score = []
    
    for i in range(len(cosine_sim)-1):
        score.append((cosine_sim[len(corpus_pro)-1, i], answer[i]))
    
    score = sorted(score, reverse=True)
    top_k_sentences = score[:top_k]
    return top_k_sentences


###### Classify ######

def three_class_classify(claim, context, model_classify_3_class, tokenizer_3_class, device):
    model_classify_3_class.to(device)
    model_classify_3_class.eval()

    encoding = tokenizer_3_class(
            claim,
            context,
            truncation="only_second",
            add_special_tokens=True,
            max_length=256,
            padding='max_length',
            return_attention_mask=True,
            return_token_type_ids=False,
            return_tensors='pt',
        )

    inputs = {
                'input_ids': encoding['input_ids'].reshape((1, 256)),
                'attention_masks': encoding['attention_mask'].reshape((1, 256)),
            }

    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_masks'].to(device)

    with torch.no_grad():
        outputs = model_classify_3_class(
            input_ids=input_ids,
            attention_mask=attention_mask,

        )
    outputs = F.softmax(outputs, dim=1)

    prob3class, pred = torch.max(outputs, dim=1)
    return prob3class, pred[0].item()

def binary_classify(claim, evidence, model_classify_binary, tokenizer_binary, device):
    model_classify_binary.to(device)
    model_classify_binary.eval()

    context_sub =evidence
    claim_sub = claim

    encoding = tokenizer_binary(
            claim_sub,
            context_sub,
            truncation="only_second",
            add_special_tokens=True,
            max_length=256,
            padding='max_length',
            return_attention_mask=True,
            return_token_type_ids=False,
            return_tensors='pt',
        )

    inputs = {
                'input_ids': encoding['input_ids'].reshape((1, 256)),
                'attention_masks': encoding['attention_mask'].reshape((1, 256)),
            }

    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_masks'].to(device)

    with torch.no_grad():
        outputs = model_classify_binary(
            input_ids=input_ids,
            attention_mask=attention_mask,

        )
    outputs = F.softmax(outputs, dim=1)
    _, pred = torch.max(outputs, dim=1)

    return  _, pred.item()
