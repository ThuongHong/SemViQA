import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pyvi import ViTokenizer, ViPosTagger
from rank_bm25 import BM25Okapi
from data_processing.pipline import split_sentence, preprocess_text
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_extraction.text import TfidfVectorizer

###### Evidence ######
def qatc(claim, context, model_evidence_QA, tokenizer_QA, device):
    model_evidence_QA.to(device)
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
 
    if '</s></s>' in evidence :
        evidence = evidence.split('</s></s>')[1] 
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
    
def qatc_faster(claim, context, full_context, model_evidence_QA, tokenizer_QA, device = "cuda:0"):
    if isinstance(claim, str ):
        claim = [claim]
        context = [context]
    model_evidence_QA.to(device)
    model_evidence_QA.eval()
    inputs = tokenizer_QA(
        claim, 
        context, 
        max_length = 512, 
        return_tensors="pt",
        truncation="only_second", 
        padding="max_length"
    )
    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)

    with torch.no_grad():
        pt, start_logits, end_logits = model_evidence_QA(input_ids = input_ids, attention_mask = attention_mask)

    start_logits = start_logits.view(start_logits.shape[0], start_logits.shape[1])
    end_logits = end_logits.view(end_logits.shape[0], end_logits.shape[1])
    
    start_index = start_logits.argmax(dim = -1).tolist()
    end_index = end_logits.argmax(dim = -1).tolist()
    answer_ids = []
    for i in range(len(start_index)):
        answer_ids.append(inputs['input_ids'][0][start_index[i]:end_index[i] + 1])

    evidences = tokenizer_QA.batch_decode(answer_ids, skip_special_tokens=True)
    true_evi = []
    for i, evi in enumerate(evidences):
        if isinstance(evi, str) == False: continue
        if len(preprocess_text(evi).split()) > 3:
            true_evi.append((evi, i))
    if len(true_evi) != 1:
        return -1 
    evidence = true_evi[0][0].lstrip(".")
    lines = split_sentence(full_context)
    processed_evidence = preprocess_text(evidence) 
    for line in lines:
        if processed_evidence in preprocess_text(line):
            return line 
    return -1
    
def tfidf_topk(context, claim, thres= 0.6, top_k=1):
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

def bm25_topk(context, claim, top_k=None):
    context = split_sentence(context)
    if top_k is None:
        return context

    tokenized_context = [doc.split(' ') for doc in context]
    bm25 = BM25Okapi(tokenized_context)
    scores = bm25.get_scores(claim.split())
 
    max_score = max(scores)
    min_score = min(scores)
    normalized_scores = [
        (score - min_score) / (max_score - min_score) if max_score > min_score else 0
        for score in scores
    ]
 
    score_sentence_pairs = sorted(zip(normalized_scores, context), reverse=True)
    highest_sentence = score_sentence_pairs[:top_k] if top_k else score_sentence_pairs

    return highest_sentence

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def sbert_topk(context, claim, tokenizer_sbert, model_sbert, top_k = 1, device = 'cuda'):
    context = split_sentence(context)
    sentences = [claim] + context

    encoded_input = tokenizer_sbert(sentences, padding=True, truncation=True, return_tensors='pt')
    encoded_input = {key: value.to(device) for key, value in encoded_input.items()}

    with torch.no_grad():
        model_output = model_sbert(**encoded_input)

    sentence_embeddings = mean_pooling(model_output, encoded_input["attention_mask"])

    claim_embedding = sentence_embeddings[0].unsqueeze(0)

    similarities = []
    cosine = nn.CosineSimilarity(dim=1, eps=1e-6)
    
    for i in range(1, len(sentence_embeddings)):
        evidence_embedding = sentence_embeddings[i].unsqueeze(0)
        similarity = cosine(claim_embedding, evidence_embedding).item()
        similarities.append((similarity, sentences[i]))
 
    simi_values = [s[0] for s in similarities]
    scaler = MinMaxScaler()
    scaled_simi_values = scaler.fit_transform(np.array(simi_values).reshape(-1, 1)).flatten()
 
    similarities = [(round(scaled_simi_values[i], 2), similarities[i][1]) for i in range(len(similarities))]

    similarities.sort(key=lambda x: x[0], reverse=True)

    return similarities[:top_k]

###### Classify ######
def three_class_classification(claim, context, model_classify_3_class, tokenizer_3_class, device):
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

def binary_classification(claim, evidence, model_classify_binary, tokenizer_binary, device):
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


