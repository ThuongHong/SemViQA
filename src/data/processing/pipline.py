import re
from rank_bm25 import BM25Okapi
import nltk
nltk.download('punkt')
from nltk.tokenize import sent_tokenize

def preprocess_text(text: str) -> str:    
    text = re.sub(r"['\",\.\?:\!]", "", text)
    text = text.strip()
    text = " ".join(text.split())
    return text.lower()


def split_sentence(paragraph):
    context_list = []
    if paragraph[-2:] == '\n\n':
        paragraph = paragraph[:-2]

    paragraph = paragraph.rstrip()  
    start = 0
    paragraph_length = len(paragraph)

    while start < paragraph_length:  
        context = ""
        initial_start = start

        for i in range(start, paragraph_length):
            if paragraph[i] == ".":

                if i + 2 < paragraph_length and paragraph[i + 1] == "\n":
                    if paragraph[i + 2].isalpha() and paragraph[i + 2].isupper():
                        break

                if i + 1 < paragraph_length and paragraph[i + 1] == " ":
                    context += paragraph[i]
                    start = i + 1
                    break

            context += paragraph[i]

            if i == paragraph_length - 1:
                start = paragraph_length
                break

        if start == paragraph_length:
            context += paragraph[start:]
        
        context = preprocess_text(context.strip())  
        if len(context.split()) > 2:
            context_list.append(context)

        if start == initial_start:
            print("Warning: No progress detected. Exiting loop.")
            break

    return context_list

def process_data(text):
    return '. '.join(split_sentence(text))

def procce_meta(meta): 
    meta = meta.drop_duplicates()
    count_value = meta.claim.value_counts().values
    value = meta.claim.value_counts().keys()
    cnt = 0
    wrong_claims = []
    for i in range(len(count_value)):
        if count_value[i] > 1:
            cnt+=1
            wrong_claims.append(value[i])
    idx_list = meta[meta.claim.isin(wrong_claims)].sort_values(by='claim').index
    meta.drop(idx_list, inplace=True)
    id_n = []
    for i in meta.index:
        if meta.verdict[i] != 'NEI':
            if "\n\n" in meta.evidence[i]:
                id_n.append(i)
    meta.drop(id_n, inplace=True)
    meta.drop(meta[meta.claim == meta.evidence].index, inplace= True)
    evi = []
    for i in meta.index:
        if meta.verdict[i] == 'NEI':
            evi.append("")
        else:
            evi.append(process_data(meta.evidence[i]))
    meta.evidence = evi
    return meta

def load_data(data):
    data_old = {}

    for i in data.index:
        if data.id[i] not in data_old.keys():
            data_old[data.id[i]] = [
                {
                    'id': data.id[i],
                    'context': data.context[i],
                    'claim': data.claim[i]
                }
            ]
        else:
            data_old[data.id[i]].append(
                    {
                        'id': data.id[i],
                        'context': data.context[i],
                        'claim': data.claim[i]
                    }
                )
    return data_old

def get_top_context(context, claim = None, topk = None):
    context = split_sentence(context)
    if topk == None:
        return context
    
    
    context = [line for line in context]

    tokenized_context = [doc.split(' ') for doc in context]
    bm25 = BM25Okapi(tokenized_context)
    scores = bm25.get_scores(claim.split())

    score_sentence_pairs = sorted(zip(scores, context), reverse=True)
    highest_sentence = []

    for _, x in score_sentence_pairs[:topk]:

        highest_sentence.append(x)
        
    return highest_sentence