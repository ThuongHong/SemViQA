from openai import OpenAI
import os
import pandas as pd
import argparse

client = OpenAI(api_key='')

def gpt(claim, content, prompt, args):
    response = client.chat.completions.create(
        model=args.model,
        messages=[
            {"role": "system", "content": prompt},  
            {"role": "user", "content": content}  
        ],
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        top_p=args.top_p,
    )
    return response.choices[0].message.content

def main(args):
    data = pd.read_csv(args.input_file)

    evi = []
    for i in data.index:
        context = data.loc[i, 'context']
        claim = data.loc[i, 'claim']
