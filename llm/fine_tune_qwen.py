import os
import pandas as pd
import torch
from tqdm.auto import tqdm
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    default_data_collator,
    DataCollatorWithPadding
)
import time
import numpy as np
import random
import argparse

class Data(Dataset):
    def __init__(self, df, tokenizer, system_prompt, config):
        self.df = df
        self.tokenizer = tokenizer
        self.config = config
        self.system_prompt = system_prompt
    
    def __len__(self):
        return len(self.df)
        
    def __getitem__(self, index):
        row = self.df.iloc[index]
        claim, context, ids, label, evidence = self.get_input_data(row)
         
        user_prompt = (
            f"Claim: {claim}\n"
            f"Context: {context}\n"
            "Please extract the sentence from the context that supports or contradicts the claim."
        ) 
        assistant_response = f"Evidence: {evidence}"
         
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": assistant_response}
        ]
         
        if hasattr(self.tokenizer, "apply_chat_template"):
            prompt_text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        else: 
            prompt_text = (
                self.system_prompt + "\n" +
                user_prompt + "\n" +
                assistant_response
            )
        
        encoding = self.tokenizer(
            prompt_text,
            truncation=True,
            max_length=self.config.max_len,
            padding="max_length"
        )
        return encoding
        

    def get_input_data(self, row):
        claim = row['claim']
        context = row['context']
        ids = row['id']
        label = row['verdict']
        evidence = row['evidence']
        return claim, context, ids, label, evidence

def main(args): 
    system_prompt = (
        "You are an expert fact-checking analyst. "
        "Your task is to extract the sentence that provides evidence from the context based on the given claim."
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    train = pd.read_csv(args.train_data)
    dev = pd.read_csv(args.dev_data)

    train_dataset = Data(train, tokenizer, system_prompt, args)
    dev_dataset = Data(dev, tokenizer, system_prompt, args)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=True,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_dir=args.output_dir,
        logging_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        tokenizer=tokenizer,
        data_collator=default_data_collator,
    )

    trainer.train()
    trainer.save_model(args.output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True, help="Pretrained model name")
    parser.add_argument("--train_data", type=str, required=True, help="Path to the training CSV file")
    parser.add_argument("--dev_data", type=str, required=True, help="Path to the development CSV file")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the trained model")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size per device")
    parser.add_argument("--max_len", type=int, default=6400, help="Maximum input length")
    args = parser.parse_args()
    
    main(args)
