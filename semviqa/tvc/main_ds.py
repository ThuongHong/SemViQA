import torch
import pandas as pd
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, XLMRobertaTokenizerFast
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score, accuracy_score, classification_report, precision_score, recall_score, cohen_kappa_score, matthews_corrcoef, confusion_matrix, roc_auc_score
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
from tqdm import tqdm
import os
import time
import multiprocessing
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration, set_seed, DeepSpeedPlugin

from .data_utils import Data
from .model import ClaimModelConfig, ClaimModelForClassification

multiprocessing.set_start_method('spawn', force=True)

def set_seed_all(seed):
    set_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def count_parameters(model):
    print(f'The model has {sum(p.numel() for p in model.parameters() if p.requires_grad):,} trainable parameters')
    print(f'The model has {sum(p.numel() for p in model.parameters()):,} parameters')

def main(args):
    # Accelerate + DeepSpeed setup
    logging_dir = os.path.join(args.output_dir, "logs")
    accelerator_project_config = ProjectConfiguration(
        project_dir=args.output_dir, logging_dir=logging_dir
    )
    ds_plugin = DeepSpeedPlugin(
        zero_stage=2,
        gradient_accumulation_steps=args.accumulation_steps,
        hf_ds_config={
            "train_batch_size": args.train_batch_size * args.accumulation_steps,
            "gradient_accumulation_steps": args.accumulation_steps,
            "fp16": {"enabled": False},
            "zero_optimization": {
                "stage": 2,
                "allgather_partitions": True,
                "allgather_bucket_size": 200000000,
                "reduce_scatter": True,
                "reduce_bucket_size": 200000000,
                "overlap_comm": True
            }
        }
    )
    print(args.train_batch_size)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.accumulation_steps,
        project_config=accelerator_project_config,
        deepspeed_plugin=ds_plugin
    )
    set_seed_all(42)

    train_data = pd.read_csv(args.train_data).reset_index()
    dev_data = pd.read_csv(args.dev_data).reset_index()
    train_data["id"] = train_data.index
    dev_data["id"] = dev_data.index
    if args.n_classes == 2:
        train_data = train_data[train_data["verdict"] != "NEI"]
        dev_data = dev_data[dev_data["verdict"] != "NEI"]
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if args.is_pretrained:
        config = ClaimModelConfig.from_pretrained(args.model_name)
        config.num_labels = args.n_classes
        config.dropout = args.dropout_prob
        config.loss_type = args.type_loss
        model = ClaimModelForClassification.from_pretrained(args.model_name, config=config)
    else:
        config = ClaimModelConfig(
            model_name=args.model_name,
            num_labels=args.n_classes,
            dropout=args.dropout_prob,
            loss_type=args.type_loss
        )
        model = ClaimModelForClassification(config)
    count_parameters(model)
    train_dataset = Data(train_data, tokenizer, args, max_len=args.max_len)
    dev_dataset = Data(dev_data, tokenizer, args, max_len=args.max_len)
    train_loader = DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=True, num_workers=args.num_workers)
    dev_loader = DataLoader(dev_dataset, batch_size=args.train_batch_size, shuffle=False, num_workers=args.num_workers)
    optimizer = AdamW(model.parameters(), lr=args.lr)
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=0, 
        num_training_steps=len(train_loader)*args.epochs
    )
    # Prepare for multi-GPU
    model, optimizer, train_loader, dev_loader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_loader, dev_loader, lr_scheduler
    )
    info_epoch = {}
    logs = []
    best_acc = 0
    cnt = 0
    i = 0
    total_time = time.time()
    for epoch in range(args.epochs):
        set_seed_all(42)
        info_epoch[epoch] = {}
        if accelerator.is_main_process:
            print(f'Epoch {epoch+1}/{args.epochs}')
            print('-'*30)
        start_training_time = time.time()
        model.train()
        train_losses = []
        true_labels = []
        predicted_labels = []
        for data in tqdm(train_loader, disable=not accelerator.is_local_main_process):
            y_true = data['targets'].to(accelerator.device)
            input_ids = data['input_ids'].to(accelerator.device)
            attention_mask = data['attention_masks'].to(accelerator.device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=y_true)
            loss = outputs["loss"]
            logits = outputs["logits"] 
            train_losses.append(loss.item())
            accelerator.backward(loss)
            if (i + 1) % args.accumulation_steps == 0:
                optimizer.step()
                lr_scheduler.step() 
                optimizer.zero_grad()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            i += 1
            _, pred = torch.max(logits, dim=1)
            true_labels.extend(y_true.cpu().numpy())
            predicted_labels.extend(pred.cpu().numpy())
        train_f1 = f1_score(true_labels, predicted_labels, average='macro')
        train_acc = accuracy_score(true_labels, predicted_labels)
        epoch_training_time = time.time() - start_training_time
        if accelerator.is_main_process:
            print(f'Training time: {epoch_training_time:.2f}s Train Loss: {np.mean(train_losses):.4f} F1: {train_f1:.4f} Acc: {train_acc:.4f}')
        model.eval()
        eval_losses = []
        y_true_list = []
        y_pred_list = []
        start_eval_time = time.time()
        with torch.no_grad():
            for data in tqdm(dev_loader, disable=not accelerator.is_local_main_process):
                y_true = data['targets'].to(accelerator.device)
                input_ids = data['input_ids'].to(accelerator.device)
                attention_mask = data['attention_masks'].to(accelerator.device)
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=y_true)
                loss = outputs["loss"]
                logits = outputs["logits"]
                eval_losses.append(loss.item())
                _, pred = torch.max(logits, dim=1)
                y_true_list.extend(y_true.cpu().numpy())
                y_pred_list.extend(pred.cpu().numpy())
        dev_f1 = f1_score(y_true_list, y_pred_list, average='macro')
        dev_acc = accuracy_score(y_true_list, y_pred_list)
        epoch_eval_time = time.time() - start_eval_time
        if accelerator.is_main_process:
            print(f'Dev time: {epoch_eval_time}s Dev Loss: {np.mean(eval_losses):.4f} F1: {dev_f1:.4f} Acc: {dev_acc:.4f}')
        info_epoch[epoch] = {
            "time_train": epoch_training_time,
            "epoch": epoch,
            "train_loss": np.mean(train_losses),
            "train_acc": train_acc,
            "f1-train": train_f1,
            "time_val": epoch_eval_time,
            "val_acc": dev_acc,
            "val_loss": np.mean(eval_losses),
            "f1-val": dev_f1
        }
        if dev_f1 > best_acc and accelerator.is_main_process:
            cnt = 0
            tokenizer.save_pretrained(output_dir)
            model.save_pretrained(output_dir)
            config.save_pretrained(output_dir)
            print(f'Saved best_model at epoch {epoch+1}')
            best_acc = dev_f1
        else:
            cnt += 1
        if cnt >= args.patience:
            if accelerator.is_main_process:
                print('Early stopping')
            break
        torch.cuda.empty_cache()
    if accelerator.is_main_process:
        print(f'Total training time: {time.time() - total_time:.2f}s')
        pd.DataFrame(info_epoch).T.to_csv(os.path.join(output_dir, 'info_epoch.csv'), index=False)

def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_data', type=str, default='data/train.csv')
    parser.add_argument('--dev_data', type=str, default='data/dev.csv')
    parser.add_argument('--model_name', type=str, default='MoritzLaurer/ernie-m-large-mnli-xnli')
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--train_batch_size', type=int, default=8)
    parser.add_argument('--max_len', type=int, default=256)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--type_loss', type=str, default='cross_entropy')
    parser.add_argument('--output_dir', type=str, default='output')
    parser.add_argument('--n_classes', type=int, default=3)
    parser.add_argument('--is_weighted', type=int, default=0)
    parser.add_argument('--dropout_prob', type=float, default=0.3)
    parser.add_argument('--accumulation_steps', type=int, default=1)
    parser.add_argument('--is_pretrained', type=int, default=0)
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    main(args)
