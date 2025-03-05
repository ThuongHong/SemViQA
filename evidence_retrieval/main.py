import logging
import os
import pandas as pd
import numpy as np
import torch
import gc
from sklearn.metrics import f1_score, cohen_kappa_score, matthews_corrcoef
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from tqdm import tqdm
from transformers import default_data_collator
from torch.utils.data import DataLoader
import argparse
import time
from safetensors.torch import load_model

from qatc_model import QATCConfig, QATCForQuestionAnswering
from data_utils import load_data
from loss import comboLoss

os.environ["WANDB__SERVICE_WAIT"] = "300"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params}")
    active_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {active_params}")

def load_models(args):
    config = QATCConfig(
        model_name=args.model_name,
        freeze_text_encoder=args.freeze_text_encoder,
        alpha=args.alpha,
        beta=args.beta
    )
    
    model = QATCForQuestionAnswering.from_pretrained(args.model_name, config=config)
    
    if args.is_load_weight:
        print("loading train weight")
        if "safetensors" in args.weight_model:
            load_model(model, f"{args.weight_model}")
        else:
            model.load_state_dict(torch.load(args.weight_model), strict=False)
    
    count_parameters(model)
    return model

def setting_optimizer(config, model):
    optimizer_cls = torch.optim.AdamW
    return optimizer_cls(model.parameters(), lr=config.learning_rate)

def main(args):
    logger = get_logger(__name__, log_level="INFO")

    logging_dir = os.path.join(args.output_dir, args.logging_dir)
    accelerator_project_config = ProjectConfiguration(
        project_dir=args.output_dir, logging_dir=logging_dir
    )

    accelerator = Accelerator(
        log_with=args.report_to,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        project_config=accelerator_project_config
    )

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)

    if args.seed is not None:
        set_seed(args.seed)

    if accelerator.is_main_process and args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)

    model = load_models(args)
    model = accelerator.prepare(model)

    optimizer = setting_optimizer(args, model)
    train_dataset, test_dataset = load_data(args)
    
    train_dataloader = DataLoader(
        train_dataset, shuffle=True, collate_fn=default_data_collator, batch_size=args.train_batch_size
    )
    
    eval_dataloader = DataLoader(
        test_dataset, collate_fn=default_data_collator, batch_size=args.train_batch_size
    )

    optimizer, train_dataloader = accelerator.prepare(optimizer, train_dataloader)

    progress_bar = tqdm(desc="Steps", disable=not accelerator.is_local_main_process)

    min_loss = float("inf")
    best_acc = 0.0

    print("Starting training...")
    global_step = 1
    start_time = time.time()

    for epoch in range(args.num_train_epochs):
        model.train()
        train_loss = 0.0
        for step, batch in enumerate(train_dataloader):
            for key in batch:
                batch[key] = batch[key].to(accelerator.device)

            output = model(
                input_ids=batch['input_ids'], 
                attention_mask=batch['attention_mask'], 
                start_positions=batch['start_positions'], 
                end_positions=batch['end_positions'],
                tagging_labels=batch['Tagging']
            )

            loss = output.loss
            accelerator.backward(loss)

            optimizer.step()
            optimizer.zero_grad()

            train_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())
            progress_bar.update(1)
            global_step += 1

        train_loss /= len(train_dataloader)
        print(f"Epoch {epoch+1} - Train Loss: {train_loss}")

        model.eval()
        eval_loss = 0.0
        predictions, true_positions = [], []
        
        for step, batch in enumerate(eval_dataloader):
            with torch.no_grad():
                for key in batch:
                    batch[key] = batch[key].to("cuda")
                
                output = model(
                    input_ids=batch['input_ids'], 
                    attention_mask=batch['attention_mask']
                )

                loss = output.loss
                eval_loss += loss.item()

                start_preds = torch.argmax(output.start_logits, axis=1).cpu().detach().tolist()
                end_preds = torch.argmax(output.end_logits, axis=1).cpu().detach().tolist()
                start_true = batch['start_positions'].flatten().cpu().tolist()
                end_true = batch['end_positions'].flatten().cpu().tolist()

                predictions.extend(list(zip(start_preds, end_preds)))
                true_positions.extend(list(zip(start_true, end_true)))

        eval_loss /= len(eval_dataloader)
        accuracy = np.mean([p == t for p, t in zip(predictions, true_positions)])

        print(f"Epoch {epoch+1} - Eval Loss: {eval_loss} - Accuracy: {accuracy}")

        if accuracy > best_acc:
            best_acc = accuracy
            save_path = os.path.join(args.output_dir, "best_model.pth")
            torch.save(model.state_dict(), save_path)
            print("Saved best model.")

    print("Training completed in:", time.time() - start_time, "seconds.")

def parse_args():
    parser = argparse.ArgumentParser(description="Train and evaluate QATCForQuestionAnswering")
    
    parser.add_argument("--model_name", type=str, default="/kaggle/input/model-base")
    parser.add_argument("--output_dir", type=str, default="/kaggle/working/")
    parser.add_argument("--seed", type=int, default=40)
    parser.add_argument("--logging_dir", type=str, default="logs")
    parser.add_argument("--train_batch_size", type=int, default=12)
    parser.add_argument("--num_train_epochs", type=int, default=100)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--report_to", type=str, default="wandb")
    parser.add_argument("--freeze_text_encoder", type=int, default=0)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--is_load_weight", type=int, default=0)
    parser.add_argument("--weight_model", type=str, default="/kaggle/input/weight-QACT/pytorch_model.bin")

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    main(args)
