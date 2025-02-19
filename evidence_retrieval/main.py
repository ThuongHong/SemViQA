import logging
import math
import os
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
import gc
from sklearn.metrics import f1_score, roc_auc_score, cohen_kappa_score, matthews_corrcoef
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from tqdm import tqdm
from transformers import default_data_collator, AutoModelForQuestionAnswering
from torch.utils.data import DataLoader
from accelerate import notebook_launcher
import wandb
from safetensors.torch import load_model
import argparse
from torch.utils.tensorboard import SummaryWriter
import time

from model.qatc import QATC
from model.SMOE.moe import MoELayer
from data_utils import load_data
from loss import comboLoss

os.environ["WANDB__SERVICE_WAIT"] = "300"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total number of parameters: {total_params}")
    active_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total number of active (trainable) parameters: {active_params}")

def load_models(args):
    model = QATC(config=args)  
    if args.is_load_weight:
        print("loading train weight")
        if "safetensors" in args.weight_model:
            load_model(model, f"{args.weight_model}")
        else:
            model.load_state_dict(torch.load(args.weight_model), strict=False)
    
    count_parameters(model)
    return model


def setting_optimizer(config):
    # Initialize the optimizer
    if config.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "Please install bitsandbytes to use 8-bit Adam. You can do so by running `pip install bitsandbytes`"
            )
        optimizer_cls = bnb.optim.AdamW8bit
    else:
        optimizer_cls = torch.optim.AdamW
    return optimizer_cls

def main(args):
    logger = get_logger(__name__, log_level="INFO")

    # WandB login
    # wandb.login(key=args.key_wandb)
    # wandb.init()

    logging_dir = os.path.join(args.output_dir, args.logging_dir)
    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)

    accelerator = Accelerator(
        log_with=args.report_to,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        project_config=accelerator_project_config
    )
    
    # accelerator.init_trackers(project_name=args.project)
    # accelerator.init_trackers(
    #     project_name=args.project,
    #     init_kwargs={"wandb": {"entity": "xuandienk4", 'tags': args.tags, 'name': f"{args.name} - {'smoe' if args.use_smoe else 'not_smoe'}"}}
    # )
    
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)

    if args.seed is not None:
        set_seed(args.seed)

    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

    model = load_models(args)
    model = accelerator.prepare(model)
    
    optimizer_cls = setting_optimizer(config=args)
    optimizer = optimizer_cls(
        model.parameters(),
        lr=args.learning_rate,
    )
    if args.is_load_weight:
        print("loading optimizer weight")
        optimizer.load_state_dict(torch.load(args.weight_optimizer))

    train_dataset, test_dataset = load_data(args)
    train_dataloader = DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=default_data_collator,
        batch_size=args.train_batch_size,
    ) 

    eval_dataloader = DataLoader(
        test_dataset,
        collate_fn=default_data_collator,
        batch_size=args.train_batch_size
    )

    lr_scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=args.learning_rate, max_lr=args.max_lr, cycle_momentum=False)

    if args.is_load_weight: 
        print("loading scheduler weight")
        lr_scheduler.load_state_dict(torch.load(args.weight_scheduler))

    if args.path_finetune_model:
        print(f"Update weight {args.path_finetune_model}")
        accelerator.load_state(args.path_finetune_model)
        torch.cuda.empty_cache()
        gc.collect()

    optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        optimizer, train_dataloader, lr_scheduler
    )

    initial_global_step = 0
    progress_bar = tqdm(
        initial=initial_global_step,
        desc="Steps",
        disable=not accelerator.is_local_main_process,
    )

    loss_fn = comboLoss(args)
    min_loss = 1000000
    prev_train_loss = None
    best_acc = 0.0
    iter = 1
    print("Running training")
    global_step = 1
    info_epoch = {}
    start_time = time.time()
    logs = []
 
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    # writer = SummaryWriter(log_dir=logging_dir)

    # remove cuda cache
    torch.cuda.empty_cache()
    # Training
    for epoch in range(args.num_train_epochs):
        model.train()
        train_loss = 0.0 
        start_train_time = time.time()
        info_epoch[epoch] = {}
        for step, batch in enumerate(train_dataloader):
            for key in batch:
                batch[key] = batch[key].to(accelerator.device)
            Tagging = batch['Tagging'].to(torch.float32)
            pt, start_logits, end_logits = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'])

            output = {
                "attention_mask": batch['attention_mask'],
                "start_logits": start_logits,
                "end_logits": end_logits,
                "start_positions": batch['start_positions'],
                "end_positions": batch['end_positions'],
                "Tagging": Tagging, 
                "pt": pt,
            }
            
            # Calculate loss
            loss, loss_base, reg_loss, retation_tagg_loss = loss_fn(output=output)

            balance_loss = 0.0
            if args.use_smoe:
                for layer in model.model.encoder.layer:
                    if isinstance(layer.attention.output, MoELayer):
                        balance_loss += layer.attention.output.last_balance_loss
                loss += args.gama*balance_loss
            avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean()
            train_loss += avg_loss.item() / args.gradient_accumulation_steps

            iter += 1
            accelerator.backward(loss)
            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                
            if global_step % args.gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                lr_scheduler.step()
            
            logs = {"step": f"{step}/{len(train_dataloader)}", "step_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)
            global_step +=1


            if global_step % args.max_iter == 0:
                train_loss = round(train_loss / args.max_iter, 4)

                # writer.add_scalar('Train/Loss', train_loss, global_step)
                # writer.add_scalar('Train/global_step', global_step, global_step)
                # writer.add_scalar('Train/loss_base', loss_base, global_step)
                # writer.add_scalar('Train/retation_tagg_loss', retation_tagg_loss, global_step)
                # accelerator.log(
                #     {
                #         'global_step': global_step,
                #         'Train loss': train_loss,
                #         'Base loss': loss_base,
                #         'retation tagg loss': retation_tagg_loss,
                #         "Balance loss": balance_loss if args.use_smoe else -1,
                #     },
                #     step=global_step
                # )

                # if train_loss < min_loss:
                #     save_path = os.path.join(args.output_dir, f"best {args.name}")
                #     if not os.path.exists(save_path):
                #         os.makedirs(save_path)
                #     accelerator.save_state(save_path)
                #     min_loss = train_loss
                #     print("Save model")
                print({
                        'global_step': global_step,
                        'Train loss': train_loss,
                        'Base loss': loss_base,
                        'retation tagg loss': retation_tagg_loss,
                        'reg_loss': reg_loss,
                        "Balance loss": balance_loss if args.use_smoe else -1,
                        "epoch": epoch,
                    })
                train_loss = 0.0
        train_loss = round(train_loss / len(train_dataloader), 4) 
        epoch_training_time = time.time() - start_train_time
        info_train = f"Epoch {epoch+1} - epoch_training_time: {epoch_training_time} - Train Loss: {train_loss} - Base Loss: {loss_base} - Retation Tagg Loss: {retation_tagg_loss} - Balance Loss: {balance_loss if args.use_smoe else -1}"
        print(info_train)
        logs.append(info_train)

        info_epoch[epoch] = {
            "Time train": epoch_training_time,
            'Train loss': train_loss,
            'Base loss': loss_base,
            'retation tagg loss': retation_tagg_loss,
            "Balance loss": balance_loss if args.use_smoe else -1,
        }

        if torch.isnan(torch.tensor(train_loss)):
            print(f"Stopping training as train loss is NaN at epoch {epoch}.")
            break

        # Evaluation
        model.eval()
        eval_loss = 0.0
        predictions = []
        true_positions = []
        start_eval_time = time.time()
        cnt = 0

        for step, batch in enumerate(eval_dataloader):
            with torch.no_grad():
                for key in batch:
                    batch[key] = batch[key].to("cuda")
                Tagging = batch['Tagging'].to(torch.float32)
                pt, start_logits, end_logits = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'])

                output = {
                    "attention_mask": batch['attention_mask'],
                    "start_logits": start_logits,
                    "end_logits": end_logits,
                    "start_positions": batch['start_positions'],
                    "end_positions": batch['end_positions'],
                    "Tagging": Tagging,
                    "pt": pt
                }

                loss, _, _,_ = loss_fn(output=output)
                eval_loss += loss.item()

                start_logits = start_logits.cpu().detach()
                end_logits = end_logits.cpu().detach()

                start_preds = torch.argmax(start_logits, axis=1)
                end_preds = torch.argmax(end_logits, axis=1)

                predictions.extend(list(zip(start_preds, end_preds)))
                true_positions.extend(list(zip(batch['start_positions'].flatten().tolist(), batch['end_positions'].flatten().tolist())))
                
            progress_bar.update(1)
        
        eval_loss = eval_loss / len(eval_dataloader)
        predictions = np.array(predictions)
        true_positions = np.array(true_positions)

        acc = (np.all(true_positions == predictions, axis=1)).astype(int)
        accuracy = np.sum(acc) / len(acc)
        

        start_preds_flat = [start for start, end in predictions]
        end_preds_flat = [end for start, end in predictions]
        start_true_flat = [start for start, end in true_positions]
        end_true_flat = [end for start, end in true_positions]
        try:
            eval_f1 = (f1_score(start_true_flat, start_preds_flat, average='weighted') + f1_score(end_true_flat, end_preds_flat, average='weighted')) / 2
        except ValueError:
            eval_f1 = "N/A"
        try:
            eval_kappa = (cohen_kappa_score(start_true_flat, start_preds_flat) + cohen_kappa_score(end_true_flat, end_preds_flat)) / 2
        except ValueError:
            eval_kappa = "N/A"
        try:
            eval_matthews_corrcoef = (matthews_corrcoef(start_true_flat, start_preds_flat) + matthews_corrcoef(end_true_flat, end_preds_flat)) / 2
        except ValueError:
            eval_matthews_corrcoef = "N/A" 
            

        if accuracy > best_acc:
            save_path = os.path.join(args.output_dir, f"best {args.name}")
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            accelerator.save_state(save_path)
            best_acc = accuracy
            print("Save model best acc at epoch", epoch)
            cnt = 0
        else:
            cnt += 1
        if cnt == args.patience:
            print(f"Early stopping at epoch {epoch}.")
            break
        epoch_eval_time = time.time() - start_eval_time
        info_eval = f"Epoch {epoch+1} - epoch_eval_time: {epoch_eval_time} - Accuracy: {accuracy} - Eval Loss: {eval_loss} - F1: {eval_f1} - Kappa: {eval_kappa} - Matthews Corrcoef: {eval_matthews_corrcoef}"
        print(info_eval)
        logs.append(info_eval)

        info_epoch[epoch]= {
            "Time eval": epoch_eval_time,
            'Eval loss': eval_loss,
            'Accuracy eval': accuracy,
            'F1': eval_f1,
            'Kappa': eval_kappa,
            'Matthews Corrcoef': eval_matthews_corrcoef
        }

        # save info_epoch to json
        df = pd.DataFrame.from_dict(info_epoch, orient='index')
        df.to_json(os.path.join(args.output_dir, 'info_epoch.json'))

        # accelerator.log({"Accuracy eval": accuracy, "Eval loss": eval_loss}, step=global_step)
        # writer.add_scalar('Eval/Accuracy', accuracy, global_step)
        # writer.add_scalar('Eval/Loss', eval_loss, global_step)
    accelerator.end_training()
    # writer.close()

    # save tensorboard logs
    

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate VNFASHIONDIFF")
    
    # W&B related arguments
    parser.add_argument("--key_wandb", type=str, default="", help="Wandb API Key")
    parser.add_argument("--project", type=str, default="Find-Evidence-Retrieval", help="Wandb project name")
    parser.add_argument("--tags", type=str, default="Full-Data", help="Wandb tags")
    parser.add_argument("--name", type=str, default="Find-Evidence", help="Wandb run name")
    
    # Model and training related arguments
    parser.add_argument("--model_name", type=str, default="/kaggle/input/model-base", help="Path to the base model")
    parser.add_argument("--path_finetune_model", type=int, default=0, help="Path to fine-tuned model")
    parser.add_argument("--output_dir", type=str, default="/kaggle/working/", help="Output directory")
    parser.add_argument("--seed", type=int, default=40, help="Random seed")
    parser.add_argument("--logging_dir", type=str, default="logs", help="Logging directory")
    parser.add_argument("--train_batch_size", type=int, default=12, help="Training batch size")
    parser.add_argument("--num_train_epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--num_epoch_eval", type=int, default=1, help="Number of epochs to evaluate")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2, help="Gradient accumulation steps")
    parser.add_argument("--use_8bit_adam", type=int, default=None, help="Use 8-bit Adam optimizer")
    parser.add_argument("--learning_rate", type=float, default=0.00001, help="Learning rate")
    parser.add_argument("--max_lr", type=float, default=0.00003, help="Maximum learning rate")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Maximum gradient norm")
    parser.add_argument("--report_to", type=str, default="wandb", help="Reporting destination (e.g., 'wandb')")
    
    # Data related arguments
    parser.add_argument("--train_data", type=str, default='train.csv', help="Path to training data")
    parser.add_argument("--eval_data", type=str, default='test.csv', help="Path to evaluation data")
    
    # Mixture of Experts (MoE) related arguments
    parser.add_argument("--use_smoe", type=int, default=0, help="Use SMOE instead of MOE")
    parser.add_argument("--num_experts", type=int, default=4, help="Number of experts in MoE")
    parser.add_argument("--num_experts_per_token", type=int, default=2, help="Number of experts per token in MoE")

    # Others
    parser.add_argument("--freeze_text_encoder", type=int, default=0, help="Whether to freeze the text encoder")
    parser.add_argument("--beta", type=float, default=0.1, help="Beta value for MoE")
    parser.add_argument("--alpha", type=float, default=1, help="Alpha value for MoE")
    parser.add_argument("--gama", type=float, default=0.1, help="Gama value for MoE")
    parser.add_argument("--is_load_weight", type=int, default=0, help="Load weights from pre-trained model")
    parser.add_argument("--weight_model", type=str, default="/kaggle/input/weight-QACT/pytorch_model.bin", help="Path to model weights")
    parser.add_argument("--weight_optimizer", type=str, default="/kaggle/input/weight-QACT/optimizer.bin", help="Path to optimizer weights")
    parser.add_argument("--weight_scheduler", type=str, default="/kaggle/input/weight-QACT/scheduler.bin", help="Path to scheduler weights")
    parser.add_argument("--max_iter", type=int, default=100, help="Maximum number of iterations")
    parser.add_argument("--show_loss", type=int, default=1, help="Whether to display loss during training")
    parser.add_argument("--is_train", type=int, default=1, help="Set to True if training")
    parser.add_argument("--is_eval", type=int, default=1, help="Set to Eval")
    parser.add_argument("--stop_threshold", type=float, default=0.0005, help="Stop threshold")

    parser.add_argument("--patience", type=int, default=5, help="Patience for early stopping")
    args = parser.parse_args()
    return args
if __name__ == "__main__":
    args = parse_args()
    main(args)