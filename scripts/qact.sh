#!/bin/bash
 
export PYTHONPATH="/kaggle/working/SemViQA:$PYTHONPATH"
echo "Starting script..."
 
python3 src/training/qact.py \
    --key_wandb "" \
    --project "Find-Evidence-UIT" \
    --tags "Full-Data" \
    --name "Find-Evidence" \
    --model_name "/kaggle/input/model-base" \
    --path_finetune_model 0 \
    --output_dir "/kaggle/working/" \
    --seed 40 \
    --logging_dir "logs" \
    --train_batch_size 12 \
    --num_train_epochs 100 \
    --gradient_accumulation_steps 2 \
    --use_8bit_adam 0 \
    --learning_rate 0.00001 \
    --max_lr 0.00003 \
    --max_grad_norm 1.0 \
    --report_to "wandb" \
    --train_data "train.csv" \
    --eval_data "test.csv" \
    --use_smoe 0 \
    --num_experts 4 \
    --num_experts_per_token 2 \
    --freeze_text_encoder 0 \
    --beta 0.1 \
    --alpha 1.0 \
    --gama 0.1 \
    --is_load_weight 0 \
    --weight_model "/kaggle/input/weight-QACT/pytorch_model.bin" \
    --weight_optimizer "/kaggle/input/weight-QACT/optimizer.bin" \
    --weight_scheduler "/kaggle/input/weight-QACT/scheduler.bin" \
    --max_iter 100 \
    --show_loss 1 \
    --is_train 1 \
    --is_eval 1 \
    --stop_threshold 0.0005

 
echo "Script completed!"
