#!/bin/bash

export PYTHONPATH="/kaggle/working/SemViQA:$PYTHONPATH"
echo "Running training script..."

python3 src/training/bc.py \
    --train_data "data/train.csv" \
    --dev_data "data/dev.csv" \
    --model_name "MoritzLaurer/ernie-m-large-mnli-xnli" \
    --lr 1e-5 \
    --alpha 0.25 \
    --gamma 2.0 \
    --reduction "mean" \
    --epochs 10 \
    --accumulation_steps 2 \
    --batch_size 2 \
    --max_len 256 \
    --num_workers 2 \
    --patience 5 \
    --plt_loss "loss.png" \
    --plt_acc "acc.png"

echo "Training script completed!"
