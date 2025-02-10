#!/bin/bash

export PYTHONPATH="/kaggle/working/SemViQA:$PYTHONPATH"
echo "Starting the training process..."

python3 src/training/classify.py \
    --train_data "data/train.csv" \
    --dev_data "data/dev.csv" \
    --model_name "MoritzLaurer/ernie-m-large-mnli-xnli" \
    --lr 1e-5 \
    --epochs 10 \
    --accumulation_steps 2 \
    --batch_size 8 \
    --max_len 256 \
    --num_workers 2 \
    --patience 5 \
    --type_loss "cross_entropy" \ 
    --output_dir "" \
    --n_classes 3\
    

echo "Training process completed!"
