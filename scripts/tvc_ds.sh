#!/usr/bin/env bash
nvidia-smi
source ~/.bashrc

# Log GPU utilization in background
while true; do nvidia-smi > gpu_status.txt; sleep 5; done &

export PYTHONPATH="/kaggle/working/SemViQA:$PYTHONPATH"

cd /kaggle/working/SemViQA
# ---------------------------
# Training parameters
# ---------------------------
BS=6            # Effective per-device batch size
EPOCHS=9        # Number of training epochs
LR=2e-6          # Learning rate
MODEL_NAME="microsoft/infoxlm-large"  # Base model (change if desired)
OUTPUT_DIR="./output_deepspeed"

# Launch training with 🤗 Accelerate + DeepSpeed (Stage-2)
accelerate launch \
    --multi_gpu \
    --num_processes $(nvidia-smi -L | wc -l) \
    -m semviqa.tvc.main_ds \
    --train_data "/kaggle/input/semviqa-data/data/classify/viwiki_train.csv" \
    --dev_data "/kaggle/input/semviqa-data/data/classify/viwiki_test.csv" \
    --model_name $MODEL_NAME \
    --lr 3e-5 \
    --epochs $EPOCHS \
    --accumulation_steps 1 \
    --train_batch_size $BS \
    --max_len 256 \
    --num_workers 2 \
    --patience 5 \
    --type_loss "ce" \
    --output_dir "./output" \
    --n_classes 3\
    --is_pretrained 0

echo "DeepSpeed TVC training completed!"