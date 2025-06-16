nvidia-smi
source ~/.bashrc

# Log GPU utilization in background
while true; do nvidia-smi > gpu_status.txt; sleep 5; done &

# Activate conda environment (edit if different)
module load shared conda
. $CONDAINIT
conda activate sem_deep

conda info --envs
echo "Conda Environment: $CONDA_DEFAULT_ENV"

export PYTHONPATH="SemViQA:$PYTHONPATH"

# ---------------------------
# Training parameters
# ---------------------------
BS=12            # Effective per-device batch size
EPOCHS=16        # Number of training epochs
LR=2e-6          # Learning rate
MODEL_NAME="microsoft/infoxlm-large"  # Base model (change if desired)
OUTPUT_DIR="./output_deepspeed"
DS_CONFIG="SemViQA/semviqa/ser/ds_zero2.json"

# Launch training with 🤗 Accelerate + DeepSpeed (Stage-2)
accelerate launch \
    --multi_gpu \
    --num_processes $(nvidia-smi -L | wc -l) \
    --module semviqa.ser.main \
    # SemViQA/semviqa/ser/main.py \
        --model_name "$MODEL_NAME" \
        --output_dir "$OUTPUT_DIR" \
        --train_batch_size $BS \
        --num_train_epochs $EPOCHS \
        --learning_rate $LR \
        --train_data "train.csv" \
        --eval_data "test.csv" \
        --patience 5 \
        --is_pretrained 0 \
        --ds_config "$DS_CONFIG"

echo "DeepSpeed QATC training completed!" 