nvidia-smi
source ~/.bashrc

while true; do nvidia-smi > gpu_status.txt; sleep 5; done &

module load shared conda
. $CONDAINIT
conda activate sem_deep

conda info --envs
echo "Conda Environment: $CONDA_DEFAULT_ENV"

export PYTHONPATH="SemViQA:$PYTHONPATH"
echo "Running training script..."

BS=36
python3 semviqa/ser/main.py \
    --project "isedsc" \
    --model_name "microsoft/infoxlm-large" \
    --output_dir "./infoxlm-large_isedsc_qact" \
    --train_batch_size $BS \
    --num_train_epochs 16 \
    --learning_rate 2e-6 \
    --train_data  "train.csv" \
    --eval_data  "test.csv"\
    --patience 5 \
    --is_pretrained 1\

echo "Training completed for model"