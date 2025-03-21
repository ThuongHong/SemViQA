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

BS=104 
python3 semviqa.tvc.main \
    --train_data "train.csv" \
    --dev_data "test.csv" \
    --model_name "FacebookAI/xlm-roberta-large" \
    --lr 3e-5 \
    --epochs 20 \
    --accumulation_steps 1 \
    --batch_size $BS \
    --max_len 256 \
    --num_workers 2 \
    --patience 5 \
    --type_loss "ce" \
    --output_dir "./xlm-roberta-large_viwiki_2class_focal" \
    --n_classes 3\
    --is_pretrained 1\

echo "Training script completed!"
