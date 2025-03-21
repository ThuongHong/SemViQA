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
# If you want to fine-tune a pre-trained model, run the following script.
# type_loss = "ce" or "focal"
# tc
# model_name get from https://huggingface.co/collections/SemViQA/tc-67db9b88ba7abc78559edb4a
python3 -m semviqa.tvc.main \
    --train_data "train.csv" \
    --dev_data "test.csv" \
    --model_name "SemViQA/tc-infoxlm-viwikifc" \
    --lr 3e-5 \
    --epochs 20 \
    --accumulation_steps 1 \
    --batch_size $BS \
    --max_len 256 \
    --num_workers 2 \
    --patience 5 \
    --type_loss "ce" \
    --output_dir "./output" \
    --n_classes 3\
    --is_pretrained 1\

# If you want to fine-tune an untrained model, run the following script.
# model_name = "microsoft/infoxlm-large", "FacebookAI/xlm-roberta-large" or "MoritzLaurer/ernie-m-large-mnli-xnli", ...
# type_loss = "ce" or "focal"
# tc
python3 -m semviqa.tvc.main \
    --train_data "train.csv" \
    --dev_data "test.csv" \
    --model_name "microsoft/infoxlm-large" \
    --lr 3e-5 \
    --epochs 20 \
    --accumulation_steps 1 \
    --batch_size $BS \
    --max_len 256 \
    --num_workers 2 \
    --patience 5 \
    --type_loss "ce" \
    --output_dir "./output" \
    --n_classes 3\
    --is_pretrained 0\

echo "Training script completed!"
