#!/bin/bash
#SBATCH --job-name=test_slurm # define job name
#SBATCH --nodes=1             # define node
#SBATCH --gpus-per-node=1     # define gpu limmit in 1 node
#SBATCH --ntasks=1            # define number tasks
#SBATCH --cpus-per-task=24    # There are 24 CPU cores
#SBATCH --time=4-00:10:00     # Max running time = 10 minutes
#aSBATCH --mem=1000
#aSBATCH --mem-per-gpu=80000
#SBATCH --nodelist=node001
nvidia-smi
source ~/.bashrc
#rm -rf /data2/cmdir/home/test04/.conda/envs/sem_deep
#conda env list
#echo "Setup conda env"
#conda create --name sem_deep python=3.10 -y

while true; do nvidia-smi > gpu_status.txt; sleep 5; done &

module load shared conda
. $CONDAINIT
conda activate sem_deep

conda info --envs
echo "Conda Environment: $CONDA_DEFAULT_ENV"

export PYTHONPATH="SemViQA:$PYTHONPATH"

#conda env list
#echo "Install torch"
#pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu118
#pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu118
#echo "Install other requirements"
#pip install -r requirements.txt
#echo "Done setup environment"

#export PYTHONPATH = "/data2/cmdir/home/test04/test_slurm/thanhtt/SemViQA:$PYTHONPATH"
#pip install sentencepiece==0.2.0

echo "Starting the training process..."

# model
# FacebookAI/xlm-roberta-large
# MoritzLaurer/ernie-m-large-mnli-xnli
# microsoft/infoxlm-large
# vinai/phobert-large

BS=48
models_classify=("FacebookAI/xlm-roberta-large" "MoritzLaurer/ernie-m-large-mnli-xnli" "microsoft/infoxlm-large" "vinai/phobert-large")
loss_types=("focal_loss")

for model in "${models_classify[@]}"
do 
    model_name=$(basename "$model")

    echo "Starting training for model: $model_name"
 
    python3 src/training/classify.py \
        --train_data "./data/classify/isedsc_train.csv" \
        --dev_data "./data/classify/isedsc_test.csv" \
        --model_name "$model" \
        --lr 1e-5 \
        --epochs 20 \
        --accumulation_steps 1 \
        --batch_size $BS \
        --max_len 256 \
        --num_workers 2 \
        --patience 3 \
        --type_loss "focal_loss" \
        --output_dir "./${model_name}_isedsc_2class_focal" \
        --n_classes 2\
        --is_weighted 1

    echo "Training completed for model: $model_name"
done

echo "Starting training for model: $model_name"
 
    python3 src/training/classify.py \
        --train_data "./data/classify/viwiki_train.csv" \
        --dev_data "./data/classify/viwiki_test.csv" \
        --model_name "FacebookAI/xlm-roberta-large" \
        --lr 1e-5 \
        --epochs 20 \
        --accumulation_steps 1 \
        --batch_size $BS \
        --max_len 256 \
        --num_workers 2 \
        --patience 3 \
        --type_loss "focal_loss" \
        --output_dir "./xlm-roberta-large_viwiki_2class_focal" \
        --n_classes 2\
        --is_weighted 1

    echo "Training completed for model: $model_name"

