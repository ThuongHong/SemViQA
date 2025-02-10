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

BS=36
model_evidence_qact=("microsoft/infoxlm-large" "nguyenvulebinh/vi-mrc-large")
# origin
for model in "${model_evidence_qact[@]}"
do
    model_name=$(basename "$model")
    echo "Starting training for model: $model_name"

    python src/training/qact_origin.py \
        --project "viwiki" \
        --model_name "./$model" \
        --output_dir "./${model_name}_viwiki_evidence_origin" \
        --train_batch_size $BS \
        --num_train_epochs 20 \
        --learning_rate 2e-6 \
        --train_data  "./data/evi/viwiki_train.csv" \
        --eval_data  "./data/evi/viwiki_test.csv"\
        --patience 5

    echo "Training completed for model: $model_name"
done

# evidence for isedsc
for model in "${model_evidence_qact[@]}"
do
    model_name=$(basename "$model")
    echo "Starting training for model: $model_name"

    python src/training/qact_origin.py \
        --project "isedsc" \
        --model_name "./$model" \
        --output_dir "./${model_name}_isedsc_evidence_origin" \
        --train_batch_size $BS \
        --num_train_epochs 20 \
        --learning_rate 2e-6 \
        --train_data  "./data/evi/isedsc_train.csv" \
        --eval_data  "./data/evi/isedsc_test.csv"\
        --patience 5

    echo "Training completed for model: $model_name"
done
# qatc
for model in "${model_evidence_qact[@]}"
do
    model_name=$(basename "$model")
    echo "Starting training for model: $model_name"

    python src/training/qact.py \
        --project "viwiki" \
        --model_name "./$model" \
        --output_dir "./${model_name}_viwiki_qact" \
        --train_batch_size $BS \
        --num_train_epochs 20 \
        --learning_rate 2e-6 \
        --train_data  "./data/evi/viwiki_train.csv" \
        --eval_data  "./data/evi/viwiki_test.csv"\
        --patience 5

    echo "Training completed for model: $model_name"
done

# evidence for isedsc
for model in "${model_evidence_qact[@]}"
do
    model_name=$(basename "$model")
    echo "Starting training for model: $model_name"

    python src/training/qact.py \
        --project "isedsc" \
        --model_name "./$model" \
        --output_dir "./${model_name}_isedsc_qact" \
        --train_batch_size $BS \
        --num_train_epochs 20 \
        --learning_rate 2e-6 \
        --train_data  "./data/evi/isedsc_train.csv" \
        --eval_data  "./data/evi/isedsc_test.csv"\
        --patience 5

    echo "Training completed for model: $model_name"
done