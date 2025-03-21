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
# If you want to fine-tune a pre-trained model, run the following script.
python3 -m semviqa.ser.main \
    --project "isedsc" \
    --model_name "SemViQA/qatc-infoxlm-viwikifc" \
    --output_dir "./output" \
    --train_batch_size $BS \
    --num_train_epochs 16 \
    --learning_rate 2e-6 \
    --train_data  "train.csv" \
    --eval_data  "test.csv"\
    --patience 5 \
    --is_pretrained 1\

# If you want to fine-tune an untrained model, run the following script.
# model_name = "microsoft/infoxlm-large", "microsoft/infoxlm-base" or "nguyenvulebinh/vi-mrc-large", "nguyenvulebinh/vi-mrc-base"
python3 -m semviqa.ser.main \
    --project "isedsc" \
    --model_name "microsoft/infoxlm-large" \
    --output_dir "./output" \
    --train_batch_size $BS \
    --num_train_epochs 16 \
    --learning_rate 2e-6 \
    --train_data  "train.csv" \
    --eval_data  "test.csv"\
    --patience 5 \
    --is_pretrained 0\

echo "Training completed for model"