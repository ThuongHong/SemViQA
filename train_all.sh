#!/bin/bash
# pip install -r requirements.txt
# pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu118

export PYTHONPATH="SemViQA:$PYTHONPATH"
echo "Starting the training process..."

# model
# FacebookAI/xlm-roberta-large
# MoritzLaurer/ernie-m-large-mnli-xnli
# microsoft/infoxlm-large
# vinai/phobert-large

# 3 class: cross_entropy
# 2 class: focal_loss, AFM
models_classify=("FacebookAI/xlm-roberta-large" "MoritzLaurer/ernie-m-large-mnli-xnli" "microsoft/infoxlm-large" "vinai/phobert-large")
loss_types=("focal_loss")

# # 3 class for viwiki
# for model in "${models_classify[@]}"
# do 
#     model_name=$(basename "$model")
    
#     echo "Starting training for model: $model_name"
 
#     python3 src/training/classify.py \
#         --train_data "./data/classify/viwiki_train.csv" \
#         --dev_data "./data/classify/viwiki_test.csv" \
#         --model_name "./$model" \
#         --lr 1e-5 \
#         --epochs 20 \
#         --accumulation_steps 1 \
#         --batch_size 64 \
#         --max_len 256 \
#         --num_workers 2 \
#         --patience 5 \
#         --type_loss "cross_entropy" \
#         --output_dir "./${model_name}_viwiki_3class_cross" \
#         --n_classes 3\
#         --is_weighted 0

#     echo "Training completed for model: $model_name"
# done

# 3 class for isedsc
for model in "${models_classify[@]}"
do 
    model_name=$(basename "$model")

    echo "Starting training for model: $model_name"
 
    python3 src/training/classify.py \
        --train_data "./data/classify/isedsc_train.csv" \
        --dev_data "./data/classify/isedsc_test.csv" \
        --model_name "./$model" \
        --lr 1e-5 \
        --epochs 20 \
        --accumulation_steps 1 \
        --batch_size 64 \
        --max_len 256 \
        --num_workers 2 \
        --patience 5 \
        --type_loss "cross_entropy" \
        --output_dir "./${model_name}_isedsc_3class_cross" \
        --n_classes 3\
        --is_weighted 0

    echo "Training completed for model: $model_name"
done

# 2 class for viwiki
for model in "${models_classify[@]}"
do 
    model_name=$(basename "$model")
    if [ "$model" == "FacebookAI/xlm-roberta-large" ]; then
        echo "Skipping model: $model_name"
        continue
    fi
    for loss in "${loss_types[@]}"
    do
        echo "Starting training for model: $model_name with loss: $loss"

        python3 src/training/classify.py \
            --train_data "./data/classify/viwiki_train.csv" \
            --dev_data "./data/classify/viwiki_test.csv" \
            --model_name "./$model" \
            --lr 1e-5\
            --epochs 20 \
            --accumulation_steps 1 \
            --batch_size 64 \
            --max_len 256 \
            --num_workers 2 \
            --patience 5 \
            --type_loss "$loss" \
            --output_dir "./${model_name}_viwiki_2class_${loss}" \
            --n_classes 2\
            --is_weighted 0

        echo "Training completed for model: $model_name with loss: $loss"
    done
done

# 2 class for isedsc
for model in "${models_classify[@]}"
do 
    model_name=$(basename "$model")
    if [ "$model" == "FacebookAI/xlm-roberta-large" ]; then
        echo "Skipping model: $model_name"
        continue
    fi
    for loss in "${loss_types[@]}"
    do
        echo "Starting training for model: $model_name with loss: $loss"

        python3 src/training/classify.py \
            --train_data "./data/classify/isedsc_train.csv" \
            --dev_data "./data/classify/isedsc_test.csv" \
            --model_name "./$model" \
            --lr 1e-5 \
            --epochs 20 \
            --accumulation_steps 1 \
            --batch_size 64 \
            --max_len 256 \
            --num_workers 2 \
            --patience 5 \
            --type_loss "$loss" \
            --output_dir "./${model_name}_isedsc_2class_${loss}" \
            --n_classes 2\
            --is_weighted 0

        echo "Training completed for model: $model_name with loss: $loss"
    done
done


# evidence
model_evidence=("microsoft/infoxlm-large" "microsoft/deberta-v3-large" "deepset/xlm-roberta-large-squad2")

# evidence for viwiki
for model in "${model_evidence[@]}"
do 
    model_name=$(basename "$model")
    echo "Starting training for model: $model_name"
 
    python3 src/training/qact_origin.py \
        --project "viwiki" \
        --model_name "$model" \
        --output_dir "./${model_name}_viwiki_qact_origin" \
        --train_batch_size 64 \
        --num_train_epochs 20 \
        --learning_rate 2e-6 \
        --train_data  "./data/evi/viwiki_train.csv" \
        --eval_data  "./data/evi/viwiki_test.csv"

    echo "Training completed for model: $model_name"
done

# evidence for isedsc
for model in "${model_evidence[@]}"
do 
    model_name=$(basename "$model")
    echo "Starting training for model: $model_name"
 
    python3 src/training/qact_origin.py \
        --project "isedsc" \
        --model_name "./$model" \
        --output_dir "./${model_name}_isedsc_qact_origin" \
        --train_batch_size 64 \
        --num_train_epochs 20 \
        --learning_rate 2e-6 \
        --train_data  "./data/evi/isedsc_train.csv" \
        --eval_data  "./data/evi/isedsc_test.csv"

    echo "Training completed for model: $model_name"
done

# evidence for qact
python3 src/training/qact.py \
    --project "viwiki" \
    --model_name "./nguyenvulebinh/vi-mrc-large" \
    --output_dir "vi-mrc-large_viwiki_qact" \
    --train_batch_size 64 \
    --num_train_epochs 20 \
    --learning_rate 2e-6 \
    --train_data  "./data/evi/viwiki_train.csv" \
    --eval_data  "./data/evi/viwiki_test.csv"

python3 src/training/qact.py \
    --project "isedsc" \
    --model_name "./nguyenvulebinh/vi-mrc-large" \
    --output_dir "vi-mrc-large_isedsc_qact" \
    --train_batch_size 64 \
    --num_train_epochs 20 \
    --learning_rate 2e-6 \
    --train_data  "./data/evi/isedsc_train.csv" \
    --eval_data  "./data/evi/isedsc_test.csv"