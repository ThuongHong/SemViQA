nvidia-smi
source ~/.bashrc

while true; do nvidia-smi > gpu_status.txt; sleep 5; done &

module load shared conda
. $CONDAINIT
conda activate sem_deep

conda info --envs
echo "Conda Environment: $CONDA_DEFAULT_ENV"

export PYTHONPATH="SemViQA:$PYTHONPATH"
echo "Starting the inference process..."

python3 semviqa/pipeline.py \
    --data_path "data/test.json" \
    --output_path "output.json" \
    --model_evidence_QA "SemViQA/qatc-infoxlm-viwikifc" \
    --model_2_class "SemViQA/bc-infoxlm-viwikifc" \
    --model_3_class "SemViQA/tc-infoxlm-viwikifc" \
    --thres_evidence 0.5
 
echo "Inference process completed!"
