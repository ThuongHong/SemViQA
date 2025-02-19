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
 
DATA_PATH="data/test.json"
OUTPUT_PATH="output.json"
MODEL_EVIDENCE_QA="QATC"
WEIGHT_EVIDENCE_QA="weights/QATC.pth"
MODEL_2_CLASS="2_class"
WEIGHT_2_CLASS="weights/2_class.pth"
MODEL_3_CLASS="3_class"
WEIGHT_3_CLASS="weights/3_class.pth"
THRES_EVIDENCE=0.5
 
python3 pipelines/main.py \
    --data_path $DATA_PATH \
    --output_path $OUTPUT_PATH \
    --model_evidence_QA $MODEL_EVIDENCE_QA \
    --weight_evidence_QA $WEIGHT_EVIDENCE_QA \
    --model_2_class $MODEL_2_CLASS \
    --weight_2_class $WEIGHT_2_CLASS \
    --model_3_class $MODEL_3_CLASS \
    --weight_3_class $WEIGHT_3_CLASS \
    --thres_evidence $THRES_EVIDENCE
 
echo "Inference process completed!"
