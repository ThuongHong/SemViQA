
# SemViQA: Semantic Question Answering System for fact-checking Vietnamese information


## Clone repo
```bash
git clone https://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@github.com/DAVID-NGUYEN-S16/SemViQA.git
cd SemViQA
```  
## Chuẩn bị data
unzip file data.zip 

## Download models
```bash
python3 download_model.py
```

## Cài đặt các thư viện cần thiết
```bash
pip install transformers==4.42.3
pip install torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

## Train model
```bash
bash train_all.sh
```
