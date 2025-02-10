from huggingface_hub import HfApi, upload_folder

HF_TOKEN = "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"
models = ["xlm-roberta-large_viwiki_3class_cross", "ernie-m-large-mnli-xnli_viwiki_3class_cross", "infoxlm-large_viwiki_3class_cross","phobert-large_viwiki_3class_cross",
          "xlm-roberta-large_isedsc_3class_cross", "ernie-m-large-mnli-xnli_isedsc_3class_cross", "infoxlm-large_isedsc_3class_cross","phobert-large_isedsc_3class_cross",
          "ernie-m-large-mnli-xnli_viwiki_2class_focal_loss", "infoxlm-large_viwiki_2class_focal_loss","phobert-large_viwiki_2class_focal_loss",
          "ernie-m-large-mnli-xnli_isedsc_2class_focal_loss", "infoxlm-large_isedsc_2class_focal_loss","phobert-large_isedsc_2class_focal_loss"]

repo_base = "xuandin"


api = HfApi(token=HF_TOKEN)

for model_name in models: 
    if "/" in model_name:
        model_name = model_name.split("/")[-1]
    repo_id = f"{repo_base}/{model_name}"
    
    api.create_repo(repo_id=repo_id, exist_ok=True, private=True)
    
    upload_folder(
        folder_path=model_name,
        path_in_repo="",
        repo_id=repo_id,
        commit_message=f"Upload {model_name}",
        token=HF_TOKEN,
    )
    
    print(f"✅ Đã đẩy {model_name} lên {repo_id} (Private)")
