from huggingface_hub import HfApi, upload_folder
import os

HF_TOKEN = "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"
# models = ["xlm-roberta-large_viwiki_3class_cross", "ernie-m-large-mnli-xnli_viwiki_3class_cross", "infoxlm-large_viwiki_3class_cross","phobert-large_viwiki_3class_cross",
#           "xlm-roberta-large_isedsc_3class_cross", "ernie-m-large-mnli-xnli_isedsc_3class_cross", "infoxlm-large_isedsc_3class_cross","phobert-large_isedsc_3class_cross",
#           "ernie-m-large-mnli-xnli_viwiki_2class_focal_loss", "infoxlm-large_viwiki_2class_focal_loss","phobert-large_viwiki_2class_focal_loss",
#           "ernie-m-large-mnli-xnli_isedsc_2class_focal_loss", "infoxlm-large_isedsc_2class_focal_loss","phobert-large_isedsc_2class_focal_loss"]
models = [
    "infoxlm-large_isedsc_3class_cross1"
]

repo_base = "xuandin"
api = HfApi(token=HF_TOKEN)

for model_name in models:
    # if "/" in model_name:
    #     model_name = model_name.split("/")[-1]
    repo_id = f"{repo_base}/{model_name}"

    try:
        api.create_repo(repo_id=repo_id, exist_ok=True, private=False)
    except Exception as e:
        print(f"⚠️ Không thể tạo repo {repo_id}, lỗi: {e}. Tiếp tục với model tiếp theo...")
        continue

    try:
        upload_folder(
            folder_path=model_name,
            path_in_repo="",
            repo_id=repo_id,
            commit_message=f"Update {model_name}",
            token=HF_TOKEN,
        )
        print(f"✅ Đã cập nhật {model_name} lên {repo_id} (Public)")
    except Exception as e:
        print(f"❌ Lỗi khi tải lên {model_name}: {e}. Tiếp tục với model tiếp theo...")