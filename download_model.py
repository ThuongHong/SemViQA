from huggingface_hub import snapshot_download


repos = [
    ("nguyenvulebinh/vi-mrc-large", "./nguyenvulebinh/vi-mrc-large"),
    ("FacebookAI/xlm-roberta-large", "./FacebookAI/xlm-roberta-large"),
    ("MoritzLaurer/ernie-m-large-mnli-xnli", "./MoritzLaurer/ernie-m-large-mnli-xnli"),
    ("microsoft/infoxlm-large", "./microsoft/infoxlm-large"),
    ("vinai/phobert-large", "./vinai/phobert-large"),
    # ("deepset/xlm-roberta-large-squad2", "./deepset/xlm-roberta-large-squad2"),
    # ("microsoft/deberta-v3-large", "./microsoft/deberta-v3-large"),
]

for repo_id, local_dir in repos: 
    import os
    if not os.path.exists(local_dir):
        os.makedirs(local_dir)
    snapshot_download(repo_id=repo_id, token="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", local_dir=local_dir)
