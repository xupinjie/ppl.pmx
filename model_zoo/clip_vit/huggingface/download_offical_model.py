from huggingface_hub import snapshot_download

snapshot_download(repo_id="openai/clip-vit-large-patch14", allow_patterns=["*.json", "pytorch_model.bin", "vocab.txt"], local_dir="../datas/hf")
