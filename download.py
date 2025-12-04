# download.py
# Script để tải model SDXL Base 1.0 (bản FP16 tối ưu cho T4)

import os
from huggingface_hub import snapshot_download

MODEL_DIR = "./sdxl_models/base"

def download_model():
    print(f"--- Bắt đầu tải SDXL Base 1.0 (Bản tối ưu FP16) ---")
    
    snapshot_download(
        repo_id="stabilityai/stable-diffusion-xl-base-1.0",
        local_dir=MODEL_DIR,
        local_dir_use_symlinks=False,
        # Chỉ tải file cấu hình và file trọng số fp16 nhẹ
        allow_patterns=[
            "*.json", 
            "*.fp16.safetensors", 
            "*.txt", 
            "model_index.json"
        ],
        ignore_patterns=["*.bin", "*.ckpt", "*.h5", "*onnx*"] 
    )
    print("✅ Download xong. Sẵn sàng cho Colab T4!")

if __name__ == "__main__":
    download_model()