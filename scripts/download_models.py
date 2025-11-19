# scripts/download_models.py
import os
from huggingface_hub import snapshot_download

BASE_DIR = "models"

MODELS_MAP = {
    "sdxl": "stabilityai/stable-diffusion-xl-base-1.0",
    "sdxl-img2img": "stabilityai/stable-diffusion-xl-refiner-1.0",
    "controlnet-canny": "diffusers/controlnet-canny-sdxl-1.0",
    "controlnet-depth": "diffusers/controlnet-depth-sdxl-1.0",
    "controlnet-openpose": "thibaud/controlnet-openpose-sdxl-1.0",
}

# ✅ Lấy file cấu hình và file fp16
ALLOW_PATTERNS = [
    "*.json",
    "*.txt",
    "*.model",
    "*.fp16.safetensors", # Chỉ lấy file trọng số có đuôi fp16
    "*.fp16.bin",
]

# ⛔ CHẶN TRIỆT ĐỂ (Sử dụng **/ để tìm sâu trong thư mục con)
IGNORE_PATTERNS = [
    "*.ckpt", "*.h5", "*.msgpack",
    # Chặn file root khổng lồ
    "sd_xl_base_1.0.safetensors",      
    "sd_xl_base_1.0_0.9vae.safetensors",
    "sd_xl_refiner_1.0.safetensors",
    
    # 🔥 KHẮC PHỤC VẤN ĐỀ 20GB Ở ĐÂY:
    # Chặn file UNet bản gốc (FP32) nằm trong folder "unet/"
    "**/diffusion_pytorch_model.safetensors", 
    
    # Chặn file Text Encoder bản gốc (FP32) nằm trong folder "text_encoder/"
    "**/model.safetensors",
]

def ensure_dirs():
    if not os.path.exists(BASE_DIR):
        os.makedirs(BASE_DIR)

def download_model(folder_name, repo_id):
    local_path = os.path.join(BASE_DIR, folder_name)
    print(f"\n🚀 Processing: {folder_name}")
    
    # Nếu muốn tải lại từ đầu để test dung lượng, hãy xóa folder cũ bằng tay hoặc uncomment dòng dưới
    # import shutil; shutil.rmtree(local_path, ignore_errors=True)

    snapshot_download(
        repo_id=repo_id,
        local_dir=local_path,
        local_dir_use_symlinks=False,
        revision="main",
        allow_patterns=ALLOW_PATTERNS,
        ignore_patterns=IGNORE_PATTERNS, # <-- Logic chặn mới nằm ở đây
    )
    print(f"   ✅ [DONE] Saved to {local_path}")

if __name__ == "__main__":
    ensure_dirs()
    for folder, repo in MODELS_MAP.items():
        try:
            download_model(folder, repo)
        except Exception as e:
            print(f"Error downloading {folder}: {e}")