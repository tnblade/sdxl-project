# fine_tuning/lora.py
# Script khởi chạy quá trình fine-tuning LoRA cho SDXL 
# Sử dụng script chuẩn từ thư viện Diffusers của HuggingFace với một số cấu hình tối ưu cho T4 


import os
import subprocess
import argparse
import sys
import torch
import yaml
from accelerate.utils import write_basic_config

# --- Hack đường dẫn để import Config ---
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from core.config import Config

# --- SỬA LỖI VERSION: Dùng phiên bản script khớp với Diffusers 0.34.0 ---
# Thay vì dùng 'main' (luôn thay đổi), ta dùng tag 'v0.34.0' để ổn định
SCRIPT_URL = "https://raw.githubusercontent.com/huggingface/diffusers/v0.34.0/examples/text_to_image/train_text_to_image_lora_sdxl.py"
SCRIPT_NAME = "train_lora_sdxl_script.py"
ACCELERATE_CONFIG_FILE = "accelerate_config.yaml"

def download_script():
    """Tải script training chuẩn từ HuggingFace"""
    # Xóa script cũ nếu có để đảm bảo tải bản mới đúng version
    if os.path.exists(SCRIPT_NAME):
        # Kiểm tra xem file hiện tại có phải là bản đúng không, nếu nghi ngờ xóa tải lại
        # Ở đây ta xóa luôn cho chắc ăn
        print("♻️ Đang làm mới script training để khớp phiên bản...")
        os.remove(SCRIPT_NAME)

    print(f"⏳ [LoRA] Đang tải script chuẩn (v0.34.0)...")
    try:
        subprocess.run(["wget", "-q", SCRIPT_URL, "-O", SCRIPT_NAME], check=True)
        print("✅ Đã tải xong script.")
    except Exception as e:
        print(f"❌ Lỗi tải script: {e}")
        print("⚠️ Đang thử link dự phòng (Main branch)...")
        # Link dự phòng nếu bản v0.34.0 bị lỗi
        fallback_url = "https://raw.githubusercontent.com/huggingface/diffusers/main/examples/text_to_image/train_text_to_image_lora_sdxl.py"
        subprocess.run(["wget", "-q", fallback_url, "-O", SCRIPT_NAME], check=True)

def create_accelerate_config():
    """Tạo file cấu hình accelerate dựa trên số lượng GPU thực tế"""
    gpu_count = torch.cuda.device_count()
    print(f"🚀 Phát hiện phần cứng: {gpu_count} GPU")
    
    config_dict = {
        "compute_environment": "LOCAL_MACHINE",
        "mixed_precision": "fp16",
        "distributed_type": "NO" if gpu_count <= 1 else "MULTI_GPU",
        "num_machines": 1,
        "num_processes": gpu_count,
        "use_cpu": False,
    }
    
    with open(ACCELERATE_CONFIG_FILE, "w") as f:
        yaml.dump(config_dict, f)
    
    return ACCELERATE_CONFIG_FILE

def run_lora_training(data_dir, output_dir, prompt, base_model_path):
    if output_dir is None:
        output_dir = "output_lora_result"

    # --- 1. XỬ LÝ MODEL PATH ---
    if base_model_path.endswith(".safetensors"):
        print(f"⚠️ CẢNH BÁO: Chuyển sang dùng Repo gốc StabilityAI để tránh lỗi device.")
        train_model_path = "stabilityai/stable-diffusion-xl-base-1.0"
    else:
        train_model_path = base_model_path

    # --- 2. TẠO CONFIG ---
    config_file = create_accelerate_config()

    # --- 3. LỆNH CHẠY ---
    cmd = [
        "accelerate", "launch",
        "--config_file", config_file,
        SCRIPT_NAME
    ]
    
# Các tham số training tối ưu (Đã chỉnh sửa để chạy nhanh)
    args = [
        f"--pretrained_model_name_or_path={train_model_path}",
        f"--train_data_dir={data_dir}",
        "--caption_column=text",
        "--resolution=1024",
        "--random_flip",
        "--train_batch_size=1",
        
        # --- CẤU HÌNH TỐC ĐỘ CAO (FAST TRAINING) ---
        "--num_train_epochs=4",             # Giảm từ 10 xuống 4 (đủ cho 500 ảnh)
        "--gradient_accumulation_steps=4",  # Tăng tốc độ học (gom 4 bước làm 1)
        # -------------------------------------------
        
        "--checkpointing_steps=500",
        "--learning_rate=1e-4",
        "--lr_scheduler=constant",
        "--lr_warmup_steps=0",
        "--mixed_precision=fp16",
        "--seed=42",
        f"--output_dir={output_dir}",
        f"--validation_prompt={prompt}",
        "--gradient_checkpointing", 
        "--use_8bit_adam",          
        "--report_to=tensorboard",
        "--logging_dir=logs"
    ]
    cmd.extend(args)

    print(f"\n⚡ Lệnh thực thi: {' '.join(cmd)}")
    
    try:
        subprocess.run(cmd, check=True)
        print(f"\n✅ [LoRA] Training hoàn tất! File tại: {output_dir}/pytorch_lora_weights.safetensors")
    except subprocess.CalledProcessError as e:
        print(f"\n❌ [LoRA] Lỗi trong quá trình train.")
        # Mẹo fix lỗi thư viện
        print("💡 Gợi ý: Nếu lỗi 'ImportError', hãy thử chạy lệnh: pip install -U git+https://github.com/huggingface/diffusers.git")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True, help="Folder ảnh train")
    parser.add_argument("--prompt", type=str, required=True, help="Prompt kích hoạt")
    parser.add_argument("--output_dir", type=str, default=None, help="Folder lưu kết quả")
    parser.add_argument("--base_model", type=str, default=None, help="Đường dẫn Base Model")
    
    args = parser.parse_args()
    
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    download_script()
    
    if args.base_model:
        final_model_path = args.base_model
    else:
        final_model_path = Config.get_model_path()
    
    run_lora_training(args.data_dir, args.output_dir, args.prompt, final_model_path)