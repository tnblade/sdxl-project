# fine_tuning/lora.py
# Script khởi chạy quá trình fine-tuning LoRA cho SDXL 
# Sử dụng script chuẩn từ thư viện Diffusers của HuggingFace với một số cấu hình tối ưu cho T4 


import os
import subprocess
import argparse
import sys
import torch
from accelerate.utils import write_basic_config

# --- Hack đường dẫn để import Config ---
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from core.config import Config

SCRIPT_URL = "https://raw.githubusercontent.com/huggingface/diffusers/main/examples/text_to_image/train_text_to_image_lora_sdxl.py"
SCRIPT_NAME = "train_lora_sdxl_script.py"

def download_script():
    """Tải script training chuẩn"""
    if not os.path.exists(SCRIPT_NAME):
        print(f"⏳ [LoRA] Đang tải script chuẩn từ HuggingFace...")
        try:
            subprocess.run(["wget", "-q", SCRIPT_URL, "-O", SCRIPT_NAME], check=True)
        except Exception as e:
            print(f"❌ Lỗi tải script: {e}")
            sys.exit(1)

def run_lora_training(data_dir, output_dir, prompt, base_model_path):
    if output_dir is None:
        output_dir = "output_lora_result"

    # --- 1. XỬ LÝ MODEL PATH (FIX LỖI DEVICE MISMATCH) ---
    # Script training chuẩn KHÔNG hỗ trợ file .safetensors đơn lẻ tốt.
    # Nếu phát hiện input là file đơn, ta buộc phải dùng repo gốc trên HuggingFace
    # để đảm bảo script tải đúng cấu trúc thư mục (UNet/VAE/TextEncoder) về GPU.
    if base_model_path.endswith(".safetensors"):
        print(f"⚠️ CẢNH BÁO: Script training không hỗ trợ trực tiếp file đơn (.safetensors).")
        print(f"🔄 Đang chuyển sang dùng Repo gốc: stabilityai/stable-diffusion-xl-base-1.0")
        print(f"   (Việc này giúp tránh lỗi 'Expected all tensors to be on the same device')")
        train_model_path = "stabilityai/stable-diffusion-xl-base-1.0"
    else:
        train_model_path = base_model_path

    # --- 2. CẤU HÌNH MULTI-GPU (TỰ ĐỘNG) ---
    # Kiểm tra số lượng GPU
    gpu_count = torch.cuda.device_count()
    print(f"🚀 Phát hiện {gpu_count} GPU.")
    
    # Tạo config mặc định cho accelerate (tránh lỗi chưa config)
    write_basic_config(mixed_precision="fp16")

    cmd = ["accelerate", "launch"]

    # Nếu có nhiều GPU, thêm tham số để chạy song song (Nhanh gấp đôi)
    if gpu_count > 1:
        print("🔥 Kích hoạt chế độ Multi-GPU Training!")
        cmd.extend([
            "--multi_gpu",
            f"--num_processes={gpu_count}"
        ])

    # Thêm script và các tham số
    cmd.append(SCRIPT_NAME)
    
    # Các tham số training tối ưu
    args = [
        f"--pretrained_model_name_or_path={train_model_path}",
        f"--train_data_dir={data_dir}",
        "--caption_column=text",
        "--resolution=1024",
        "--random_flip",
        "--train_batch_size=1",
        "--num_train_epochs=10",
        "--checkpointing_steps=500",
        "--learning_rate=1e-4",
        "--lr_scheduler=constant",
        "--lr_warmup_steps=0",
        "--mixed_precision=fp16",
        "--seed=42",
        f"--output_dir={output_dir}",
        f"--validation_prompt={prompt}",
        
        # --- CÁC THAM SỐ TỐI ƯU BỘ NHỚ QUAN TRỌNG ---
        "--gradient_checkpointing", # Tiết kiệm VRAM
        "--use_8bit_adam",          # Optimizer nhẹ
        "--report_to=tensorboard",
        "--logging_dir=logs"
    ]
    
    # Nếu dùng repo HF, ta cần preload model vào cache để tránh lỗi timeout khi chạy multi-process
    # Nhưng accelerate usually handles this.
    
    cmd.extend(args)

    print(f"\nexecuting command: {' '.join(cmd)}")
    print(f"📂 Model Training: {train_model_path}")
    
    try:
        subprocess.run(cmd, check=True)
        print(f"\n✅ [LoRA] Training hoàn tất! File tại: {output_dir}/pytorch_lora_weights.safetensors")
    except subprocess.CalledProcessError as e:
        print(f"\n❌ [LoRA] Lỗi trong quá trình train. Hãy kiểm tra log phía trên.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True, help="Folder ảnh train")
    parser.add_argument("--prompt", type=str, required=True, help="Prompt kích hoạt")
    parser.add_argument("--output_dir", type=str, default=None, help="Folder lưu kết quả")
    parser.add_argument("--base_model", type=str, default=None, help="Đường dẫn Base Model")
    
    args = parser.parse_args()
    
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    download_script()
    
    # Lấy path từ Config nếu không truyền vào
    if args.base_model:
        final_model_path = args.base_model
    else:
        final_model_path = Config.get_model_path()
    
    run_lora_training(args.data_dir, args.output_dir, args.prompt, final_model_path)