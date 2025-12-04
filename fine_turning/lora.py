# fine_turning/lora.py
# Script khởi chạy quá trình fine-tuning LoRA cho SDXL 
# Sử dụng script chuẩn từ thư viện Diffusers của HuggingFace với một số cấu hình tối ưu cho T4 


import os
import subprocess
import argparse
import sys

# --- MỚI: Hack đường dẫn để import được Config từ thư mục cha ---
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from core.config import Config
# -------------------------------------------------------------

SCRIPT_URL = "https://raw.githubusercontent.com/huggingface/diffusers/main/examples/text_to_image/train_text_to_image_lora_sdxl.py"
SCRIPT_NAME = "train_lora_sdxl_script.py"

def download_script():
    """Tải script training LoRA chuẩn từ HuggingFace"""
    if not os.path.exists(SCRIPT_NAME):
        print(f"⏳ [LoRA] Đang tải script chuẩn: {SCRIPT_NAME}...")
        try:
            subprocess.run(["wget", "-q", SCRIPT_URL, "-O", SCRIPT_NAME], check=True)
            print("✅ Đã tải xong script LoRA.")
        except Exception as e:
            print(f"❌ Lỗi tải script: {e}")
            sys.exit(1)
    else:
        print(f"✅ Đã tìm thấy script LoRA: {SCRIPT_NAME}")

def run_lora_training(data_dir, output_dir, prompt, base_model_path):
    if output_dir is None:
        output_dir = "output_lora_result"

    # In ra để kiểm tra xem nó nhận đúng model Kaggle chưa
    print(f"🛠️ Base Model Path: {base_model_path}")

    cmd = [
        "accelerate", "launch", SCRIPT_NAME,
        f"--pretrained_model_name_or_path={base_model_path}",
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
        "--gradient_checkpointing",       
        "--use_8bit_adam",
        "--report_to=tensorboard",
        "--logging_dir=logs"
    ]

    print(f"\n🚀 [LoRA] BẮT ĐẦU TRAINING...")
    try:
        subprocess.run(cmd, check=True)
        print(f"\n✅ [LoRA] Hoàn tất! File tại: {output_dir}/pytorch_lora_weights.safetensors")
    except subprocess.CalledProcessError as e:
        print(f"\n❌ [LoRA] Lỗi training: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True, help="Folder ảnh train")
    parser.add_argument("--prompt", type=str, required=True, help="Prompt kích hoạt")
    parser.add_argument("--output_dir", type=str, default=None, help="Folder lưu kết quả")
    
    # Cho phép người dùng truyền model path thủ công, nếu không thì tự lấy từ Config
    parser.add_argument("--base_model", type=str, default=None, help="Đường dẫn Base Model (Tùy chọn)")
    
    args = parser.parse_args()
    
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    download_script()
    
    # LOGIC MỚI: Tự động xác định model path
    if args.base_model:
        final_model_path = args.base_model
    else:
        final_model_path = Config.get_model_path()
    
    run_lora_training(args.data_dir, args.output_dir, args.prompt, final_model_path)