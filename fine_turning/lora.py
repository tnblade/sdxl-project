import os
import subprocess
import argparse
import sys

# Đường dẫn đến script chuẩn của Diffusers (chúng ta sẽ tải về máy để dùng)
SCRIPT_URL = "https://raw.githubusercontent.com/huggingface/diffusers/main/examples/text_to_image/train_text_to_image_lora_sdxl.py"
SCRIPT_NAME = "train_text_to_image_lora_sdxl.py"

def download_script():
    """Tải script training chuẩn từ HuggingFace nếu chưa có"""
    if not os.path.exists(SCRIPT_NAME):
        print(f"⏳ Đang tải script training chuẩn: {SCRIPT_NAME}...")
        try:
            subprocess.run(["wget", "-q", SCRIPT_URL, "-O", SCRIPT_NAME], check=True)
            print("✅ Đã tải xong script.")
        except Exception as e:
            print(f"❌ Lỗi tải script: {e}")
            sys.exit(1)
    else:
        print(f"✅ Đã tìm thấy script: {SCRIPT_NAME}")

def run_training(data_dir, output_dir, prompt, base_model_path):
    """Chạy lệnh training với cấu hình tối ưu cho Google Colab T4"""
    
    # Cấu hình "bí thuật" cho T4 GPU (15GB VRAM)
    # Chúng ta dùng accelerate launch để quản lý bộ nhớ
    cmd = [
        "accelerate", "launch", SCRIPT_NAME,
        f"--pretrained_model_name_or_path={base_model_path}",
        f"--train_data_dir={data_dir}",
        "--caption_column=text",
        "--resolution=1024",
        "--random_flip",
        "--train_batch_size=1",           # Batch size 1 để tiết kiệm VRAM
        "--num_train_epochs=10",          # Train 10 vòng (có thể sửa)
        "--checkpointing_steps=500",
        "--learning_rate=1e-4",
        "--lr_scheduler=constant",
        "--lr_warmup_steps=0",
        "--mixed_precision=fp16",         # Chạy FP16 cho nhẹ
        "--seed=42",
        f"--output_dir={output_dir}",
        f"--validation_prompt={prompt}",  # Tự động test thử với prompt này
        "--gradient_checkpointing",       # QUAN TRỌNG: Tiết kiệm VRAM
        "--use_8bit_adam"                 # QUAN TRỌNG: Optimizer 8-bit siêu nhẹ
    ]

    print("\n🚀 BẮT ĐẦU TRAINING VỚI CẤU HÌNH TỐI ƯU...")
    print("Command:", " ".join(cmd))
    
    try:
        subprocess.run(cmd, check=True)
        print(f"\n✅ Training hoàn tất! File LoRA nằm tại: {output_dir}/pytorch_lora_weights.safetensors")
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Có lỗi xảy ra trong quá trình train: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SDXL LoRA Trainer Launcher")
    parser.add_argument("--data_dir", type=str, required=True, help="Thư mục chứa ảnh training")
    parser.add_argument("--prompt", type=str, required=True, help="Prompt mô tả đối tượng (VD: 'a photo of sks dog')")
    
    args = parser.parse_args()
    
    # 1. Chuyển vào thư mục fine_tuning để chạy cho gọn
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    # 2. Tải script gốc
    download_script()
    
    # 3. Xác định đường dẫn model gốc (từ Task 1)
    # Lưu ý: Script đang chạy trong folder fine_tuning nên phải lùi ra 1 cấp (..)
    base_model = "../sdxl_models/base"
    output_dir = "../lora_output"
    
    # 4. Chạy training
    run_training(args.data_dir, output_dir, args.prompt, base_model)