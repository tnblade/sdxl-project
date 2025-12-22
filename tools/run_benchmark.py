# tools/run_benchmark.py
# Tool tự động sinh ảnh benchmark và chấm điểm mô hình Stable Diffusion XL

import os
import sys
import json
import torch
import pandas as pd
from diffusers import StableDiffusionXLPipeline, AutoencoderKL
from PIL import Image
from tqdm import tqdm

# --- IMPORT MODULE CỦA BẠN ---
# Thêm đường dẫn project vào hệ thống để import được core
sys.path.append('/kaggle/working/sdxl-project')
try:
    from core.scorer import ImageScorer
except ImportError:
    print("❌ Không tìm thấy file core/scorer.py! Hãy kiểm tra lại cấu trúc thư mục.")
    sys.exit(1)

# --- CẤU HÌNH ---
PROMPT_FILE = "benchmark_prompts.json"
OUTPUT_ROOT = "benchmark_results"
BASE_MODEL = "/kaggle/input/stable-diffusion-xl/pytorch/base-1-0/1/sd_xl_base_1.0.safetensors"

# Đường dẫn LoRA của bạn (Tự động tìm)
LORA_DIR = "/kaggle/working/sdxl-project/fine_tuning/lora"

def generate_images(pipe, prompts, folder_name, use_lora_path=None):
    print(f"\\n🎨 Đang sinh ảnh cho: {folder_name}...")
    save_dir = os.path.join(OUTPUT_ROOT, folder_name)
    os.makedirs(save_dir, exist_ok=True)
    
    # Nạp LoRA nếu cần
    if use_lora_path:
        pipe.load_lora_weights(use_lora_path)
        print(f"   -> Đã nạp LoRA: {os.path.basename(use_lora_path)}")
    else:
        pipe.unload_lora_weights()
        print("   -> Chế độ Base (Không LoRA)")

    generated_count = 0
    for i, prompt in enumerate(prompts):
        # Kiểm tra nếu ảnh đã có rồi thì bỏ qua (để chạy lại không mất time)
        img_path = os.path.join(save_dir, f"test_{i}.png")
        txt_path = os.path.join(save_dir, f"test_{i}.txt")
        
        if not os.path.exists(img_path):
            image = pipe(prompt, num_inference_steps=30, height=1024, width=1024).images[0]
            image.save(img_path)
            # Lưu prompt kèm theo để chấm điểm
            with open(txt_path, "w") as f:
                f.write(prompt)
            generated_count += 1
            
    print(f"✅ Đã xong {folder_name} ({generated_count} ảnh mới).")

def main():
    # 1. SETUP MODEL GENERATION
    print("🚀 Khởi động Benchmark...")
    vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)
    pipe = StableDiffusionXLPipeline.from_single_file(
        BASE_MODEL, vae=vae, torch_dtype=torch.float16
    ).to("cuda")
    
    # Load đề thi
    with open(PROMPT_FILE, "r") as f:
        prompts = json.load(f)

    # --- A. SINH ẢNH MODEL GỐC (BASE) ---
    generate_images(pipe, prompts, "1_SDXL_Base_Original", use_lora_path=None)

    # --- B. SINH ẢNH MODEL CỦA BẠN (LORA) ---
    # Tìm file LoRA Character
    lora_char = os.path.join(LORA_DIR, "char_v1.safetensors")
    if os.path.exists(lora_char):
        generate_images(pipe, prompts, "2_My_Model_Character", use_lora_path=lora_char)
    else:
        print("⚠️ Không tìm thấy LoRA Character để test.")

    # Tìm file LoRA Scenery
    lora_scene = os.path.join(LORA_DIR, "scene_v1.safetensors")
    if os.path.exists(lora_scene):
        generate_images(pipe, prompts, "3_My_Model_Scenery", use_lora_path=lora_scene)
    
    # Giải phóng VRAM để nhường chỗ cho model chấm điểm
    del pipe
    del vae
    torch.cuda.empty_cache()
    
    # --- C. CHẤM ĐIỂM (DÙNG CORE/SCORER.PY) ---
    print("\\n📊 BẮT ĐẦU CHẤM ĐIỂM (Scoring)...")
    scorer = ImageScorer() # Class của bạn
    
    results = []
    
    # Quét tất cả thư mục trong benchmark_results
    if not os.path.exists(OUTPUT_ROOT): os.makedirs(OUTPUT_ROOT)
    
    model_folders = sorted(os.listdir(OUTPUT_ROOT))
    
    for model_name in model_folders:
        model_path = os.path.join(OUTPUT_ROOT, model_name)
        if not os.path.isdir(model_path): continue
        
        print(f"   Evaluating: {model_name}")
        images = [f for f in os.listdir(model_path) if f.endswith(".png")]
        
        total_clip = 0
        total_aes = 0
        count = 0
        
        for img_file in tqdm(images):
            img_full_path = os.path.join(model_path, img_file)
            txt_full_path = img_full_path.replace(".png", ".txt")
            
            # Đọc prompt tương ứng của ảnh
            if os.path.exists(txt_full_path):
                with open(txt_full_path, "r") as f:
                    prompt_text = f.read()
                
                # Mở ảnh và chấm điểm
                pil_image = Image.open(img_full_path).convert("RGB")
                c_score, a_score = scorer.get_scores(pil_image, prompt_text)
                
                total_clip += c_score
                total_aes += a_score
                count += 1
        
        if count > 0:
            results.append({
                "Model Name": model_name,
                "CLIP Score (Độ hiểu)": round(total_clip / count, 2),
                "Aesthetic Score (Độ đẹp)": round(total_aes / count, 2),
                "Số ảnh": count
            })

    # --- D. XUẤT BÁO CÁO ---
    if results:
        df = pd.DataFrame(results)
        print("\\n🏆 KẾT QUẢ BENCHMARK:")
        print(df.to_markdown(index=False))
        df.to_csv("benchmark_report.csv", index=False)
    else:
        print("❌ Không có kết quả nào được tạo.")

if __name__ == "__main__":
    main()