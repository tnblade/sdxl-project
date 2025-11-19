# test_generation.py
import os
import torch
from core.models import load_sdxl_text
from diffusers.utils import load_image

# 1. Load Model (Hệ thống sẽ dùng cache từ core/models.py mà ta vừa sửa)
print("⏳ Loading SDXL Model from local path...")
try:
    pipe = load_sdxl_text()
    print("✅ Model loaded successfully on CUDA!")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    print("💡 Gợi ý: Kiểm tra lại đường dẫn trong file .env xem đã đúng folder models/sdxl chưa?")
    exit()

# 2. Cấu hình Prompt
prompt = "A futuristic city in Vietnam with flying motorbikes, cyberpunk style, neon lights, 8k resolution, highly detailed"
negative_prompt = "low quality, bad anatomy, blurry, pixelated"

print(f"\n🎨 Generating image for: '{prompt}'")

# 3. Chạy model
# Lưu ý: Lần chạy đầu tiên sẽ mất vài giây để làm nóng GPU
with torch.inference_mode():
    image = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        width=1024,
        height=1024,
        guidance_scale=7.5,
        num_inference_steps=3 # SDXL chuẩn thường là 30-50 bước
    ).images[0]

# 4. Lưu ảnh
output_path = "test_result.png"
image.save(output_path)
print(f"🎉 Success! Image saved to: {os.path.abspath(output_path)}")