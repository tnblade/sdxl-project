# core/text_to_image.py
import uuid
import hashlib
import torch
from pathlib import Path
from core.models import load_sdxl_text
from config import IMAGES_DIR, DEFAULT_WIDTH, DEFAULT_HEIGHT, GUIDANCE_SCALE

# Tạo thư mục cache nếu chưa có
CACHE_DIR = Path(IMAGES_DIR) / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Cập nhật hàm Hash để bao gồm cả steps (nếu steps thay đổi thì tạo ảnh mới, không dùng cache cũ)
def _prompt_hash(prompt: str, width: int, height: int, guidance: float, seed: int|None, steps: int):
    s = f"{prompt}|{width}|{height}|{guidance}|{seed}|{steps}"
    return hashlib.sha256(s.encode('utf-8')).hexdigest()

def generate_text_image(
    prompt: str, 
    negative_prompt: str = "low quality, blurry", 
    width=DEFAULT_WIDTH, 
    height=DEFAULT_HEIGHT, 
    guidance=GUIDANCE_SCALE, 
    seed=None, 
    num_inference_steps=20,  # <--- ĐÃ THÊM THAM SỐ NÀY ĐỂ FIX LỖI
    use_cache=True
):
    # 1. Check Cache
    h = _prompt_hash(prompt, width, height, guidance, seed, num_inference_steps)
    outf = CACHE_DIR / f"{h}.png"
    
    if use_cache and outf.exists():
        print(f"[CACHE HIT] Returning existing image for: {h}")
        return str(outf)

    # 2. Load Pipeline
    pipe = load_sdxl_text()
    
    # 3. Xử lý Seed
    generator = None
    if seed is not None:
        generator = torch.Generator(device=pipe.device).manual_seed(int(seed))

    print(f"[GEN] Processing: '{prompt[:30]}...' | Size: {width}x{height} | Steps: {num_inference_steps}")
    
    # 4. Generate
    try:
        with torch.inference_mode():
            out = pipe(
                prompt=prompt, 
                negative_prompt=negative_prompt,
                width=width, 
                height=height, 
                guidance_scale=guidance, 
                generator=generator,
                num_inference_steps=num_inference_steps # <--- ĐÃ TRUYỀN VÀO MODEL
            )
        
        # 5. Save Result
        img = out.images[0]
        img.save(outf)
        return str(outf)
        
    except Exception as e:
        print(f"❌ Error during generation: {e}")
        raise e