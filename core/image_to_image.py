# core/image_to_image.py
from pathlib import Path
from PIL import Image
import torch
import uuid
from core.models import load_img2img
from config import IMAGES_DIR, IMG2IMG_STRENGTH, GUIDANCE_SCALE

def img2img_edit(
    init_image_path: str, 
    prompt: str, 
    negative_prompt: str = "low quality, blurry",
    strength=IMG2IMG_STRENGTH, 
    guidance=GUIDANCE_SCALE, 
    seed=None, 
    num_inference_steps=20,  # <--- ĐÃ THÊM: Để khớp với tasks.py
    mask_path: str|None=None
):
    # 1. Load Pipeline
    pipe = load_img2img()
    
    # 2. Xử lý ảnh đầu vào
    try:
        init = Image.open(init_image_path).convert('RGB')
        # Resize ảnh về kích thước chia hết cho 8 (tránh lỗi Dimension Error)
        init = init.resize((init.width // 8 * 8, init.height // 8 * 8))
        
        mask = None
        if mask_path:
            mask = Image.open(mask_path).convert('L')
            mask = mask.resize((init.width, init.height))
    except Exception as e:
        print(f"❌ Error loading input image: {e}")
        raise e

    # 3. Xử lý Seed
    generator = None
    if seed is not None:
        generator = torch.Generator(device=pipe.device).manual_seed(int(seed))

    print(f"[IMG2IMG] Processing: '{prompt[:30]}...' | Strength: {strength} | Steps: {num_inference_steps}")

    # 4. Generate (Trong chế độ tiết kiệm RAM)
    with torch.inference_mode():
        if mask:
            # Nếu có mask (Inpaint mode - Cần pipeline inpaint nhưng ở đây dùng tạm img2img)
            # Lưu ý: img2img pipeline chuẩn không nhận mask, code này demo giả định pipeline hỗ trợ
            out = pipe(
                prompt=prompt, 
                negative_prompt=negative_prompt,
                image=init, 
                mask_image=mask,
                strength=strength, 
                guidance_scale=guidance, 
                generator=generator,
                num_inference_steps=num_inference_steps
            )
        else:
            # Chế độ thường
            out = pipe(
                prompt=prompt, 
                negative_prompt=negative_prompt,
                image=init, 
                strength=strength, 
                guidance_scale=guidance, 
                generator=generator,
                num_inference_steps=num_inference_steps
            )

    # 5. Lưu ảnh
    img = out.images[0]
    out_path = Path(IMAGES_DIR) / f"img2img_{uuid.uuid4().hex}.png"
    img.save(out_path)
    return str(out_path)