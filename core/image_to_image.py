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
    num_inference_steps=20,
    mask_path: str|None=None
):
    # 1. Load Pipeline
    pipe = load_img2img()
    
    # 2. Xử lý ảnh đầu vào
    try:
        init = Image.open(init_image_path).convert('RGB')
        # Resize ảnh mịn hơn
        init = init.resize((init.width // 8 * 8, init.height // 8 * 8), Image.LANCZOS)
        
        mask = None
        if mask_path:
            mask = Image.open(mask_path).convert('L')
            mask = mask.resize((init.width, init.height))
    except Exception as e:
        print(f"❌ Error loading input image: {e}")
        raise e

    # 3. Xử lý Seed
    generator = None
    if seed is not None and int(seed) != -1:
        generator = torch.Generator(device=pipe.device).manual_seed(int(seed))

    print(f"[IMG2IMG] Processing: '{prompt[:30]}...' | Strength: {strength} | Steps: {num_inference_steps}")

    # 4. Generate
    with torch.inference_mode():
        if mask:
            # Logic Inpaint (nếu pipeline hỗ trợ)
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
            # Logic Img2Img thường
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