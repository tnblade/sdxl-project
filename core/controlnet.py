# core/controlnet.py
from functools import lru_cache
import torch
import uuid
from pathlib import Path
from PIL import Image
from config import SDXL_TEXT, DEVICE, TORCH_DTYPE, IMAGES_DIR, DEFAULT_WIDTH, DEFAULT_HEIGHT
from core.utils import image_to_canny  # <--- Import hàm xử lý ảnh

# Setup Cache thư mục
CACHE_DIR = Path(IMAGES_DIR) / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

DTYPE = torch.float16 if TORCH_DTYPE == 'float16' else torch.float32

@lru_cache()
def get_controlnet_pipe(control_type):
    """Load pipeline dựa trên loại ControlNet"""
    print(f"⏳ Loading ControlNet: {control_type}...")
    from diffusers import ControlNetModel, StableDiffusionControlNetPipeline, StableDiffusionXLControlNetPipeline
    
    # Mapping model ID
    model_map = {
        "canny": "lllyasviel/sd-controlnet-canny", 
        "depth": "lllyasviel/sd-controlnet-depth",
        "openpose": "lllyasviel/sd-controlnet-openpose",
    }
    
    # Tự động chọn class Pipeline
    PipelineClass = StableDiffusionControlNetPipeline 
    if "xl" in SDXL_TEXT:
         PipelineClass = StableDiffusionXLControlNetPipeline

    cn_id = model_map.get(control_type, "lllyasviel/sd-controlnet-canny")
    
    # Load ControlNet
    cn = ControlNetModel.from_pretrained(cn_id, torch_dtype=DTYPE, use_safetensors=True)
    
    # Load Pipeline chính
    pipe = PipelineClass.from_pretrained(
        SDXL_TEXT, controlnet=cn, torch_dtype=DTYPE, use_safetensors=True, low_cpu_mem_usage=True
    )
    
    if DEVICE != "cpu":
        pipe.to(DEVICE)
    return pipe

def generate_controlnet_image(
    prompt, 
    control_image_path, 
    control_type="canny", 
    negative_prompt="low quality", 
    width=DEFAULT_WIDTH, 
    height=DEFAULT_HEIGHT, 
    steps=20, 
    seed=None,
    guidance=7.5
):
    # 1. Load Pipeline
    pipe = get_controlnet_pipe(control_type)
    
    # 2. Xử lý ảnh đầu vào (Pre-processing)
    if control_type == "canny":
        # Nếu là Canny, tự động chuyển ảnh thường thành ảnh nét đứt
        print("⚙️ Converting image to Canny edges...")
        image = image_to_canny(control_image_path)
    else:
        # Các loại khác (Depth/OpenPose) cứ load tạm ảnh gốc
        image = Image.open(control_image_path).convert("RGB")
    
    # Resize ảnh về đúng kích thước yêu cầu
    image = image.resize((width, height))
    
    # 3. Seed
    generator = None
    if seed is not None and int(seed) != -1:
        generator = torch.Generator(device=DEVICE).manual_seed(int(seed))

    print(f"[CONTROLNET] Type: {control_type} | Steps: {steps}")
    
    # 4. Generate
    with torch.inference_mode():
        out = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=image,
            width=width,
            height=height,
            num_inference_steps=steps,
            guidance_scale=guidance,
            generator=generator
        )
    
    # 5. Lưu ảnh
    out_path = Path(IMAGES_DIR) / f"cn_{control_type}_{uuid.uuid4().hex}.png"
    out.images[0].save(out_path)
    return str(out_path)