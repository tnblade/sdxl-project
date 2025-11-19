# core/models.py
import torch
from functools import lru_cache
import logging
from config import (
    SDXL_TEXT, SDXL_IMG2IMG, CONTROLNET_CANNY, 
    CONTROLNET_DEPTH, CONTROLNET_OPENPOSE, CAPTION_MODEL, 
    DEVICE, TORCH_DTYPE
)

# CPU bắt buộc phải dùng float32
RUN_DTYPE = torch.float32
# Nhưng file model tải về là fp16
LOAD_VARIANT = "fp16"

@lru_cache(maxsize=2)
def load_sdxl_text(model_id=SDXL_TEXT, device=DEVICE, dtype=RUN_DTYPE):
    # Tự động kiểm tra xem đang dùng SDXL hay SD 1.5
    if "xl" in model_id.lower() or "sdxl" in model_id.lower():
        from diffusers import StableDiffusionXLPipeline
        PipelineClass = StableDiffusionXLPipeline
    else:
        # Nếu là SD 1.5 thì dùng pipeline thường
        from diffusers import StableDiffusionPipeline
        PipelineClass = StableDiffusionPipeline
        print("💡 Detected SD 1.5 Model (Lighter version)")

    print(f"⏳ Loading Model to {device}...")

    try:
        pipe = PipelineClass.from_pretrained(
            model_id, 
            torch_dtype=dtype,       
            variant=LOAD_VARIANT,    
            use_safetensors=True,
            low_cpu_mem_usage=True # Vẫn giữ dòng này cho nhẹ
        )
    except:
        # Fallback cho SD 1.5 nếu không có file fp16
        pipe = PipelineClass.from_pretrained(
            model_id, 
            torch_dtype=dtype,       
            use_safetensors=True,
            low_cpu_mem_usage=True
        )

    if device != "cpu":
        pipe = pipe.to(device)

    return pipe


@lru_cache(maxsize=2)
def load_img2img(model_id=SDXL_IMG2IMG, device=DEVICE, dtype=RUN_DTYPE):
    from diffusers import StableDiffusionImg2ImgPipeline
    
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        model_id, 
        torch_dtype=dtype,
        variant=LOAD_VARIANT,
        use_safetensors=True,
        low_cpu_mem_usage=True,
    )
    
    if device != "cpu":
        pipe = pipe.to(device)
        try:
            pipe.enable_model_cpu_offload()
        except:
            pass
            
    pipe.enable_attention_slicing()
    return pipe


@lru_cache(maxsize=4)
def load_controlnet(controlnet_id, device=DEVICE, dtype=RUN_DTYPE):
    from diffusers import ControlNetModel
    try:
        cn = ControlNetModel.from_pretrained(
            controlnet_id, 
            torch_dtype=dtype,
            variant=LOAD_VARIANT, 
            use_safetensors=True
        )
    except OSError:
        cn = ControlNetModel.from_pretrained(
            controlnet_id, 
            torch_dtype=dtype,
            use_safetensors=True,
            low_cpu_mem_usage=True,
        )
    # ControlNet load thẳng vào device được
    if device != "cpu":
        cn = cn.to(device)
        
    return cn
