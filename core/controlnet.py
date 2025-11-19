# core/controlnet.py
from functools import lru_cache
import torch
from config import SDXL_TEXT, DEVICE, TORCH_DTYPE

# Xác định kiểu dữ liệu (float32 cho CPU, float16 cho GPU)
DTYPE = torch.float16 if TORCH_DTYPE == 'float16' else torch.float32

@lru_cache()
def build_controlnet_pipeline(base_model_id: str, controlnet_ids: list):
    print(f"⏳ Loading ControlNet Pipeline: Base={base_model_id} + CN={controlnet_ids}")
    
    from diffusers import ControlNetModel
    
    # 1. Load các model ControlNet con (Dùng low_cpu_mem_usage để tránh crash)
    cn_models = []
    for cid in controlnet_ids:
        try:
            cn = ControlNetModel.from_pretrained(
                cid, 
                torch_dtype=DTYPE, 
                use_safetensors=True,
                low_cpu_mem_usage=True  # <--- QUAN TRỌNG CHO RAM 16GB
            )
            cn_models.append(cn)
        except Exception as e:
            print(f"❌ Failed to load ControlNet {cid}: {e}")
            raise e

    # 2. Xác định Pipeline Class (SDXL hay SD 1.5)
    if "xl" in base_model_id.lower() or "sdxl" in base_model_id.lower():
        from diffusers import StableDiffusionXLControlNetPipeline
        PipelineClass = StableDiffusionXLControlNetPipeline
    else:
        from diffusers import StableDiffusionControlNetPipeline
        PipelineClass = StableDiffusionControlNetPipeline

    # 3. Load Pipeline chính
    pipe = PipelineClass.from_pretrained(
        base_model_id, 
        controlnet=cn_models, 
        torch_dtype=DTYPE,
        use_safetensors=True,
        low_cpu_mem_usage=True # <--- QUAN TRỌNG
    )
    
    if DEVICE != "cpu":
        pipe = pipe.to(DEVICE)
    
    # Tối ưu bộ nhớ
    pipe.enable_attention_slicing()
    
    # Chỉ bật CPU offload nếu có GPU (để tiết kiệm VRAM)
    if DEVICE != "cpu":
        try:
            pipe.enable_model_cpu_offload()
        except:
            pass
            
    return pipe

# Helpers
def sdxl_with_canny():
    from config import CONTROLNET_CANNY
    return build_controlnet_pipeline(SDXL_TEXT, [CONTROLNET_CANNY])

def sdxl_with_depth():
    from config import CONTROLNET_DEPTH
    return build_controlnet_pipeline(SDXL_TEXT, [CONTROLNET_DEPTH])

def sdxl_with_openpose():
    from config import CONTROLNET_OPENPOSE
    return build_controlnet_pipeline(SDXL_TEXT, [CONTROLNET_OPENPOSE])