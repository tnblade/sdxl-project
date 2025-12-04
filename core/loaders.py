#   core/loaders.py
# Đây là "thủ kho". Nhiệm vụ duy nhất là Load Model. Sau này sẽ thêm hàm load_lora, load_controlnet vào class này cực kỳ gọn gàng.

import torch
from diffusers import AutoPipelineForText2Image, AutoPipelineForImage2Image
from .config import Config

class ModelLoader:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.pipeline = None
        self.current_type = None # "txt2img" hoặc "img2img"

    def load_base_pipeline(self, task_type="txt2img"):
        """Load hoặc chuyển đổi pipeline giữa các chế độ"""
        
        # 1. Load mới nếu chưa có
        if self.pipeline is None:
            model_id = Config.get_model_path()
            print(f"📥 Loading Base Model from: {model_id}...")
            
            self.pipeline = AutoPipelineForText2Image.from_pretrained(
                model_id, 
                torch_dtype=torch.float16, 
                variant="fp16", 
                use_safetensors=True
            )
            # Tối ưu cho Colab T4
            self.pipeline.enable_model_cpu_offload()
            print("✅ Model loaded.")

        # 2. Chuyển đổi (Switching) mà không load lại RAM
        if task_type == "img2img" and self.current_type != "img2img":
            self.pipeline = AutoPipelineForImage2Image.from_pipe(self.pipeline)
            self.current_type = "img2img"
            
        elif task_type == "txt2img" and self.current_type != "txt2img":
            self.pipeline = AutoPipelineForText2Image.from_pipe(self.pipeline)
            self.current_type = "txt2img"
            
        return self.pipeline

    # --- Sau này thêm tính năng ở đây ---
    def load_lora_weights(self, lora_path):
        if self.pipeline:
            print(f"Loading LoRA from {lora_path}")
            self.pipeline.load_lora_weights(lora_path)
            
    def unload_lora(self):
        if self.pipeline:
            self.pipeline.unload_lora_weights()
            
    # --- THÊM 2 HÀM NÀY CHO BENCHMARK ---
    def load_lora(self, lora_path, adapter_name="default"):
        """Nạp LoRA vào pipeline đang chạy"""
        if self.pipeline:
            print(f"🔄 Loading LoRA adapter: {lora_path}")
            self.pipeline.load_lora_weights(lora_path, adapter_name=adapter_name)
            self.pipeline.fuse_lora() # Kết hợp weights để chạy nhanh hơn
            print("✅ LoRA Loaded & Fused.")

    def unload_lora(self):
        """Gỡ bỏ LoRA để quay về model gốc"""
        if self.pipeline:
            print(f"🔄 Unloading LoRA...")
            self.pipeline.unfuse_lora()
            self.pipeline.unload_lora_weights()
            print("✅ LoRA Unloaded.")