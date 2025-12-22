# core/loaders.py
# Loader cho Model và LoRA Adapters

import torch
import os
# Thêm StableDiffusionXLPipeline vào dòng import
from diffusers import AutoPipelineForText2Image, AutoPipelineForImage2Image, StableDiffusionXLPipeline
from .config import Config

class ModelLoader:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.pipeline = None
        self.current_type = None 

    def load_base_pipeline(self, task_type="txt2img"):
        
        if self.pipeline is None:
            model_id = Config.get_model_path()
            print(f"📥 Loading Base Model from: {model_id}...")
            
            # --- LOGIC ĐÃ SỬA: GỌI ĐÍCH DANH SDXL PIPELINE ---
            if str(model_id).endswith(".safetensors"):
                print("⚡ Phát hiện Single File -> Dùng StableDiffusionXLPipeline")
                # Dùng class cụ thể thay vì AutoPipeline để tránh lỗi Attribute Error
                self.pipeline = StableDiffusionXLPipeline.from_single_file(
                    model_id, 
                    torch_dtype=torch.float16, 
                    use_safetensors=True
                )
            else:
                print("☁️ Phát hiện Folder -> Dùng AutoPipeline")
                self.pipeline = AutoPipelineForText2Image.from_pretrained(
                    model_id, 
                    torch_dtype=torch.float16, 
                    variant="fp16", 
                    use_safetensors=True
                )
            # -------------------------------------------------

            self.pipeline.enable_model_cpu_offload()
            print("✅ Model loaded successfully.")

        # Logic chuyển đổi pipeline giữ nguyên
        if task_type == "img2img" and self.current_type != "img2img":
            print("🔄 Switching to Img2Img pipeline...")
            self.pipeline = AutoPipelineForImage2Image.from_pipe(self.pipeline)
            self.current_type = "img2img"
            
        elif task_type == "txt2img" and self.current_type != "txt2img":
            print("🔄 Switching to Txt2Img pipeline...")
            self.pipeline = AutoPipelineForText2Image.from_pipe(self.pipeline)
            self.current_type = "txt2img"
            
        return self.pipeline

    def load_lora(self, lora_path, adapter_name="default"):
        if self.pipeline:
            print(f"🔄 Loading LoRA adapter: {lora_path}")
            try:
                self.pipeline.load_lora_weights(lora_path, adapter_name=adapter_name)
                self.pipeline.fuse_lora()
                print("✅ LoRA Loaded & Fused.")
            except Exception as e:
                print(f"❌ Lỗi load LoRA: {e}")

    def unload_lora(self):
        if self.pipeline:
            print(f"🔄 Unloading LoRA...")
            self.pipeline.unfuse_lora()
            self.pipeline.unload_lora_weights()
            print("✅ LoRA Unloaded.")