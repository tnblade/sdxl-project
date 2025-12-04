#   core/loaders.py
# Đây là "thủ kho". Nhiệm vụ duy nhất là Load Model. Sau này sẽ thêm hàm load_lora, load_controlnet vào class này cực kỳ gọn gàng.

import torch
import os
from diffusers import AutoPipelineForText2Image, AutoPipelineForImage2Image
from .config import Config

class ModelLoader:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.pipeline = None
        self.current_type = None # "txt2img" hoặc "img2img"

    def load_base_pipeline(self, task_type="txt2img"):
        """Load model linh hoạt: Hỗ trợ cả Folder lẫn Single File (.safetensors)"""
        
        # 1. Load mới nếu chưa có
        if self.pipeline is None:
            model_id = Config.get_model_path()
            print(f"📥 Loading Base Model from: {model_id}...")
            
            # --- LOGIC MỚI: TỰ ĐỘNG NHẬN DIỆN LOẠI MODEL ---
            if str(model_id).endswith(".safetensors"):
                print("⚡ Phát hiện Single File (.safetensors) -> Dùng from_single_file")
                # Dành cho Kaggle Input hoặc file tải về lẻ
                self.pipeline = AutoPipelineForText2Image.from_single_file(
                    model_id, 
                    torch_dtype=torch.float16, 
                    use_safetensors=True
                )
            else:
                print("☁️ Phát hiện Folder/Repo -> Dùng from_pretrained")
                # Dành cho HuggingFace Repo hoặc thư mục đã giải nén
                self.pipeline = AutoPipelineForText2Image.from_pretrained(
                    model_id, 
                    torch_dtype=torch.float16, 
                    variant="fp16", 
                    use_safetensors=True
                )
            # -----------------------------------------------

            # Tối ưu bộ nhớ cho T4/P100
            self.pipeline.enable_model_cpu_offload()
            print("✅ Model loaded successfully.")

        # 2. Chuyển đổi pipeline (Txt2Img <-> Img2Img) mà không load lại RAM
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
            print(f"Bm Loading LoRA adapter: {lora_path}")
            try:
                self.pipeline.load_lora_weights(lora_path, adapter_name=adapter_name)
                self.pipeline.fuse_lora() # Gộp weights để chạy nhanh hơn
                print("✅ LoRA Loaded & Fused.")
            except Exception as e:
                print(f"❌ Lỗi load LoRA: {e}")

    def unload_lora(self):
        if self.pipeline:
            print(f"Bm Unloading LoRA...")
            self.pipeline.unfuse_lora()
            self.pipeline.unload_lora_weights()
            print("✅ LoRA Unloaded.")