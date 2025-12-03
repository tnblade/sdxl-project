import torch
import os
from diffusers import AutoPipelineForText2Image, AutoPipelineForImage2Image
from diffusers.utils import load_image
import gc

class SDXLManager:
    def __init__(self, model_path="./sdxl_models/base"):
        # Kiểm tra xem đã tải model về local chưa, nếu chưa thì fallback về Huggingface ID
        if os.path.exists(model_path) and os.listdir(model_path):
            self.model_id = model_path
            print(f"Loading from LOCAL path: {self.model_id}")
        else:
            self.model_id = "stabilityai/stable-diffusion-xl-base-1.0"
            print(f"Local path not found. Loading from HuggingFace: {self.model_id}")
            
        self.pipeline = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def load_pipeline(self, task_type="txt2img"):
        if self.pipeline is None:
            print("Initializing Pipeline...")
            self.pipeline = AutoPipelineForText2Image.from_pretrained(
                self.model_id, 
                torch_dtype=torch.float16, 
                variant="fp16", 
                use_safetensors=True
            )
            self.pipeline.enable_model_cpu_offload() 
            print("Model loaded successfully.")
        
        # Chuyển đổi pipeline logic
        if task_type == "img2img" and not isinstance(self.pipeline, AutoPipelineForImage2Image):
            self.pipeline = AutoPipelineForImage2Image.from_pipe(self.pipeline)
        elif task_type == "txt2img" and not isinstance(self.pipeline, AutoPipelineForText2Image):
            self.pipeline = AutoPipelineForText2Image.from_pipe(self.pipeline)
            
        return self.pipeline

    def generate(self, prompt, negative_prompt, steps, width, height, seed, input_image=None):
        # Set seed
        generator = torch.Generator(device="cpu").manual_seed(seed)
        
        if input_image:
            # Img2Img Mode
            pipe = self.load_pipeline("img2img")
            init_img = load_image(input_image).convert("RGB")
            # Resize input image to fit logic width/height roughly if needed
            init_img = init_img.resize((width, height))
            
            image = pipe(
                prompt=prompt, 
                negative_prompt=negative_prompt, 
                image=init_img,
                num_inference_steps=steps,
                strength=0.8, # Default strength
                generator=generator,
                guidance_scale=7.5
            ).images[0]
        else:
            # Txt2Img Mode
            pipe = self.load_pipeline("txt2img")
            image = pipe(
                prompt=prompt, 
                negative_prompt=negative_prompt, 
                width=width,
                height=height,
                num_inference_steps=steps,
                generator=generator,
                guidance_scale=7.5
            ).images[0]
            
        return image