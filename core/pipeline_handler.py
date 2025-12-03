import torch
import os
from diffusers import AutoPipelineForText2Image, AutoPipelineForImage2Image
from diffusers.utils import load_image
import gc

class SDXLManager:
    def __init__(self, model_path="./sdxl_models/base"):
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
        
        if task_type == "img2img" and not isinstance(self.pipeline, AutoPipelineForImage2Image):
            self.pipeline = AutoPipelineForImage2Image.from_pipe(self.pipeline)
        elif task_type == "txt2img" and not isinstance(self.pipeline, AutoPipelineForText2Image):
            self.pipeline = AutoPipelineForText2Image.from_pipe(self.pipeline)
            
        return self.pipeline

    # CẬP NHẬT: Thêm tham số num_images và trả về danh sách ảnh (List)
    def generate(self, prompt, negative_prompt, steps, width, height, seed, num_images=1, input_image=None):
        generated_images = []
        
        for i in range(num_images):
            # Mỗi ảnh sẽ có seed tăng dần để tạo sự khác biệt (seed, seed+1, seed+2...)
            current_seed = seed + i
            generator = torch.Generator(device="cpu").manual_seed(current_seed)
            
            print(f"Generating image {i+1}/{num_images} with seed {current_seed}...")
            
            if input_image:
                # Img2Img Mode
                pipe = self.load_pipeline("img2img")
                init_img = load_image(input_image).convert("RGB")
                init_img = init_img.resize((width, height))
                
                image = pipe(
                    prompt=prompt, 
                    negative_prompt=negative_prompt, 
                    image=init_img,
                    num_inference_steps=steps,
                    strength=0.8,
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
            
            generated_images.append(image)
            
        return generated_images # Trả về một danh sách (List) các ảnh