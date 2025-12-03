# core/generator.py
# Đây là bộ phận chính xử lý việc tạo ảnh từ prompt, sử dụng pipeline đã được load từ loaders.py
# File này sẽ kết hợp ModelLoader và utils để tạo ra sản phẩm cuối cùng. Nó thay thế cho SDXLManager cũ nhưng gọn hơn.

import torch
from .loaders import ModelLoader
from .utils import process_input_image, free_memory
from .config import Config

class SDXLManager:
    def __init__(self):
        # Khởi tạo bộ phận load model
        self.loader = ModelLoader()

    def generate(self, prompt, negative_prompt, steps, width, height, seed, num_images=1, input_image=None):
        generated_images = []
        
        # Xác định chế độ
        task_type = "img2img" if input_image else "txt2img"
        
        # Lấy pipeline từ loader
        pipe = self.loader.load_base_pipeline(task_type)
        
        # Xử lý ảnh đầu vào (nếu có)
        init_img = process_input_image(input_image, width, height)
        
        try:
            for i in range(num_images):
                current_seed = seed + i
                # Generator cho pytorch (Seed)
                generator = torch.Generator(device="cpu").manual_seed(current_seed)
                
                print(f"🎨 Generating {i+1}/{num_images} | Seed: {current_seed} | Mode: {task_type}")
                
                # Tham số chung cho cả 2 mode
                common_args = {
                    "prompt": prompt,
                    "negative_prompt": negative_prompt,
                    "num_inference_steps": steps,
                    "guidance_scale": Config.DEFAULT_GUIDANCE,
                    "generator": generator
                }
                
                if task_type == "img2img":
                    image = pipe(
                        image=init_img,
                        strength=0.8, # Có thể đưa ra config nếu muốn
                        **common_args
                    ).images[0]
                else:
                    image = pipe(
                        width=width,
                        height=height,
                        **common_args
                    ).images[0]
                
                generated_images.append(image)
                
        except Exception as e:
            print(f"❌ Error during generation: {e}")
            raise e
        finally:
            # Dọn dẹp nhẹ sau khi vẽ xong
            free_memory()
            
        return generated_images