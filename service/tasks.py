# service/tasks.py
import sys
import os

# Thêm thư mục gốc vào đường dẫn
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.text_to_image import generate_text_image
from config import DEFAULT_WIDTH, DEFAULT_HEIGHT

def run_generation_task(prompt: str, negative_prompt: str = "low quality, blurry", width=DEFAULT_WIDTH, height=DEFAULT_HEIGHT, steps=20, seed=-1):
    """
    Hàm này nhận đầy đủ tham số từ UI gửi xuống.
    """
    print(f"🔨 [TASK STARTED] Prompt: {prompt[:50]}... | Size: {width}x{height} | Steps: {steps}")
    
    try:
        # Chuyển đổi seed: UI gửi -1 nghĩa là ngẫu nhiên (None)
        gen_seed = int(seed) if int(seed) != -1 else None
        
        output_path = generate_text_image(
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=int(width),
            height=int(height),
            num_inference_steps=int(steps), # Truyền số bước vào Core
            seed=gen_seed
        )
        
        print(f"✅ [TASK DONE] Saved at: {output_path}")
        return output_path
        
    except Exception as e:
        print(f"❌ [TASK FAILED] Error: {e}")
        # Quan trọng: Phải raise lỗi để Redis biết là Job này bị Fail
        raise e