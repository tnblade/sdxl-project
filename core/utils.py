# core/utils.py
# Chứa các hàm phụ trợ (Helper functions) như xử lý ảnh và dọn dẹp bộ nhớ. Tách biệt hoàn toàn với logic AI.

import torch
import gc
from PIL import Image
from diffusers.utils import load_image

def free_memory():
    """Hàm dọn dẹp VRAM/RAM cực mạnh"""
    gc.collect()
    torch.cuda.empty_cache()
    # print("🧹 Memory cleared.")

def process_input_image(input_image, width, height):
    """Chuẩn hóa đầu vào ảnh: String path hoặc PIL Image đều về PIL RGB"""
    if input_image is None:
        return None
        
    final_image = None
    
    if isinstance(input_image, str):
        # Nếu là đường dẫn file
        final_image = load_image(input_image)
    elif isinstance(input_image, Image.Image):
        # Nếu đã là PIL Image
        final_image = input_image
    else:
        raise ValueError("Input image phải là đường dẫn (str) hoặc PIL Image")
        
    # Convert sang RGB và Resize
    return final_image.convert("RGB").resize((width, height))