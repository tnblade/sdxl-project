# core/utils.py
from PIL import Image
import cv2
import numpy as np

def resize_image_for_diffusion(image_path: str, max_dim: int = 1024):
    """
    Resize ảnh sao cho cạnh lớn nhất = max_dim, 
    và cả 2 cạnh đều chia hết cho 8 (Yêu cầu bắt buộc của Stable Diffusion)
    """
    img = Image.open(image_path).convert("RGB")
    w, h = img.size
    
    # Tính tỷ lệ resize
    scale = max_dim / max(w, h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    
    # Làm tròn để chia hết cho 8
    new_w = new_w - (new_w % 8)
    new_h = new_h - (new_h % 8)
    
    return img.resize((new_w, new_h), Image.LANCZOS)

def image_to_canny(image_path: str):
    """Chuyển ảnh thường thành ảnh nét đứt (Canny edge) cho ControlNet"""
    # Đọc ảnh bằng PIL rồi chuyển sang Numpy
    pil_image = Image.open(image_path).convert("RGB")
    image = np.array(pil_image)
    
    # Dùng OpenCV để lấy nét
    low_threshold = 100
    high_threshold = 200
    image = cv2.Canny(image, low_threshold, high_threshold)
    
    # Chuyển về định dạng PIL 3 kênh màu để đưa vào model
    image = image[:, :, None]
    image = np.concatenate([image, image, image], axis=2)
    return Image.fromarray(image)