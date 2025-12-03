# core/config.py
# Nơi tập trung mọi cấu hình cứng. Sau này muốn đổi đường dẫn model, chỉ cần sửa ở đây mà không sợ tìm không ra.

import os

class Config:
    # Đường dẫn Model
    BASE_MODEL_PATH = "./sdxl_models/base"
    HF_REPO_ID = "stabilityai/stable-diffusion-xl-base-1.0"
    
    # Cấu hình mặc định
    DEFAULT_STEPS = 30
    DEFAULT_GUIDANCE = 7.5
    DEFAULT_SIZE = 1024
    
    # Kiểm tra xem đang chạy local hay cần tải từ HF
    @classmethod
    def get_model_path(cls):
        if os.path.exists(cls.BASE_MODEL_PATH) and os.listdir(cls.BASE_MODEL_PATH):
            return cls.BASE_MODEL_PATH
        return cls.HF_REPO_ID