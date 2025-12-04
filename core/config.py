# core/config.py
# Nơi tập trung mọi cấu hình cứng. Sau này muốn đổi đường dẫn model, chỉ cần sửa ở đây mà không sợ tìm không ra.

import os

class Config:
    # --- CÁC ĐƯỜNG DẪN CÓ THỂ CÓ ---
    
    # 1. Đường dẫn Dataset chuẩn của Kaggle (Input Read-only)
    # (Có thể có nhiều version path, ta liệt kê các path phổ biến)
    KAGGLE_PATH_1 = "/kaggle/input/stable-diffusion-xl/pytorch/base-1-0/1/sd_xl_base_1.0.safetensors"
    KAGGLE_PATH_2 = "/kaggle/input/stable-diffusion-xl-base-1-0/sd_xl_base_1.0.safetensors"
    
    # 2. Đường dẫn Local (Khi chạy download.py trên Colab/PC)
    LOCAL_PATH = "./sdxl_models/base"
    
    # 3. ID trên HuggingFace (Fallback cuối cùng)
    HF_REPO_ID = "stabilityai/stable-diffusion-xl-base-1.0"
    
    # --- CẤU HÌNH MẶC ĐỊNH ---
    DEFAULT_STEPS = 30
    DEFAULT_GUIDANCE = 7.5
    DEFAULT_SIZE = 1024

    @classmethod
    def get_model_path(cls):
        """Logic ưu tiên: Kaggle -> Local -> HuggingFace"""
        
        # Check Kaggle Path 1
        if os.path.exists(cls.KAGGLE_PATH_1):
            print(f"✅ Config: Đang chạy trên KAGGLE (Path 1).")
            return cls.KAGGLE_PATH_1
            
        # Check Kaggle Path 2
        if os.path.exists(cls.KAGGLE_PATH_2):
            print(f"✅ Config: Đang chạy trên KAGGLE (Path 2).")
            return cls.KAGGLE_PATH_2
            
        # Check Local Path (Folder phải tồn tại và có file)
        if os.path.exists(cls.LOCAL_PATH) and len(os.listdir(cls.LOCAL_PATH)) > 0:
            print(f"✅ Config: Đang chạy trên LOCAL/COLAB.")
            return cls.LOCAL_PATH
            
        # Fallback
        print(f"⚠️ Config: Không tìm thấy model offline. Dùng HuggingFace ID: {cls.HF_REPO_ID}")
        return cls.HF_REPO_ID