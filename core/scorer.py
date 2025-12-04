# core/scorer.py
# Bộ phận đánh giá chất lượng ảnh tạo ra dựa trên CLIP Score và Aesthetic Score

import torch
import clip
from PIL import Image
import torch.nn as nn
import os
import urllib.request

class AestheticPredictor(nn.Module):
    """Mạng Neural nhỏ để dự đoán độ thẩm mỹ (Aesthetic)"""
    def __init__(self, input_size):
        super().__init__()
        self.input_size = input_size
        self.layers = nn.Sequential(
            nn.Linear(self.input_size, 1024),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Linear(1024, 128),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.Dropout(0.1),
            nn.ReLU(),
            nn.Linear(64, 16),
            nn.Linear(16, 1)
        )

    def forward(self, x):
        return self.layers(x)

class ImageScorer:
    def __init__(self, device="cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        self.clip_model = None
        self.preprocess = None
        self.aesthetic_model = None
        
        # Đường dẫn lưu model aesthetic
        self.aesthetic_path = "aesthetic_predictor.pth"
        self.aesthetic_url = "https://github.com/christophschuhmann/improved-aesthetic-predictor/raw/main/sac+logos+ava1-l14-linearMSE.pth"

    def load_models(self):
        """Chỉ load khi cần dùng để tiết kiệm VRAM"""
        if self.clip_model is None:
            print("📊 Loading CLIP for scoring...")
            # Load CLIP ViT-L/14 (Chuẩn phổ biến)
            self.clip_model, self.preprocess = clip.load("ViT-L/14", device=self.device)
            
            # Load Aesthetic Model
            if not os.path.exists(self.aesthetic_path):
                print("Downloading Aesthetic Model...")
                urllib.request.urlretrieve(self.aesthetic_url, self.aesthetic_path)
            
            self.aesthetic_model = AestheticPredictor(768)
            self.aesthetic_model.load_state_dict(torch.load(self.aesthetic_path))
            self.aesthetic_model.to(self.device)
            self.aesthetic_model.eval()
            print("✅ Scoring models loaded.")

    def get_scores(self, image, prompt):
        """Trả về Aesthetic Score (1-10) và CLIP Score (0-100)"""
        self.load_models()
        
        # Xử lý ảnh và text
        image_input = self.preprocess(image).unsqueeze(0).to(self.device)
        text_input = clip.tokenize([prompt], truncate=True).to(self.device)

        with torch.no_grad():
            # 1. Tính CLIP Embeddings
            image_features = self.clip_model.encode_image(image_input)
            text_features = self.clip_model.encode_text(text_input)
            
            # Chuẩn hóa vector
            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)

            # 2. Tính CLIP Score (Độ tương đồng cosine * 100)
            clip_score = (image_features @ text_features.T).item() * 100
            
            # 3. Tính Aesthetic Score
            # Aesthetic model input là CLIP image features (float32)
            aesthetic_score = self.aesthetic_model(image_features.float()).item()

        return round(clip_score, 2), round(aesthetic_score, 2)
        
    def unload(self):
        """Giải phóng RAM khi không dùng"""
        self.clip_model = None
        self.aesthetic_model = None
        torch.cuda.empty_cache()