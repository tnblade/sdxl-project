# app.py
# Ứng dụng Streamlit cho SDXL Anime Lab


import streamlit as st
import sys
import os

# --- CẤU HÌNH HỆ THỐNG ---
# Thêm đường dẫn hiện tại vào sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

# Import module theo cấu trúc của bạn
from core.generator import SDXLManager  # <--- SỬA Ở ĐÂY: Dùng generator.py
from core.scorer import ImageScorer
from ui.sidebar import show_sidebar
from ui.home import show_home

# 1. Setup Trang
st.set_page_config(page_title="SDXL Anime Lab", page_icon="🎨", layout="wide")

# CSS cho nút bấm
st.markdown("<style>div.stButton > button:first-child {background-color: #FF4B4B; color: white; font-weight: bold;}</style>", unsafe_allow_html=True)

# 2. Khởi tạo Core System (Cache Resource)
@st.cache_resource
def load_core():
    print("🐢 Đang khởi tạo hệ thống (Generator + Scorer)...")
    
    # Class SDXLManager của bạn trong generator.py tự init Loader bên trong
    # nên không cần truyền tham số gì cả (theo code bạn gửi)
    manager = SDXLManager() 
    
    scorer = ImageScorer()
    return manager, scorer

try:
    manager, scorer = load_core()
except Exception as e:
    st.error(f"❌ Lỗi khởi động hệ thống: {e}")
    st.stop()

# 3. Main Loop
def main():
    # Load UI Sidebar -> Nhận config
    config = show_sidebar(manager)
    
    # Load UI Home -> Vẽ ảnh
    show_home(manager, scorer, config)

if __name__ == "__main__":
    main()