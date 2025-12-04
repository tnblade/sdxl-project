# app.py
# File chính khởi động toàn bộ hệ thống UI và Core
# Sử dụng Streamlit để xây dựng giao diện web đơn giản


import streamlit as st
from core import SDXLManager, ImageScorer
from ui import show_sidebar, show_home

# 1. Setup Trang
st.set_page_config(page_title="AI Project", page_icon="🎨", layout="wide")
st.markdown("<style>div.stButton > button:first-child {background-color: #ff0000; color: white;}</style>", unsafe_allow_html=True)

# 2. Init Core System (Chạy 1 lần)
@st.cache_resource
def load_core():
    return SDXLManager(), ImageScorer()

manager, scorer = load_core()

# 3. Load UI Sidebar -> Nhận về config
config = show_sidebar(manager)

# 4. Load UI Home -> Truyền config vào để vẽ
show_home(manager, scorer, config)