# ui/sidebar.py
# File này chịu trách nhiệm hiển thị các thanh trượt và trả về một bộ dữ liệu cấu hình (dictionary).

import streamlit as st
import random
import os

def show_sidebar(manager):
    """Hiển thị Sidebar và trả về các tham số cấu hình"""
    with st.sidebar:
        st.header("1. Cấu hình Cơ bản")
        
        # Gom các tham số vào dictionary
        config = {}
        
        config['num_images'] = st.slider("Số lượng ảnh", 1, 5, 1)
        config['width'] = st.select_slider("Chiều rộng", options=[512, 768, 1024], value=1024)
        config['height'] = st.select_slider("Chiều cao", options=[512, 768, 1024], value=1024)
        config['steps'] = st.slider("Số bước (Steps)", 10, 50, 30)
        
        seed_input = st.number_input("Seed (-1 ngẫu nhiên)", value=-1)
        if seed_input == -1:
            config['seed'] = random.randint(0, 2147483647)
        else:
            config['seed'] = seed_input
            
        st.markdown("---")
        st.header("2. Fine-tuning (LoRA)")
        
        lora_path = st.text_input("Đường dẫn file LoRA", 
                                  placeholder="output_lora/pytorch_lora_weights.safetensors")
        
        col1, col2 = st.columns(2)
        if col1.button("Nạp LoRA"):
            if lora_path and os.path.exists(lora_path):
                with st.spinner("Đang nạp..."):
                    manager.loader.load_lora(lora_path)
                    st.success("Đã nạp!")
            else:
                st.error("Lỗi đường dẫn")
                
        if col2.button("Gỡ LoRA"):
            manager.loader.unload_lora()
            st.info("Đã gỡ.")

        st.markdown("---")
        st.header("3. Công cụ")
        config['enable_scoring'] = st.checkbox("Bật chấm điểm", value=False)
        
        return config