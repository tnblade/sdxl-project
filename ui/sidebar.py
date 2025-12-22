# ui/sidebar.py
# Sidebar cấu hình vẽ ảnh với LoRA và các thông số

import streamlit as st
import random
import os
import glob

# Đường dẫn chứa file LoRA (Chuẩn Linux cho Kaggle)
LORA_DIR = "sdxl-project/fine_tuning/lora"

def get_available_loras():
    # Tạo thư mục nếu chưa có
    if not os.path.exists(LORA_DIR):
        os.makedirs(LORA_DIR, exist_ok=True)
        
    # Quét tất cả file .safetensors
    files = glob.glob(os.path.join(LORA_DIR, "*.safetensors"))
    # Lấy tên file
    lora_list = ["None"] + [os.path.basename(f) for f in files]
    return lora_list

def show_sidebar(manager):
    with st.sidebar:
        st.header("🎨 Cấu hình Vẽ")
        
        # --- PHẦN 1: QUẢN LÝ LORA ---
        st.subheader("1. Chọn LoRA (Fine-tune)")
        st.caption(f"📁 Folder: {LORA_DIR}")
        
        # Quét file tự động
        available_loras = get_available_loras()
        selected_lora_name = st.selectbox("Danh sách LoRA:", available_loras)
        
        # Gợi ý Trigger Word dựa trên tên file
        default_trigger = ""
        if "char" in selected_lora_name.lower(): default_trigger = "char_style_v1, anime girl"
        elif "scene" in selected_lora_name.lower(): default_trigger = "scene_style_v1, anime scenery"
        
        # Ô nhập Trigger Word
        trigger_word = st.text_input("Trigger Word:", value=default_trigger)
        st.session_state['current_trigger'] = trigger_word
        
        # Nút Nạp/Gỡ
        col1, col2 = st.columns(2)
        if col1.button("📥 Nạp LoRA", type="primary"):
            if selected_lora_name != "None":
                full_path = os.path.join(LORA_DIR, selected_lora_name)
                if os.path.exists(full_path):
                    with st.spinner(f"Đang nạp {selected_lora_name}..."):
                        try:
                            manager.loader.load_lora(full_path)
                            st.success("✅ Đã nạp!")
                        except Exception as e:
                            st.error(f"Lỗi: {e}")
                else:
                    st.error("❌ File không tồn tại!")
            else:
                st.warning("Vui lòng chọn file!")
                
        if col2.button("❌ Gỡ bỏ"):
            manager.loader.unload_lora()
            st.info("Đã gỡ LoRA.")

        st.markdown("---")
        
        # --- PHẦN 2: THÔNG SỐ ---
        st.subheader("2. Thông số ảnh")
        config = {}
        config['num_images'] = st.slider("Số lượng ảnh", 1, 4, 1)
        config['width'] = st.select_slider("Chiều rộng", options=[768, 1024, 1280], value=1024)
        config['height'] = st.select_slider("Chiều cao", options=[768, 1024, 1280], value=1024)
        config['steps'] = st.slider("Số bước (Steps)", 20, 50, 30)
        
        seed_input = st.number_input("Seed (-1 ngẫu nhiên)", value=-1)
        config['seed'] = random.randint(0, 2147483647) if seed_input == -1 else seed_input
            
        config['enable_scoring'] = st.checkbox("Bật chấm điểm AI", value=False)
        
        return config