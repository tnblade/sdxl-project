# ui/sidebar.py
# Sidebar cấu hình vẽ ảnh với LoRA và các thông số

import streamlit as st
import random
import os
import glob

# Đường dẫn chứa file LoRA
LORA_DIR = "/kaggle/working/sdxl-project/fine_tuning/lora"

def get_available_loras():
    if not os.path.exists(LORA_DIR):
        try:
            os.makedirs(LORA_DIR, exist_ok=True)
        except: pass
        
    search_path = os.path.join(LORA_DIR, "*.safetensors")
    files = glob.glob(search_path)
    lora_list = ["None"] + [os.path.basename(f) for f in files]
    return lora_list

def show_sidebar(manager):
    with st.sidebar:
        st.header("Cấu hình")
        
        # --- QUẢN LÝ LORA ---
        st.subheader("Fine-tuning (LoRA)")
        st.caption(f"📁 Folder: .../fine_tuning/lora")
        
        available_loras = get_available_loras()
        if len(available_loras) == 1:
            st.warning("⚠️ Chưa có file LoRA nào.")

        selected_lora_name = st.selectbox("Chọn Model LoRA:", available_loras)
        
        # Trigger Word Logic
        default_trigger = ""
        name_lower = selected_lora_name.lower()
        if "char" in name_lower: default_trigger = "char_style_v1, anime girl"
        elif "scene" in name_lower: default_trigger = "scene_style_v1, anime scenery"
        
        trigger_word = st.text_input("Trigger Word:", value=default_trigger)
        st.session_state['current_trigger'] = trigger_word
        
        col1, col2 = st.columns(2)
        
        # --- NÚT NẠP LORA ---
        if col1.button("📥 Nạp LoRA", type="primary"):
            if selected_lora_name != "None":
                full_path = os.path.join(LORA_DIR, selected_lora_name)
                if os.path.exists(full_path):
                    with st.spinner(f"Đang nạp {selected_lora_name}..."):
                        try:
                            # Gọi thẳng vào loader bên trong manager
                            # manager -> instance của SDXLManager
                            # manager.loader -> instance của ModelLoader
                            # manager.loader.load_lora -> hàm load_lora trong loaders.py
                            manager.loader.load_lora(full_path)
                            st.success("✅ Đã nạp!")
                        except Exception as e:
                            st.error(f"Lỗi: {e}")
                else:
                    st.error("❌ File không tồn tại!")
            else:
                st.warning("Chọn file trước!")
                
        # --- NÚT GỠ LORA ---
        if col2.button("❌ Gỡ bỏ"):
            try:
                manager.loader.unload_lora()
                st.info("Đã gỡ LoRA.")
            except Exception as e:
                st.error(f"Lỗi gỡ: {e}")

        st.markdown("---")
        
        # --- THÔNG SỐ ---
        st.subheader("Thông số ảnh")
        config = {}
        config['num_images'] = st.slider("Số lượng ảnh", 1, 4, 1)
        config['width'] = st.select_slider("Chiều rộng", options=[768, 1024], value=1024)
        config['height'] = st.select_slider("Chiều cao", options=[768, 1024], value=1024)
        config['steps'] = st.slider("Số bước (Steps)", 20, 50, 30)
        
        seed_input = st.number_input("Hạt giống (-1 ngẫu nhiên)", value=-1)
        config['seed'] = random.randint(0, 2147483647) if seed_input == -1 else seed_input
            
        config['enable_scoring'] = st.checkbox("Bật chấm điểm AI", value=False)
        
        return config