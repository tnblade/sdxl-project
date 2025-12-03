import streamlit as st
from core.pipeline_handler import SDXLManager
from PIL import Image
import random

# Thiết lập trang
st.set_page_config(page_title="AI Image Generator", page_icon="✨", layout="wide")

st.markdown("""
<style>
    div.stButton > button:first-child {
        background-color: #ff0000;
        color: white;
        width: 100%;
        border-radius: 5px;
        height: 50px;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def get_manager():
    return SDXLManager()

manager = get_manager()

# --- SIDEBAR ---
with st.sidebar:
    st.header("Cấu hình")
    
    # CẬP NHẬT: Thêm slider chọn số lượng ảnh
    num_images = st.slider("Số lượng ảnh", min_value=1, max_value=5, value=1)
    
    width = st.select_slider("Chiều rộng", options=[512, 768, 1024], value=1024)
    height = st.select_slider("Chiều cao", options=[512, 768, 1024], value=1024)
    steps = st.slider("Số bước vẽ (Steps)", 10, 50, 30)
    
    seed = st.number_input("Hạt giống (-1 ngẫu nhiên)", value=-1)
    if seed == -1:
        seed = random.randint(0, 2147483647)

    st.markdown("---")
    st.info("Task 2: LoRA loader sẽ được thêm vào đây sau.")

# --- MAIN SCREEN ---
st.title("✨ AI Image Generator ✨")

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("1. Nhập liệu")
    prompt = st.text_area("Mô tả ảnh", value="a city", height=100)
    negative_prompt = st.text_input("Loại bỏ (Negative Prompt)", value="ugly, blurry, low quality")
    
    st.markdown("### Ảnh mẫu (Tùy chọn cho Img2Img)")
    uploaded_file = st.file_uploader("Kéo thả file vào đây", type=['png', 'jpg', 'jpeg'])
    
    generate_btn = st.button("TẠO ẢNH")

with col2:
    st.subheader("2. Kết quả")
    
    if generate_btn:
        with st.spinner(f'Đang vẽ {num_images} ảnh...'):
            try:
                input_img = uploaded_file if uploaded_file else None
                
                # Gọi hàm generate với tham số num_images mới
                result_images = manager.generate(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    steps=steps,
                    width=width,
                    height=height,
                    seed=seed,
                    num_images=num_images, # Truyền số lượng ảnh vào
                    input_image=input_img
                )
                
                # Hiển thị kết quả (Duyệt qua danh sách ảnh trả về)
                for idx, img in enumerate(result_images):
                    st.image(
                        img, 
                        caption=f"Ảnh {idx+1} - Seed: {seed + idx}", 
                        use_container_width=True
                    )
                
            except Exception as e:
                st.error(f"Có lỗi xảy ra: {str(e)}")