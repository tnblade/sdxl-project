# ui/home.py
# File này chịu trách nhiệm hiển thị màn hình chính, upload ảnh và gọi hàm vẽ.

import streamlit as st
from PIL import Image

def show_home(manager, scorer, config):
    """Hiển thị màn hình chính"""
    st.title("✨ AI Image Generator ✨")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Nhập liệu")
        prompt = st.text_area("Mô tả ảnh (Prompt)", value="a photo of sks style", height=100)
        negative_prompt = st.text_input("Loại bỏ (Negative)", value="ugly, blurry")
        
        uploaded_file = st.file_uploader("Ảnh mẫu (Img2Img)", type=['png', 'jpg', 'jpeg'])
        real_input_image = None
        
        if uploaded_file:
            st.image(uploaded_file, caption="Input", use_container_width=True)
            uploaded_file.seek(0)
            real_input_image = Image.open(uploaded_file).convert("RGB")
        
        generate_btn = st.button("TẠO ẢNH")

    with col2:
        st.subheader("Kết quả")
        if generate_btn:
            with st.spinner(f"Đang vẽ {config['num_images']} ảnh..."):
                try:
                    # Gọi Core để vẽ
                    images = manager.generate(
                        prompt=prompt,
                        negative_prompt=negative_prompt,
                        steps=config['steps'],
                        width=config['width'],
                        height=config['height'],
                        seed=config['seed'],
                        num_images=config['num_images'],
                        input_image=real_input_image
                    )
                    
                    # Hiển thị
                    for idx, img in enumerate(images):
                        st.image(img, caption=f"Seed: {config['seed']+idx}", use_container_width=True)
                        
                        # Chấm điểm (Nếu bật)
                        if config['enable_scoring']:
                            c_score, a_score = scorer.get_scores(img, prompt)
                            m1, m2 = st.columns(2)
                            m1.metric("CLIP", c_score)
                            m2.metric("Aesthetic", f"{a_score}/10")
                            
                except Exception as e:
                    st.error(f"Lỗi: {e}")