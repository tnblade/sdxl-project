# ui/home.py
  

import streamlit as st
from PIL import Image

def show_home(manager, scorer, config):
    """Hiển thị màn hình chính"""
    st.title("✨ AI Image Generator ✨")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Nhập liệu")
        # Sửa Prompt mặc định chi tiết hơn để tránh ra ảnh đen
        prompt = st.text_area("Mô tả ảnh (Prompt)", value="beautiful blue sky with white fluffy clouds, high quality, 4k", height=100)
        negative_prompt = st.text_input("Loại bỏ (Negative)", value="ugly, blurry, low quality, lowres, dark, nsfw")
        
        uploaded_file = st.file_uploader("Ảnh mẫu (Img2Img)", type=['png', 'jpg', 'jpeg'])
        real_input_image = None
        
        if uploaded_file:
            # SỬA LỖI 1: Mở ảnh thành PIL trước khi hiển thị để tránh lỗi format
            image_preview = Image.open(uploaded_file)
            st.image(image_preview, caption="Input Image", use_container_width=True)
            
            # Reset con trỏ file để đọc lại cho model
            uploaded_file.seek(0)
            real_input_image = Image.open(uploaded_file).convert("RGB")
        
        generate_btn = st.button("TẠO ẢNH")

    with col2:
        st.subheader("Kết quả")
        if generate_btn:
            with st.spinner(f"Đang vẽ {config['num_images']} ảnh..."):
                try:
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
                    
                    for idx, img in enumerate(images):
                        # Kiểm tra xem ảnh có bị đen (NSFW) không
                        # (Mẹo: Ảnh đen thường có độ lệch chuẩn màu rất thấp)
                        
                        # SỬA LỖI 2: Dùng use_container_width=True cho lành
                        st.image(img, caption=f"Seed: {config['seed']+idx}", use_container_width=True)
                        
                        if config['enable_scoring']:
                            with st.spinner("Đang chấm..."):
                                c_score, a_score = scorer.get_scores(img, prompt)
                                m1, m2 = st.columns(2)
                                m1.metric("CLIP", c_score)
                                m2.metric("Aesthetic", f"{a_score}/10")
                            
                except Exception as e:
                    st.error(f"Lỗi: {e}")