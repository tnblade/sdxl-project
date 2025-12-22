# ui/home.py
# Trang chính giao diện vẽ ảnh với Stable Diffusion

import streamlit as st
from PIL import Image

def show_home(manager, scorer, config):
    st.title("✨ AI Image Generator ✨")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("1. Nhập liệu")
        
        # Tự động điền Trigger Word vào đầu prompt
        current_trigger = st.session_state.get('current_trigger', '')
        base_prompt = "masterpiece, best quality, highres, 8k, vivid colors"
        
        if current_trigger:
            default_val = f"{current_trigger}, {base_prompt}"
            st.caption(f"💡 Đang dùng Style: {current_trigger}")
        else:
            default_val = base_prompt

        prompt = st.text_area("Mô tả ảnh (Prompt):", value=default_val, height=150)
        negative_prompt = st.text_input("Loại bỏ (Negative):", value="ugly, blurry, low quality, lowres, deformed, bad anatomy, nsfw, text, watermark, bad hands")
        
        st.markdown("### Ảnh mẫu (Tùy chọn cho Img2Img)")
        uploaded_file = st.file_uploader("Kéo thả file vào đây", type=['png', 'jpg', 'jpeg'])
        real_input_image = None
        
        if uploaded_file:
            # Sửa lỗi hiển thị: Mở ảnh bằng PIL trước
            image_preview = Image.open(uploaded_file)
            st.image(image_preview, caption="Input", use_container_width=True)
            
            # Reset con trỏ và convert để đưa vào model
            uploaded_file.seek(0)
            real_input_image = Image.open(uploaded_file).convert("RGB")
        
        # Nút tạo ảnh màu đỏ giống hình mẫu
        st.markdown("<br>", unsafe_allow_html=True)
        generate_btn = st.button("TẠO ẢNH", type="primary", use_container_width=True)

    with col2:
        st.subheader("2. Kết quả")
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
                        st.image(img, caption=f"Seed: {config['seed']+idx}", use_container_width=True)
                        
                        if config['enable_scoring']:
                            with st.spinner("Đang chấm điểm..."):
                                c_score, a_score = scorer.get_scores(img, prompt)
                                m1, m2 = st.columns(2)
                                m1.metric("CLIP", c_score)
                                m2.metric("Aesthetic", f"{a_score}/10")
                            
                except Exception as e:
                    st.error(f"Lỗi: {e}")
                    st.info("💡 Mẹo: Nếu gặp lỗi bộ nhớ (OOM), hãy giảm kích thước ảnh hoặc restart session.")