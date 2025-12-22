import streamlit as st
from PIL import Image

def show_home(manager, scorer, config):
    st.title("✨ SDXL Anime Lab ✨")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("📝 Nhập liệu")
        
        # Tự động điền Trigger Word vào đầu prompt nếu có
        current_trigger = st.session_state.get('current_trigger', '')
        base_prompt = "masterpiece, best quality, highres, 8k, vivid colors"
        
        if current_trigger:
            default_val = f"{current_trigger}, {base_prompt}"
            st.info(f"💡 Đang dùng Trigger: {current_trigger}")
        else:
            default_val = base_prompt

        prompt = st.text_area("Mô tả ảnh (Prompt):", value=default_val, height=150)
        negative_prompt = st.text_input("Loại bỏ (Negative):", value="ugly, blurry, low quality, lowres, deformed, bad anatomy, nsfw, text, watermark, bad hands")
        
        uploaded_file = st.file_uploader("Ảnh mẫu (Img2Img - Tùy chọn)", type=['png', 'jpg', 'jpeg'])
        real_input_image = None
        
        if uploaded_file:
            # Sửa lỗi hiển thị: Mở ảnh bằng PIL trước
            image_preview = Image.open(uploaded_file)
            st.image(image_preview, caption="Ảnh đầu vào", use_container_width=True)
            
            # Reset con trỏ và convert để đưa vào model
            uploaded_file.seek(0)
            real_input_image = Image.open(uploaded_file).convert("RGB")
        
        generate_btn = st.button("🎨 TẠO ẢNH NGAY", type="primary", use_container_width=True)

    with col2:
        st.subheader("🖼️ Kết quả")
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
                                m1.metric("Đúng đề (CLIP)", c_score)
                                m2.metric("Thẩm mỹ (Aes)", f"{a_score}/10")
                            
                except Exception as e:
                    st.error(f"Lỗi: {e}")
                    st.info("💡 Mẹo: Nếu gặp lỗi bộ nhớ (OOM), hãy giảm kích thước ảnh hoặc restart session.")