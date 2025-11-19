# ui/streamlit_app.py
import streamlit as st
import requests
import time
import os
from PIL import Image

# --- CẤU HÌNH ---
API_URL = os.getenv("API_URL", "http://localhost:8000")
DATA_DIR = "data"
TMP_DIR = os.path.join(DATA_DIR, "tmp")
OUTPUT_DIR = os.path.join(DATA_DIR, "images")

# Đảm bảo thư mục tồn tại
os.makedirs(TMP_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- SETUP TRANG ---
st.set_page_config(
    page_title="SDXL AI Studio",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🎨 SDXL Local AI Studio")
st.markdown(f"**Server Status:** `{API_URL}`")

# --- SIDEBAR (CẤU HÌNH GLOBAL) ---
with st.sidebar:
    st.header("⚙️ Cấu hình chung")
    
    # Các tham số này sẽ được áp dụng cho tất cả các chế độ
    width = st.select_slider("Chiều rộng (Width)", options=[512, 768, 1024], value=512)
    height = st.select_slider("Chiều cao (Height)", options=[512, 768, 1024], value=512)
    
    steps = st.slider("Số bước (Steps)", 10, 50, 20, help="Càng cao càng chi tiết nhưng lâu hơn.")
    guidance = st.slider("Guidance Scale", 1.0, 20.0, 7.5, help="Độ bám sát prompt.")
    seed = st.number_input("Seed (-1 = Ngẫu nhiên)", value=-1, step=1)
    
    st.divider()
    st.info("ℹ️ Lưu ý: Ảnh Upload sẽ được lưu tạm vào folder data/tmp để Worker xử lý.")

# --- HÀM HỖ TRỢ ---
def save_uploaded_file(uploaded_file):
    """
    Lưu file upload từ UI vào ổ cứng (Shared Volume) 
    để Worker có thể đọc được theo đường dẫn.
    """
    if uploaded_file is not None:
        file_path = os.path.join(TMP_DIR, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        # Trả về đường dẫn tuyệt đối hoặc tương đối mà Worker có thể hiểu
        return file_path 
    return None

def poll_job(job_id):
    """Hàm chờ kết quả từ Server (Polling)"""
    status_box = st.empty()
    progress_bar = st.progress(0)
    start_time = time.time()
    
    while True:
        try:
            resp = requests.get(f"{API_URL}/status/{job_id}")
            if resp.status_code != 200:
                status_box.error("Lỗi kết nối Server.")
                break
            
            data = resp.json()
            status = data.get("status")
            elapsed = int(time.time() - start_time)
            
            if status == "finished":
                progress_bar.progress(100)
                status_box.success(f"✅ Xong sau {elapsed}s!")
                return data.get("result")
            
            elif status == "failed":
                status_box.error(f"❌ Lỗi: {data.get('error')}")
                return None
            
            elif status == "queued":
                status_box.info(f"⏳ Đang chờ... ({elapsed}s)")
                progress_bar.progress(10)
                
            elif status == "started":
                status_box.warning(f"🎨 Đang vẽ... ({elapsed}s)")
                progress_bar.progress(50)
            
            time.sleep(2)
            
        except Exception as e:
            status_box.error(f"Lỗi: {str(e)}")
            break
    return None

# --- GIAO DIỆN CHÍNH ---
tab_txt, tab_img, tab_ctl, tab_gallery = st.tabs([
    "📝 Text to Image", 
    "🖼️ Image to Image", 
    "🤖 ControlNet",
    "📂 Thư viện ảnh"
])

# === TAB 1: TEXT TO IMAGE ===
with tab_txt:
    col1, col2 = st.columns([1, 1])
    with col1:
        st.subheader("Nhập Prompt")
        t2i_prompt = st.text_area("Mô tả ảnh", "A cute robot holding a flower, isometric style, 3d render", height=150)
        t2i_negative = st.text_input("Negative Prompt", "low quality, blurry, bad anatomy")
        
        if st.button("🚀 Tạo ảnh (Text2Img)", type="primary"):
            payload = {
                "prompt": t2i_prompt,
                "negative_prompt": t2i_negative,
                "task_type": "txt2img", # Mặc định
                "width": width, "height": height, "steps": steps, "seed": seed
            }
            try:
                r = requests.post(f"{API_URL}/generate", json=payload)
                if r.status_code == 200:
                    job_id = r.json().get("job_id")
                    st.success(f"Job ID: {job_id}")
                    with col2:
                        res = poll_job(job_id)
                        if res: st.image(res, caption=t2i_prompt)
                else:
                    st.error(r.text)
            except Exception as e:
                st.error(f"Lỗi kết nối API: {e}")

# === TAB 2: IMAGE TO IMAGE ===
with tab_img:
    st.info("💡 Chế độ sửa ảnh: Biến đổi ảnh gốc dựa trên prompt.")
    col1, col2 = st.columns([1, 1])
    with col1:
        uploaded_img = st.file_uploader("Upload ảnh gốc", type=["png", "jpg", "jpeg"], key="i2i_up")
        i2i_prompt = st.text_area("Prompt biến đổi", "Make it look like a pencil sketch", height=100)
        strength = st.slider("Độ biến đổi (Strength)", 0.1, 1.0, 0.75, help="Càng cao càng khác ảnh gốc")
        
        if st.button("🎨 Chuyển đổi (Img2Img)"):
            if not uploaded_img:
                st.warning("Vui lòng upload ảnh gốc!")
            else:
                # 1. Lưu ảnh vào ổ đĩa chung
                local_path = save_uploaded_file(uploaded_img)
                
                # 2. Gửi lệnh kèm đường dẫn ảnh
                payload = {
                    "prompt": i2i_prompt,
                    "task_type": "img2img",       # Báo hiệu loại task
                    "image_path": local_path,     # Đường dẫn ảnh
                    "strength": strength,
                    "width": width, "height": height, "steps": steps, "seed": seed
                }
                try:
                    r = requests.post(f"{API_URL}/generate", json=payload)
                    if r.status_code == 200:
                        job_id = r.json().get("job_id")
                        st.success(f"Job ID: {job_id}")
                        with col2:
                            res = poll_job(job_id)
                            if res: st.image(res, caption="Kết quả Img2Img")
                except Exception as e:
                    st.error(str(e))

# === TAB 3: CONTROLNET ===
with tab_ctl:
    st.info("🤖 ControlNet: Giữ lại cấu trúc (dáng, nét) của ảnh gốc.")
    col1, col2 = st.columns([1, 1])
    with col1:
        control_type = st.selectbox("Loại ControlNet", ["canny", "depth", "openpose"])
        uploaded_ref = st.file_uploader("Upload ảnh tham chiếu", type=["png", "jpg"], key="ctl_up")
        ctl_prompt = st.text_area("Prompt mô tả", "A cyberpunk warrior, neon lights", height=100)
        
        if st.button("✨ Generate (ControlNet)"):
            if not uploaded_ref:
                st.warning("Vui lòng upload ảnh tham chiếu!")
            else:
                # 1. Lưu ảnh
                local_path = save_uploaded_file(uploaded_ref)
                
                # 2. Gửi lệnh
                payload = {
                    "prompt": ctl_prompt,
                    "task_type": "controlnet",    # Báo hiệu loại task
                    "control_type": control_type, # Loại ControlNet
                    "image_path": local_path,
                    "width": width, "height": height, "steps": steps, "seed": seed
                }
                try:
                    r = requests.post(f"{API_URL}/generate", json=payload)
                    if r.status_code == 200:
                        job_id = r.json().get("job_id")
                        st.success(f"Job ID: {job_id}")
                        with col2:
                            res = poll_job(job_id)
                            if res: st.image(res, caption=f"ControlNet: {control_type}")
                except Exception as e:
                    st.error(str(e))

# === TAB 4: GALLERY ===
with tab_gallery:
    st.subheader("📂 Lịch sử ảnh đã tạo")
    if st.button("🔄 Làm mới thư viện"):
        st.rerun()
        
    if os.path.exists(OUTPUT_DIR):
        images = [f for f in os.listdir(OUTPUT_DIR) if f.endswith(('.png', '.jpg'))]
        # Sắp xếp ảnh mới nhất lên đầu
        images.sort(key=lambda x: os.path.getmtime(os.path.join(OUTPUT_DIR, x)), reverse=True)
        
        if not images:
            st.info("Chưa có ảnh nào. Hãy sang tab bên cạnh để tạo!")
        else:
            cols = st.columns(4)
            for idx, img_name in enumerate(images):
                img_path = os.path.join(OUTPUT_DIR, img_name)
                with cols[idx % 4]:
                    try:
                        image = Image.open(img_path)
                        st.image(image, caption=img_name, use_container_width=True)
                    except:
                        pass
    else:
        st.error("Thư mục data/images chưa được tạo.")