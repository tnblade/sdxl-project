# ui/streamlit_app.py
import streamlit as st
import requests
import time
import os
from PIL import Image
from datetime import datetime

# --- CẤU HÌNH ---
# Lấy API URL từ biến môi trường (được set trong docker-compose)
API_URL = os.getenv("API_URL", "http://localhost:8000")
DATA_DIR = "data"
TMP_DIR = os.path.join(DATA_DIR, "tmp")
OUTPUT_DIR = os.path.join(DATA_DIR, "images")

# Đảm bảo thư mục tồn tại
os.makedirs(TMP_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- SETUP TRANG ---
st.set_page_config(
    page_title="SDXL Local Studio",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🎨 SDXL Local AI Studio")
st.markdown(f"**Server Status:** `{API_URL}` | **Mode:** Docker/CPU")

# --- SIDEBAR (CẤU HÌNH) ---
with st.sidebar:
    st.header("⚙️ Cấu hình Sinh ảnh")
    
    # Các tham số nâng cao
    width = st.select_slider("Chiều rộng (Width)", options=[512, 768, 1024], value=1024)
    height = st.select_slider("Chiều cao (Height)", options=[512, 768, 1024], value=1024)
    
    steps = st.slider("Số bước (Steps)", min_value=10, max_value=50, value=20, help="CPU nên để 20-30 cho nhanh.")
    guidance = st.slider("Guidance Scale", 1.0, 20.0, 7.5)
    seed = st.number_input("Seed (-1 để ngẫu nhiên)", value=-1, step=1)
    
    st.divider()
    st.info("ℹ️ Lưu ý: Chạy trên CPU sẽ mất 5-15 phút mỗi ảnh. Hãy kiên nhẫn!")

# --- HÀM HỖ TRỢ ---
def save_uploaded_file(uploaded_file):
    """Lưu file upload vào folder data/tmp để Worker đọc được"""
    if uploaded_file is not None:
        file_path = os.path.join(TMP_DIR, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        return file_path # Trả về đường dẫn tương đối
    return None

def poll_job(job_id):
    """Hàm chờ kết quả từ Server"""
    status_box = st.empty()
    progress_bar = st.progress(0)
    
    start_time = time.time()
    
    while True:
        try:
            resp = requests.get(f"{API_URL}/status/{job_id}")
            if resp.status_code != 200:
                status_box.error("Không thể lấy trạng thái Job.")
                break
            
            data = resp.json()
            status = data.get("status")
            
            elapsed = int(time.time() - start_time)
            
            if status == "finished":
                progress_bar.progress(100)
                status_box.success(f"✅ Hoàn tất sau {elapsed}s!")
                return data.get("result")
            
            elif status == "failed":
                status_box.error(f"❌ Thất bại: {data.get('error')}")
                return None
            
            elif status == "queued":
                status_box.info(f"⏳ Đang trong hàng đợi... ({elapsed}s)")
                progress_bar.progress(10)
                
            elif status == "started":
                status_box.warning(f"🔨 Worker đang xử lý... ({elapsed}s) - Đừng tắt tab này!")
                progress_bar.progress(50)
            
            time.sleep(3) # Hỏi server mỗi 3 giây
            
        except Exception as e:
            status_box.error(f"Lỗi kết nối: {str(e)}")
            break
    return None

# --- GIAO DIỆN CHÍNH ---

# Tạo Tabs cho các chế độ
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
        st.subheader("Nhập nội dung")
        t2i_prompt = st.text_area("Prompt (Mô tả ảnh)", "A cyberpunk city in Vietnam, neon lights, rain, highly detailed, 8k", height=150)
        t2i_negative = st.text_input("Negative Prompt (Loại bỏ)", "low quality, blurry, bad anatomy, watermark")
        
        if st.button("🚀 Tạo ảnh (Text2Img)", type="primary"):
            if not t2i_prompt:
                st.warning("Vui lòng nhập Prompt!")
            else:
                # Gửi Request
                payload = {
                    "prompt": t2i_prompt,
                    "negative_prompt": t2i_negative,
                    # Các tham số này backend phải hỗ trợ đọc, nếu chưa thì nó sẽ dùng mặc định
                    "width": width, "height": height, "steps": steps, "seed": seed
                }
                try:
                    r = requests.post(f"{API_URL}/generate", json=payload)
                    if r.status_code == 200:
                        job_id = r.json().get("job_id")
                        st.success(f"Job ID: {job_id}")
                        # Polling
                        with col2:
                            result_path = poll_job(job_id)
                            if result_path and os.path.exists(result_path):
                                st.image(result_path, caption=t2i_prompt, use_container_width=True)
                    else:
                        st.error(f"Server Error: {r.text}")
                except Exception as e:
                    st.error(f"Không kết nối được API. Bạn đã chạy Docker chưa? \nError: {e}")

# === TAB 2: IMAGE TO IMAGE ===
with tab_img:
    st.info("💡 Chế độ chỉnh sửa ảnh dựa trên ảnh gốc.")
    col1, col2 = st.columns([1, 1])
    with col1:
        uploaded_img = st.file_uploader("Upload ảnh gốc", type=["png", "jpg", "jpeg"])
        i2i_prompt = st.text_area("Prompt chỉnh sửa", "Make it looks like a pencil sketch", height=100)
        strength = st.slider("Độ mạnh (Strength) - Càng cao càng khác ảnh gốc", 0.1, 1.0, 0.7)
        
        if st.button("🎨 Chuyển đổi (Img2Img)"):
            if not uploaded_img:
                st.warning("Cần upload ảnh gốc!")
            else:
                # 1. Lưu ảnh vào Shared Volume
                local_path = save_uploaded_file(uploaded_img)
                # 2. Gửi lệnh (Lưu ý: Backend cần update để xử lý field 'image_path')
                st.warning("⚠️ Lưu ý: Chức năng này cần Backend update để xử lý ảnh đầu vào. Hiện tại UI đã sẵn sàng gửi path.")
                # Code ví dụ gửi request (khi backend sẵn sàng):
                # payload = {"prompt": i2i_prompt, "image_path": local_path, "strength": strength}
                # requests.post(...)

# === TAB 3: CONTROLNET ===
with tab_ctl:
    st.info("🤖 Điều khiển cấu trúc ảnh (Canny, Depth, OpenPose).")
    col1, col2 = st.columns([1, 1])
    with col1:
        control_type = st.selectbox("Chọn loại ControlNet", ["Canny (Viền)", "Depth (Độ sâu)", "OpenPose (Dáng)"])
        uploaded_ref = st.file_uploader("Upload ảnh tham chiếu", type=["png", "jpg"])
        ctl_prompt = st.text_area("Prompt mô tả", "A statue in a park", height=100)
        
        if st.button("Generate (ControlNet)"):
            if not uploaded_ref:
                st.warning("Cần ảnh tham chiếu!")
            else:
                ref_path = save_uploaded_file(uploaded_ref)
                st.write(f"Đã lưu ảnh tham chiếu tại: `{ref_path}`")
                st.warning("⚠️ Chức năng cần Backend load model ControlNet.")

# === TAB 4: GALLERY ===
with tab_gallery:
    st.subheader("📂 Lịch sử tạo ảnh")
    if os.path.exists(OUTPUT_DIR):
        images = [f for f in os.listdir(OUTPUT_DIR) if f.endswith(('.png', '.jpg'))]
        images.sort(key=lambda x: os.path.getmtime(os.path.join(OUTPUT_DIR, x)), reverse=True)
        
        if not images:
            st.write("Chưa có ảnh nào được tạo.")
        else:
            # Hiển thị dạng lưới
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