# config.py
import os
from pathlib import Path
from dotenv import load_dotenv  # <--- THÊM DÒNG NÀY

# 1. Load file .env ngay lập tức
# Nó sẽ tìm file .env trong cùng thư mục và nạp biến vào hệ thống
load_dotenv()

ROOT = Path(__file__).parent
MODELS_DIR = ROOT / "models"
DATA_DIR = ROOT / "data"
IMAGES_DIR = DATA_DIR / "images"
TMP_DIR = DATA_DIR / "tmp"
HISTORY_FILE = DATA_DIR / "history.json"

for p in (MODELS_DIR, DATA_DIR, IMAGES_DIR, TMP_DIR):
    p.mkdir(parents=True, exist_ok=True)

# 2. Lấy cấu hình từ .env (Lúc này nó sẽ nhận được CPU và float32)
DEVICE = os.getenv("DEVICE", "cuda") 
TORCH_DTYPE = os.getenv("TORCH_DTYPE", "float16")

# 3. Xử lý đường dẫn model
# Hàm này giúp chuyển đổi đường dẫn tương đối trong .env thành tuyệt đối
def resolve_path(path_str, default):
    if not path_str:
        return default
    # Nếu đường dẫn bắt đầu bằng "models/", nối nó với thư mục gốc để chính xác
    if path_str.startswith("models/") or path_str.startswith("./models/"):
        return str(ROOT / path_str.replace("./", ""))
    return path_str

SDXL_TEXT = resolve_path(os.getenv("SDXL_TEXT"), "stabilityai/sdxl-base")
SDXL_IMG2IMG = resolve_path(os.getenv("SDXL_IMG2IMG"), "stabilityai/sdxl-img2img")
CONTROLNET_CANNY = resolve_path(os.getenv("CONTROLNET_CANNY"), "lllyasviel/sd-controlnet-canny")
CONTROLNET_DEPTH = resolve_path(os.getenv("CONTROLNET_DEPTH"), "lllyasviel/sd-controlnet-depth")
CONTROLNET_OPENPOSE = resolve_path(os.getenv("CONTROLNET_OPENPOSE"), "lllyasviel/sd-controlnet-openpose")

CAPTION_MODEL = os.getenv("CAPTION_MODEL", "Salesforce/blip-image-captioning-large")
LLM_MODEL = os.getenv("LLM_MODEL", "meta-llama/Llama-3-8b")

DEFAULT_WIDTH = int(os.getenv("DEFAULT_WIDTH", 1024))
DEFAULT_HEIGHT = int(os.getenv("DEFAULT_HEIGHT", 1024))
GUIDANCE_SCALE = float(os.getenv("GUIDANCE_SCALE", 7.5))
IMG2IMG_STRENGTH = float(os.getenv("IMG2IMG_STRENGTH", 0.7))

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")