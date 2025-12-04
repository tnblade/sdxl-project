# fine_tuning/preprocess.py
# Tool chuẩn hóa Dataset cho SDXL
# Chuyển tất cả ảnh về định dạng JPG, crop vuông giữa và resize về 1024x1024
# Sử dụng thư viện Pillow để xử lý ảnh

import os
import argparse
from PIL import Image
from tqdm import tqdm # Thư viện tạo thanh tiến trình cho đẹp

def process_images(input_dir, output_dir, target_size=1024):
    # Tạo thư mục đầu ra nếu chưa có
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"📂 Đã tạo thư mục output: {output_dir}")

    # Lấy danh sách file ảnh
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
    files = [f for f in os.listdir(input_dir) if f.lower().endswith(valid_extensions)]
    
    print(f"🔍 Tìm thấy {len(files)} ảnh. Bắt đầu xử lý...")
    
    count = 0
    # Dùng tqdm để hiện thanh loading %
    for filename in tqdm(files, desc="Processing"):
        try:
            input_path = os.path.join(input_dir, filename)
            
            with Image.open(input_path) as img:
                # 1. Chuyển sang RGB (đề phòng ảnh PNG có nền trong suốt gây lỗi)
                img = img.convert("RGB")
                
                # 2. Tính toán để Crop vuông ở giữa (Center Crop)
                width, height = img.size
                
                # Tìm cạnh ngắn nhất
                min_dim = min(width, height)
                
                # Tính toán tọa độ cắt (Lấy tâm)
                left = (width - min_dim) / 2
                top = (height - min_dim) / 2
                right = (width + min_dim) / 2
                bottom = (height + min_dim) / 2
                
                # Cắt ảnh
                img_cropped = img.crop((left, top, right, bottom))
                
                # 3. Resize về 1024x1024 (Dùng LANCZOS để giữ nét tốt nhất)
                img_resized = img_cropped.resize((target_size, target_size), Image.Resampling.LANCZOS)
                
                # 4. Lưu ảnh
                # Đổi đuôi file thành .jpg hết
                new_filename = os.path.splitext(filename)[0] + ".jpg"
                output_path = os.path.join(output_dir, new_filename)
                
                img_resized.save(output_path, quality=95) # Chất lượng 95 là rất tốt
                count += 1
                
        except Exception as e:
            print(f"❌ Lỗi file {filename}: {e}")

    print(f"\n✅ Hoàn tất! Đã xử lý {count} ảnh.")
    print(f"👉 Ảnh chuẩn SDXL nằm tại: {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tool chuẩn hóa Dataset cho SDXL")
    parser.add_argument("--input", type=str, required=True, help="Thư mục chứa ảnh gốc")
    parser.add_argument("--output", type=str, default="dataset_ready", help="Thư mục lưu ảnh đã xử lý")
    parser.add_argument("--size", type=int, default=1024, help="Kích thước mong muốn (Mặc định 1024)")
    
    args = parser.parse_args()
    
    # Kiểm tra input
    if not os.path.exists(args.input):
        print(f"❌ Không tìm thấy thư mục input: {args.input}")
    else:
        process_images(args.input, args.output, args.size)