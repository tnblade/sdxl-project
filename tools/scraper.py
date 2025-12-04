# Công cụ tải ảnh từ Safebooru dựa trên từ khóa
# tools/scraper.py


import os
import requests
import argparse
from tqdm import tqdm
import time

# API của Safebooru (An toàn, không cần key, chuyên Anime)
API_URL = "https://safebooru.org/index.php"

def download_images(tags, limit, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    print(f"🔍 Đang tìm kiếm: '{tags}' | Số lượng: {limit}...")
    
    count = 0
    page = 0
    
    # Headers để giả lập trình duyệt (tránh bị chặn)
    headers = {'User-Agent': 'Mozilla/5.0'}

    pbar = tqdm(total=limit, desc="Downloading")
    
    while count < limit:
        # Gọi API lấy danh sách ảnh (XML/JSON)
        params = {
            "page": "dapi",
            "s": "post",
            "q": "index",
            "json": 1,
            "limit": 100, # Lấy 100 ảnh mỗi trang
            "pid": page,
            "tags": tags
        }
        
        try:
            response = requests.get(API_URL, params=params, headers=headers)
            if response.status_code != 200:
                print(f"❌ Lỗi kết nối: {response.status_code}")
                break
                
            posts = response.json()
            if not posts:
                print("⚠️ Hết ảnh để tải!")
                break
                
            for post in posts:
                if count >= limit: break
                
                # Bỏ qua ảnh không có URL
                if 'file_url' not in post: continue
                
                img_url = "https://safebooru.org/images/" + post['directory'] + "/" + post['image']
                
                # Tên file
                filename = f"{post['id']}.jpg"
                filepath = os.path.join(output_dir, filename)
                
                # Tải ảnh
                if not os.path.exists(filepath):
                    img_data = requests.get(img_url, headers=headers).content
                    with open(filepath, 'wb') as f:
                        f.write(img_data)
                    
                    count += 1
                    pbar.update(1)
                    
            page += 1
            time.sleep(1) # Nghỉ 1 chút để không bị server chặn
            
        except Exception as e:
            print(f"❌ Lỗi: {e}")
            break

    pbar.close()
    print(f"\n✅ Đã tải xong {count} ảnh vào thư mục: {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Anime/Genshin Image Scraper")
    parser.add_argument("--tags", type=str, required=True, help="Từ khóa (VD: genshin_impact, 1girl, solo)")
    parser.add_argument("--limit", type=int, default=20, help="Số lượng ảnh cần tải")
    parser.add_argument("--output", type=str, default="raw_images", help="Thư mục lưu")
    
    args = parser.parse_args()
    download_images(args.tags, args.limit, args.output)