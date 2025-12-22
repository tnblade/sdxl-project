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
            
            # Check if response is successful
            if response.status_code != 200:
                print(f"❌ Lỗi kết nối: {response.status_code}")
                # Print first 500 chars of content to debug
                print(f"Response content: {response.text[:500]}")
                break
            
            try:
                posts = response.json()
            except ValueError:
                # If JSON decode fails, it might be an empty response or HTML error
                # Safebooru sometimes returns empty string for empty results instead of []
                if not response.text.strip():
                     print("⚠️ API trả về dữ liệu rỗng (Hết ảnh hoặc lỗi server).")
                     break
                else:
                     print(f"❌ Lỗi định dạng JSON. Response text: {response.text[:200]}")
                     break

            if not posts:
                print("⚠️ Hết ảnh để tải!")
                break
                
            for post in posts:
                if count >= limit: break
                
                # Ưu tiên lấy file_url nếu có, nếu không thì tự build
                if 'file_url' in post:
                    img_url = post['file_url']
                elif 'image' in post and 'directory' in post:
                     img_url = f"https://safebooru.org/images/{post['directory']}/{post['image']}"
                else:
                    continue

                # Tên file
                filename = f"{post.get('id', int(time.time()))}.jpg"
                filepath = os.path.join(output_dir, filename)
                
                # Tải ảnh
                if not os.path.exists(filepath):
                    try:
                        img_data = requests.get(img_url, headers=headers, timeout=10).content
                        with open(filepath, 'wb') as f:
                            f.write(img_data)
                        
                        count += 1
                        pbar.update(1)
                    except Exception as img_err:
                        print(f"⚠️ Lỗi tải ảnh {img_url}: {img_err}")
                        continue
                    
            page += 1
            time.sleep(1) # Nghỉ 1 chút để không bị server chặn
            
        except Exception as e:
            print(f"❌ Lỗi vòng lặp chính: {e}")
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