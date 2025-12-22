# Công cụ tải ảnh từ Safebooru dựa trên từ khóa
# tools/scraper.py

import os
import requests
import argparse
from tqdm import tqdm
import time

API_URL = "https://safebooru.org/index.php"

# Tag cấm – lọc ảnh sexy / ecchi
BANNED_TAGS = {
    "bikini", "swimsuit", "under_boob", "sideboob",
    "large_breasts", "ass", "ass_focus",
    "cleavage", "micro_bikini",
    "thighs", "armpits", "pantyshot",
    "nude", "nsfw", "ecchi"
}

def download_images(tags, limit, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    print(f"🔍 Đang tìm kiếm: '{tags}' | Số lượng: {limit}")

    count = 0
    page = 0

    headers = {
        "User-Agent": "Mozilla/5.0"
    }

    pbar = tqdm(total=limit, desc="Downloading")

    while count < limit:
        params = {
            "page": "dapi",
            "s": "post",
            "q": "index",
            "json": 1,
            "limit": 100,
            "pid": page,
            "tags": tags
        }

        try:
            response = requests.get(API_URL, params=params, headers=headers, timeout=15)

            if response.status_code != 200:
                print(f"❌ HTTP {response.status_code}")
                break

            try:
                posts = response.json()
                if not isinstance(posts, list):
                    print("⚠️ API không trả list.")
                    page += 1
                    continue
            except Exception:
                if response.text.strip() == "":
                    print(f"⚠️ Page {page}: rỗng")
                    page += 1
                    continue
                else:
                    print(f"❌ JSON lỗi: {response.text[:200]}")
                    break

            # ❗ QUAN TRỌNG: KHÔNG BREAK KHI PAGE RỖNG
            if not posts:
                print(f"⚠️ Page {page}: không có post, skip")
                page += 1
                continue

            for post in posts:
                if count >= limit:
                    break

                post_tags = post.get("tags", "").split()

                # Chỉ lấy ảnh general
                if post.get("rating") != "general":
                    continue

                # ❌ BỎ ÉP highres (nguyên nhân 0 ảnh)
                # if "highres" not in post_tags:
                #     continue

                # Lọc sexy / ecchi
                if any(tag in BANNED_TAGS for tag in post_tags):
                    continue

                # Build URL ảnh
                if post.get("file_url"):
                    img_url = post["file_url"]
                elif post.get("directory") and post.get("image"):
                    img_url = f"https://safebooru.org/images/{post['directory']}/{post['image']}"
                else:
                    continue

                filename = f"{post.get('id')}.jpg"
                filepath = os.path.join(output_dir, filename)

                if os.path.exists(filepath):
                    continue

                try:
                    img_resp = requests.get(img_url, headers=headers, timeout=15)
                    if img_resp.status_code == 200:
                        with open(filepath, "wb") as f:
                            f.write(img_resp.content)
                        count += 1
                        pbar.update(1)
                except Exception as e:
                    print(f"⚠️ Lỗi tải ảnh: {e}")

            page += 1
            time.sleep(1)

        except Exception as e:
            print(f"❌ Lỗi vòng lặp chính: {e}")
            break

    pbar.close()
    print(f"\n✅ Đã tải {count} ảnh vào: {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Safebooru Anime Image Scraper (Clean Dataset)"
    )
    parser.add_argument(
        "--tags", type=str, required=True,
        help="Tag Safebooru (VD: scenery landscape no_humans rating:general)"
    )
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--output", type=str, default="raw_images")

    args = parser.parse_args()
    download_images(args.tags, args.limit, args.output)
