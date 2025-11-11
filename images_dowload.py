import os
import requests
import json
import hashlib
from tqdm import tqdm
from PIL import Image
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor, as_completed

# 🔑 API Keys
PEXELS_KEY = "xQN8670VHxdhNsyPkP5FeQQX5CTISoyMxmoz0hug1GVkyGP7LaQVEWta"

# 📁 Thư mục lưu ảnh
raw_base = r"D:/CDS/images_raw"
resized_base = r"C:/Users/vuduc/Downloads/CDS/images"
pexels_raw = os.path.join(raw_base, "pexels")
pexels_resized = os.path.join(resized_base, "pexels")
os.makedirs(pexels_raw, exist_ok=True)
os.makedirs(pexels_resized, exist_ok=True)

# 📌 Từ khóa tìm kiếm
keywords = [
    "Vietnamese people portrait", "Vietnamese woman ao dai", "Vietnamese smiling face",
    "Vietnamese market portrait", "Vietnamese family home", "Vietnamese children",
    "Vietnamese elderly", "Vietnamese farmers", "Vietnamese traditional clothes",
    "Vietnam daily life", "Vietnam street vendors"
]

captions = []
hashes = set()

# 📸 Resize ảnh
def resize_and_save(img_data, save_path, size=(512, 512)):
    try:
        img = Image.open(BytesIO(img_data)).convert("RGB")
        img = img.resize(size)
        img.save(save_path)
        return True
    except Exception as e:
        print(f"[Resize] Error: {e}")
        return False

# 🔍 Hash ảnh để loại trùng
def get_image_hash(img_data):
    return hashlib.md5(img_data).hexdigest()

# 📥 Hàm tải + xử lý 1 ảnh
def process_image(img_url, name, raw_path, resized_path, meta):
    try:
        img_data = requests.get(img_url, timeout=60).content
        img_hash = get_image_hash(img_data)
        if img_hash in hashes:
            return None
        hashes.add(img_hash)

        with open(raw_path, "wb") as f:
            f.write(img_data)
        if resize_and_save(img_data, resized_path):
            meta.update({
                "image": resized_path,
                "raw_image": raw_path
            })
            return meta
    except Exception as e:
        print(f"[Download] Error {img_url}: {e}")
    return None

# 📸 Pexels API
def fetch_pexels(query, total=300):
    per_page = 30
    headers = {"Authorization": PEXELS_KEY}
    page = 1
    results = []
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = []
        downloaded = 0
        while downloaded < total:
            url = f"https://api.pexels.com/v1/search?query={query}&per_page={per_page}&page={page}"
            res = requests.get(url, headers=headers, timeout=60)
            data = res.json()
            photos = data.get("photos", [])
            if not photos:
                break
            for photo in photos:
                if downloaded >= total:
                    break
                img_url = photo["src"].get("original") or photo["src"].get("large2x")
                name = f"{query.replace(' ', '_')}_pexels_{downloaded}.jpg"
                raw_path = os.path.join(pexels_raw, name)
                resized_path = os.path.join(pexels_resized, name)
                meta = {
                    "query": query,
                    "source": "pexels",
                    "photographer": photo.get("photographer"),
                    "url": photo.get("url")
                }
                futures.append(executor.submit(process_image, img_url, name, raw_path, resized_path, meta))
                downloaded += 1
            page += 1
        for future in tqdm(as_completed(futures), total=len(futures), desc=f"Pexels {query}"):
            result = future.result()
            if result:
                results.append(result)
    return results

# 🚀 Chạy thu thập từ Pexels
for keyword in keywords:
    captions.extend(fetch_pexels(keyword, total=800))

# 💾 Lưu JSON metadata cơ bản
with open(os.path.join(resized_base, "captions_all.json"), "w", encoding="utf-8") as f:
    json.dump(captions, f, ensure_ascii=False, indent=2)

print(f"✅ Đã tải xong ảnh từ Pexels (800 ảnh/từ khóa). Tổng số ảnh: {len(captions)}")
