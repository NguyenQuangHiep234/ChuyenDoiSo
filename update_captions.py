import os, json

# Đường dẫn
processed_dir = "C:/Users/vuduc/Downloads/CDS/images/processed"
json_in  = "C:/Users/vuduc/Downloads/CDS/images/captions_clean.json"
json_out = "C:/Users/vuduc/Downloads/CDS/images/captions_clean.json"


# Đọc metadata
with open(json_in, "r", encoding="utf-8") as f:
    data = json.load(f)

# Lấy danh sách file còn tồn tại sau khi bạn lọc thủ công
existing_files = set(os.listdir(processed_dir))

# Giữ lại entry nào mà file ảnh vẫn còn
filtered = []
for item in data:
    img_path = item.get("image")
    if img_path and os.path.basename(img_path) in existing_files:
        filtered.append(item)

# Lưu metadata mới
with open(json_out, "w", encoding="utf-8") as f:
    json.dump(filtered, f, ensure_ascii=False, indent=2)

print(f"✅ Metadata sau lọc thủ công: {len(filtered)} ảnh")
print(f"✅ Lưu tại: {json_out}")
