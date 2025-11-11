# Hệ thống Tìm kiếm Hình ảnh Người Việt Nam theo Mô tả

## 📖 Giới thiệu

Ứng dụng này tận dụng mô hình **OpenAI CLIP (ViT-B/32)** để tìm kiếm hình ảnh người Việt Nam dựa trên mô tả bằng tiếng Việt hoặc tiếng Anh. Bộ dữ liệu đã được chuẩn hóa sẵn với hơn 3.000 ảnh thuộc 11 chủ đề (trẻ em, người cao tuổi, áo dài, nông dân, chợ, đời sống hằng ngày, v.v.).

Các thành phần chính:

- **Training & Embedding** (`train.py`): tính embedding cho toàn bộ ảnh.
- **Web Search App** (`app.py`): giao diện Gradio cho phép nhập mô tả và nhận ảnh khớp nhất.
- **Model Evaluation** (`evaluate_model.py`): tạo ma trận nhầm lẫn, báo cáo chính xác theo từng nhóm ảnh.

## 🛠️ Công nghệ sử dụng

- **Mô hình**: OpenAI CLIP (ViT-B/32)
- **Framework**: PyTorch, Transformers
- **Giao diện**: Gradio Web UI
- **Phân tích**: NumPy, scikit-learn, seaborn, matplotlib
- **Ngôn ngữ**: Python 3.9+

## 📁 Cấu trúc thư mục

```
ChuyenDoiSo/
├── app.py                     # 🌐 Ứng dụng tìm kiếm Gradio
├── train.py                   # 🔥 Tính/huấn luyện embeddings
├── evaluate_model.py          # 📊 Đánh giá model với confusion matrix
├── models/
│   └── clip_model.py          # Wrapper CLIP encode ảnh & text
├── utils/
│   ├── data_loader.py         # Quản lý metadata & embedding
│   ├── search_engine.py       # Logic tìm kiếm và cache
│   └── visualizer.py          # Công cụ trực quan hoá kết quả
├── data/
│   └── processed/             # 3000+ ảnh đã chuẩn hoá sẵn
├── trained_models/
│   ├── image_embeddings.pkl   # Embedding ảnh do train.py tạo
│   ├── text_embeddings_cache.pkl
│   └── training_config.json   # Thông tin chạy gần nhất
├── requirements.txt           # Danh sách phụ thuộc
└── README.md                  # Tài liệu dự án
```

> � **Lưu ý**: Dự án **không còn** script tải dữ liệu (`download_data.py`). Bộ ảnh đã được chuẩn bị sẵn trong `data/processed/`.

## 🚀 Bắt đầu

### 1. Cài đặt phụ thuộc

```bash
pip install -r requirements.txt
```

### 2. Kiểm tra dữ liệu

Đảm bảo `data/processed/` chứa ảnh `.jpg` và (tuỳ chọn) file `sample_info.json`. Nếu thiếu, hãy copy bộ ảnh vào thư mục này.

### 3. Tính embeddings (nếu cần)

```bash
python train.py
```

Script sẽ load mô hình CLIP, encode toàn bộ ảnh và lưu `trained_models/image_embeddings.pkl`. Nếu file này đã tồn tại, có thể bỏ qua bước này.

### 4. Đánh giá mô hình (tuỳ chọn)

```bash
python evaluate_model.py
```

Sinh ra các file trong `evaluation_results/`:

- `confusion_matrix.png`
- `per_category_accuracy.png`
- `classification_report.txt`
- `evaluation_summary.json`

Có thể chỉnh `sample_size` trong `evaluate_model.py` để giảm thời gian chạy.

### 5. Chạy ứng dụng tìm kiếm

```bash
python app.py
```

Truy cập **http://localhost:7860** để nhập mô tả và xem kết quả. Ứng dụng hỗ trợ song ngữ; mô tả tiếng Việt sẽ được dịch tự động sang tiếng Anh trước khi encode.

## 💡 Ví dụ truy vấn

**Tiếng Việt:**

- "người phụ nữ mặc áo dài đỏ"
- "trẻ em đang vui chơi"
- "nông dân đang làm việc trên ruộng"
- "cụ già đội nón lá"

**Tiếng Anh:**

- "elderly woman wearing traditional clothes"
- "vietnamese market seller"
- "smiling person in ao dai"
- "Vietnamese family at home"

## ✨ Tính năng nổi bật

- Tìm kiếm top-k ảnh theo mô tả tự nhiên (VN/EN)
- Cache embedding văn bản để tăng tốc truy vấn lặp lại
- Hiển thị điểm similarity kèm ảnh kết quả
- Bộ công cụ đánh giá giúp kiểm thử chất lượng mô hình
- Có thể chạy hoàn toàn bằng CPU

## 🧠 Quy trình hoạt động

### `train.py`

1. Load mô hình CLIP và ảnh trong `data/processed/`.
2. Encode ảnh theo batch → `image_embeddings.pkl`.
3. Lưu danh sách đường dẫn ảnh và cấu hình chạy.

### `app.py`

1. Load `image_embeddings.pkl` và cache nếu có.
2. Nhận mô tả người dùng → dịch (nếu cần) → encode.
3. So khớp cosine similarity với tất cả ảnh.
4. Trả về top-k ảnh, hiển thị kèm điểm số.

### `evaluate_model.py`

1. Dò nhãn thật từ tên file ảnh.
2. Dự đoán category tốt nhất qua CLIP.
3. Xuất confusion matrix và báo cáo precision/recall.

## 🔧 Ghi chú & Khắc phục

- Muốn đánh giá nhanh hơn → giảm `sample_size` trong `run_full_evaluation`.
- Thiếu `image_embeddings.pkl` → chạy `python train.py`.
- Dataset rỗng → copy ảnh vào `data/processed/` (không yêu cầu tải COCO).
- Cài đặt chậm → đảm bảo đã cài đúng `pip install -r requirements.txt`.

## 📚 Tài liệu tham khảo

- [CLIP Paper](https://arxiv.org/abs/2103.00020)
- [CLIP GitHub](https://github.com/openai/CLIP)
- [Gradio Documentation](https://gradio.app/docs/)

## 👨‍💻 Tác giả

Dự án Chuyển đổi số – Hệ thống tìm kiếm hình ảnh AI

## 📄 License

MIT License
