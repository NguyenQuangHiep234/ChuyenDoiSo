<h2 align="center">
    <a href="https://dainam.edu.vn/vi/khoa-cong-nghe-thong-tin">
    🎓  FACULTY OF INFORMATION TECHNOLOGY (DAINAM UNIVERSITY)
    </a>
</h2>
<h2 align="center">
    HỆ THỐNG TÌM KIẾM HÌNH ẢNH AI
</h2>

<div align="center">
    <p align="center">
        <img src="docs/aiotlab_logo.png" alt="AIoTLab Logo" width="170"/>
        <img src="docs/fitdnu_logo.png" alt="FIT DNU Logo" width="180"/>
        <img src="docs/dnu_logo.png" alt="DaiNam University Logo" width="200"/>
    </p>

[![AIoTLab](https://img.shields.io/badge/AIoTLab-green?style=for-the-badge)](https://www.facebook.com/DNUAIoTLab)
[![Faculty of Information Technology](https://img.shields.io/badge/Faculty%20of%20Information%20Technology-blue?style=for-the-badge)](https://dainam.edu.vn/vi/khoa-cong-nghe-thong-tin)
[![DaiNam University](https://img.shields.io/badge/DaiNam%20University-orange?style=for-the-badge)](https://dainam.edu.vn)

</div>

---

## 📖 1. Giới thiệu hệ thống

Hệ thống **Tìm kiếm Hình ảnh AI** được xây dựng dựa trên mô hình **OpenCLIP đa ngôn ngữ (xlm-roberta-base-ViT-B-32)** nhằm:

- Hỗ trợ tìm kiếm hình ảnh người Việt Nam thông qua mô tả văn bản bằng **Tiếng Việt** hoặc **Tiếng Anh**.
- Cung cấp giao diện web hiện đại, thân thiện với **Gradio**, cho phép người dùng nhập mô tả và nhận kết quả tức thì.
- Fine-tune model với **6686+ captions tiếng Việt** để cải thiện độ chính xác trong ngữ cảnh văn hóa Việt Nam.

✨ Các chức năng chính:

- **Fine-tuning với Captions**: Huấn luyện model với dữ liệu captions tiếng Việt để hiểu ngữ cảnh tốt hơn.
- **Tìm kiếm thông minh**: Nhập mô tả chi tiết (áo dài, nón lá, ngư dân, chợ...) và nhận ảnh phù hợp nhất.
- **Embedding Cache**: Lưu trữ embeddings để tăng tốc độ tìm kiếm.
- **Đánh giá Model**: Công cụ evaluation với confusion matrix, accuracy report.

🎯 Mục tiêu hệ thống:

- **Số hóa tra cứu hình ảnh**: Thay thế tìm kiếm thủ công bằng AI thông minh.
- **Tối ưu trải nghiệm**: Giao diện trực quan, kết quả tức thì với điểm similarity.
- **Hỗ trợ đa ngôn ngữ**: Tiếng Việt và Tiếng Anh mà không cần dịch thuật.

## � 2. Các công nghệ được sử dụng

- **Ngôn ngữ:** Python 3.9+
- **Mô hình AI:** OpenCLIP (xlm-roberta-base-ViT-B-32)
- **Framework:** PyTorch, Transformers
- **Giao diện:** Gradio Web UI
- **Phân tích:** NumPy, scikit-learn, seaborn, matplotlib
- **Fine-tuning:** Contrastive Learning với image-caption pairs

<div align="center">

[![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![OpenCLIP](https://img.shields.io/badge/OpenCLIP-412991?style=for-the-badge&logo=openai&logoColor=white)](https://github.com/mlfoundations/open_clip)
[![Gradio](https://img.shields.io/badge/Gradio-FF6F00?style=for-the-badge&logo=gradio&logoColor=white)](https://gradio.app/)
[![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)](https://numpy.org/)

</div>

## 📁 3. Cấu trúc thư mục

```
ChuyenDoiSo/
├── app.py                          # 🌐 Ứng dụng tìm kiếm Gradio
├── train.py                        # 🔥 Fine-tuning & tính embeddings
├── evaluate_model.py               # 📊 Đánh giá model với confusion matrix
├── models/
│   └── clip_model.py               # Wrapper CLIP encode ảnh & text
├── utils/
│   ├── data_loader.py              # Quản lý metadata & embedding
│   ├── search_engine.py            # Logic tìm kiếm và cache
│   └── visualizer.py               # Công cụ trực quan hoá kết quả
├── data/
│   ├── captions_draft.json         # 6686 captions tiếng Việt cho fine-tuning
│   └── processed/                  # 3003 ảnh người Việt Nam
├── trained_models/
│   ├── image_embeddings.pkl        # Embedding ảnh sau training
│   ├── fine_tuned_clip_latest.pt   # Model weights sau fine-tuning
│   ├── text_embeddings_cache.pkl   # Cache query phổ biến
│   └── training_config.json        # Cấu hình training
├── docs/                           # Logo và hình ảnh minh họa
├── requirements.txt                # Danh sách dependencies
└── README.md                       # Tài liệu dự án
```

> 💡 **Lưu ý**: File `captions_draft.json` chứa 6686 captions tiếng Việt được sử dụng để fine-tune model, giúp cải thiện độ chính xác trong ngữ cảnh văn hóa Việt Nam.

## ⚙️ 4. Các bước cài đặt & sử dụng

### 1️⃣ Chuẩn bị môi trường

- Cài đặt **Python 3.9+** → [Tải tại đây](https://www.python.org/downloads/)
- Cài đặt **Git** (optional) → [Tải tại đây](https://git-scm.com/downloads)
- Hệ điều hành: **Windows 10/11**, **Linux**, hoặc **macOS**
- RAM tối thiểu: **8GB** (khuyến nghị 16GB)
- GPU: Không bắt buộc (có GPU sẽ nhanh hơn)

### 2️⃣ Tải source code

- Clone dự án từ GitHub:
  ```bash
  git clone https://github.com/your-repo/ChuyenDoiSo.git
  cd ChuyenDoiSo
  ```
- Hoặc tải file `.zip` → giải nén.

### 3️⃣ Cài đặt dependencies

```bash
pip install -r requirements.txt
```

**Dependencies chính:**
- `torch` - PyTorch framework
- `open-clip-torch` - OpenCLIP model
- `gradio` - Web UI framework
- `pillow`, `numpy` - Xử lý ảnh và tính toán
- `scikit-learn`, `seaborn`, `matplotlib` - Đánh giá model

### 4️⃣ Kiểm tra dữ liệu

Đảm bảo cấu trúc thư mục như sau:
```
data/
├── captions_draft.json    # ✅ File captions tiếng Việt (6686 items)
└── processed/             # ✅ 3003 ảnh .jpg/.png
```

**Kiểm tra nhanh:**
```bash
python -c "from pathlib import Path; print(f'Images: {len(list(Path(\"data/processed\").glob(\"*.jpg\")))}'); print(f'Captions: {Path(\"data/captions_draft.json\").exists()}')"
```

### 5️⃣ Training model (Fine-tuning + Embeddings)

```bash
python train.py
```

**Quá trình training sẽ:**
1. Load OpenCLIP model (~1.46GB - tải lần đầu sẽ mất ~15-20 phút)
2. Fine-tune model với 6686 captions tiếng Việt (1 epoch, ~515 batches)
3. Tính embeddings cho 3003 ảnh
4. Lưu kết quả vào `trained_models/`:
   - `fine_tuned_clip_latest.pt` - Model weights sau fine-tuning
   - `image_embeddings.pkl` - Embeddings của tất cả ảnh
   - `text_embeddings_cache.pkl` - Cache query phổ biến
   - `training_config.json` - Thông tin cấu hình

**Thời gian dự kiến:**
- **CPU**: 20-40 phút
- **GPU**: 5-15 phút

### 6️⃣ Chạy ứng dụng web

```bash
python app.py
```

**Hệ thống sẽ:**
1. Load dataset (3003 ảnh)
2. Load CLIP model
3. Load embeddings từ `trained_models/`
4. Khởi động Gradio server tại: **http://127.0.0.1:7860**

### 7️⃣ Sử dụng giao diện tìm kiếm

<div align="center">
  <table>
    <tr>
      <td align="center" width="50%">
        <img src="docs/search_interface.png" alt="Search Interface" width="400"/><br/>
        <i>Hình: Giao diện tìm kiếm</i>
      </td>
      <td align="center" width="50%">
        <img src="docs/search_results.png" alt="Search Results" width="400"/><br/>
        <i>Hình: Kết quả tìm kiếm</i>
      </td>
    </tr>
  </table>
</div>

**Các bước tìm kiếm:**
1. **Nhập mô tả** - Ví dụ: "người phụ nữ mặc áo dài đỏ"
2. **Điều chỉnh tham số**:
   - Số lượng kết quả: 3-30 ảnh
   - Ngưỡng độ chính xác: 0.0-0.5 (càng cao càng strict)
3. **Nhấn "Tìm kiếm"** - Xem kết quả với điểm similarity

### 8️⃣ Đánh giá model (Optional)

```bash
python evaluate_model.py
```

Kết quả được lưu trong `evaluation_results/`:
- `confusion_matrix.png` - Ma trận nhầm lẫn
- `per_category_accuracy.png` - Độ chính xác theo category
- `classification_report.txt` - Báo cáo chi tiết
- `evaluation_summary.json` - Tổng kết metrics

### 9️⃣ Ví dụ query phổ biến

**Tiếng Việt:**
- "người phụ nữ mặc áo dài đỏ"
- "trẻ em đang vui chơi"
- "nông dân đang làm việc trên ruộng"
- "cụ già đội nón lá"
- "người bán hàng ở chợ"
- "gia đình Việt Nam sum họp"

**Tiếng Anh:**
- "elderly woman wearing traditional clothes"
- "vietnamese market seller"
- "smiling person in ao dai"
- "Vietnamese family at home"
- "fisherman working on boat"

### 🔟 Kết thúc phiên làm việc

- Đóng trình duyệt hoặc nhấn **Ctrl+C** trong terminal để dừng server
- Embeddings đã được lưu tự động, lần sau không cần train lại

✅ Sau khi hoàn tất các bước trên, bạn đã có thể sử dụng hệ thống **Tìm kiếm Hình ảnh AI** với khả năng hiểu tiếng Việt được fine-tune từ 6686 captions!

## ✨ 5. Tính năng nổi bật

- 🌐 **Đa ngôn ngữ**: Hỗ trợ tìm kiếm bằng Tiếng Việt và Tiếng Anh
- 🧠 **Fine-tuned Model**: Được huấn luyện với 6686 captions tiếng Việt
- ⚡ **Tìm kiếm tức thì**: Kết quả hiện trong vài giây
- 🎯 **Độ chính xác cao**: Similarity score cho mỗi kết quả
- 💾 **Embedding Cache**: Lưu trữ embeddings để tăng tốc
- 📊 **Evaluation Tools**: Công cụ đánh giá với confusion matrix
- 💻 **CPU/GPU Support**: Chạy được trên cả CPU và GPU
- 🎨 **Giao diện đẹp**: Gradio UI hiện đại với màu sắc Đại Nam

## 🧠 6. Quy trình hoạt động

### `train.py`

1. Load mô hình OpenCLIP (xlm-roberta-base-ViT-B-32).
2. **Fine-tuning**: Train với 6686 cặp (ảnh, caption tiếng Việt) sử dụng contrastive learning.
3. **Compute Embeddings**: Encode 3003 ảnh thành vector 512 chiều → lưu `image_embeddings.pkl`.
4. **Text Cache**: Tạo cache cho captions phổ biến → `text_embeddings_cache.pkl`.
5. Lưu config và model weights → `trained_models/`.

### `app.py`

1. Load `image_embeddings.pkl` và cache embeddings.
2. Khởi tạo Gradio web interface với giao diện Đại Nam.
3. Nhận mô tả người dùng → encode text thành vector.
4. Tính cosine similarity với tất cả ảnh.
5. Trả về top-k ảnh có similarity cao nhất, kèm điểm số.

### `evaluate_model.py`

1. Dò nhãn thật từ tên file ảnh (Vietnamese_children_, Vietnamese_elderly_, ...).
2. Dự đoán category tốt nhất qua CLIP.
3. Tạo confusion matrix và báo cáo precision/recall.
4. Xuất kết quả vào `evaluation_results/`.

## 🔧 7. Ghi chú & Khắc phục

### Lỗi thường gặp:

**❌ "Trained model not found!"**
- **Nguyên nhân**: Chưa chạy `train.py`
- **Giải pháp**: `python train.py` để tạo embeddings

**❌ "Dataset trống!"**
- **Nguyên nhân**: Thư mục `data/processed/` không có ảnh
- **Giải pháp**: Copy ảnh vào `data/processed/`

**❌ "CUDA out of memory"**
- **Nguyên nhân**: GPU không đủ RAM
- **Giải pháp**: Giảm `batch_size` trong `train.py` (dòng 716) hoặc dùng CPU

**❌ Download model chậm**
- **Nguyên nhân**: Model 1.46GB tải từ internet lần đầu
- **Giải pháp**: Đợi ~15-20 phút, lần sau sẽ dùng cache

### Tips tối ưu:

- 🚀 **Training nhanh hơn**: Dùng GPU nếu có (tự động detect)
- 💾 **Tiết kiệm RAM**: Giảm `batch_size` từ 16 xuống 8
- 🎯 **Tăng độ chính xác**: Tăng `epochs` trong FineTuneConfig (dòng 715)
- ⚡ **Tìm kiếm nhanh hơn**: Tăng `min_similarity` để lọc kết quả

## 📚 8. Tài liệu tham khảo

- [OpenCLIP Paper](https://arxiv.org/abs/2103.00020) - CLIP: Learning Transferable Visual Models
- [OpenCLIP GitHub](https://github.com/mlfoundations/open_clip) - Open source implementation
- [Gradio Documentation](https://gradio.app/docs/) - Web UI framework
- [PyTorch Documentation](https://pytorch.org/docs/) - Deep learning framework
- [XLM-RoBERTa Model](https://huggingface.co/xlm-roberta-base) - Multilingual language model

## ✉️ 9. Liên hệ

Nếu bạn cần trao đổi thêm hoặc muốn phát triển mở rộng hệ thống, vui lòng liên hệ:

- 👨‍💻 **Tác giả:** [Tên của bạn]
- 📧 **Email:** [email@example.com]
- 📱 **SĐT:** [0xxxxxxxxx]
- 🌐 **GitHub:** [github.com/yourusername]
- 🏫 **Trường:** Đại học Đại Nam - Khoa Công nghệ Thông tin

<br/>

---

<div align="center">
  <p>© 2025 AIoTLab, Faculty of Information Technology, DaiNam University. All rights reserved.</p>
  <p>Made with ❤️ using OpenCLIP & Gradio</p>
</div>
