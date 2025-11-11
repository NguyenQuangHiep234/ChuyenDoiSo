"""
Ứng dụng Gradio cho hệ thống tìm kiếm hình ảnh
"""
import gradio as gr
import os
from pathlib import Path
import sys

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from models.clip_model import CLIPModel
from utils.data_loader import ImageDataset, precompute_embeddings
from utils.search_engine import ImageSearchEngine, QuerySuggester
from utils.visualizer import create_gradio_output

# Global variables
model = None
search_engine = None
dataset = None

def initialize_system():
    """Khởi tạo hệ thống"""
    global model, search_engine, dataset
    
    print("\n" + "="*70)
    print("KHỞI TẠO HỆ THỐNG TÌM KIẾM HÌNH ẢNH")
    print("="*70)
    
    # Load dataset
    print("\n📂 Loading dataset...")
    dataset = ImageDataset(data_dir="data/processed")
    
    if len(dataset) == 0:
        print("\n❌ Dataset trống!")
        return False
    
    # Load CLIP model
    print("\n🤖 Loading CLIP model...")
    model = CLIPModel(
        model_name="xlm-roberta-base-ViT-B-32",
        pretrained="laion5b_s13b_b90k",
    )
    
    # Load embeddings từ trained model
    print("\n🔢 Loading trained embeddings...")
    trained_embeddings_file = Path("trained_models/image_embeddings.pkl")
    
    if trained_embeddings_file.exists():
        print(f"✅ Loading from trained model: {trained_embeddings_file}")
        with open(trained_embeddings_file, 'rb') as f:
            import pickle
            data = pickle.load(f)
        metadata = data.get('metadata', {})
        saved_signature = (
            metadata.get('model_name'),
            metadata.get('pretrained'),
        )
        current_signature = (model.model_name, model.pretrained)

        if saved_signature != current_signature:
            print(
                "\n⚠️ Embeddings được tạo bằng mô hình khác."
                " Vui lòng chạy lại: python train.py"
            )
            return False
        embeddings = data['embeddings']
        image_paths = data['image_paths']
        print(f"✅ Loaded trained embeddings for {len(image_paths)} images")
    else:
        print("⚠️ Trained model not found! Please run: python train.py")
        return False
    
    # Khởi tạo search engine
    print("\n🔍 Initializing search engine...")
    search_engine = ImageSearchEngine(
        clip_model=model,
        image_embeddings=embeddings,
        image_paths=image_paths,
        enable_translation=False
    )
    
    # In thống kê
    stats = search_engine.get_statistics()
    print("\n📊 System Statistics:")
    print(f"  - Total images: {stats['total_images']}")
    print(f"  - Embedding dimension: {stats['embedding_dim']}")
    print(f"  - Translation: {'Enabled' if stats['translation_enabled'] else 'Disabled'}")
    print(f"  - Memory usage: {stats['memory_usage_mb']:.2f} MB")
    
    print("\n✅ HỆ THỐNG SẴN SÀNG!")
    print("="*70 + "\n")
    
    return True

def search_images(query, top_k, min_similarity):
    """
    Hàm xử lý tìm kiếm từ Gradio
    """
    if search_engine is None:
        return [], "❌ System not initialized! Please run: python download_data.py"
    
    if not query or query.strip() == "":
        return [], "⚠️ Vui lòng nhập mô tả để tìm kiếm!"
    
    try:
        threshold = float(min_similarity)
        k = int(top_k)

        # Tìm kiếm
        image_paths, scores = search_engine.search_with_filters(
            query=query.strip(),
            top_k=k,
            min_similarity=threshold,
            auto_translate=False,
        )
        
        if not image_paths:
            return [], (
                "❌ Không tìm thấy kết quả phù hợp. "
                "Hãy giảm 'Ngưỡng độ chính xác' hoặc thử mô tả khác."
            )
        
        # Format cho Gradio Gallery
        results = create_gradio_output(image_paths, scores)
        
        # Tạo message
        message = f"✅ Tìm thấy {len(results)} kết quả cho: '{query}'\n"
        message += f"📊 Score cao nhất: {scores[0]:.3f} | Thấp nhất: {scores[-1]:.3f}"
        message += f"\n🎯 Ngưỡng hiện tại: {threshold:.2f}"
        
        return results, message
        
    except Exception as e:
        return [], f"❌ Lỗi: {str(e)}"

def get_random_query():
    """Lấy query mẫu ngẫu nhiên"""
    import random
    queries = QuerySuggester.get_sample_queries()
    return random.choice(queries)

def create_interface():
    """Tạo giao diện Gradio mới với màu cam, trắng, xanh dương"""
    # Custom CSS mới - Tham khảo phối màu Đại Nam
    custom_css = """
    :root {
        --primary-orange: #FF6B35;
        --light-orange: #FFF5F0;
        --primary-blue: #1E40AF;
        --light-blue: #DBEAFE;
        --white: #FFFFFF;
        --text-dark: #333333;
        --text-muted: #666666;
        --border-light: #E5E7EB;
        --shadow: rgba(255, 107, 53, 0.1);
        --shadow-hover: rgba(255, 107, 53, 0.2);
    }
    .gradio-container {
        font-family: 'Inter', 'Segoe UI', sans-serif !important;
        max-width: none !important;
        width: 100% !important;
        margin: 0 !important;
        background: var(--white) !important;
        color: var(--text-dark) !important;
        padding: 20px !important;
    }
    .hero-section {
        background: linear-gradient(135deg, var(--primary-orange) 0%, var(--primary-blue) 100%) !important;
        border-radius: 16px !important;
        padding: 2rem 1.5rem !important;
        margin-bottom: 1.5rem !important;
        text-align: center !important;
        color: var(--white) !important;
        box-shadow: 0 8px 32px var(--shadow) !important;
        border: none !important;
        width: 100% !important;
        max-width: none !important;
        margin-left: 0 !important;
        margin-right: 0 !important;
    }
    .hero-section .gradio-group,
    .hero-section .gradio-markdown,
    .hero-section * {
        background: transparent !important;
        border: none !important;
        box-shadow: none !important;
        color: inherit !important;
    }
    .hero-title {
        font-size: 2.2rem !important;
        font-weight: 800 !important;
        margin: 0 0 0.8rem 0 !important;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3) !important;
        line-height: 1.2 !important;
    }
    .hero-subtitle {
        font-size: 1.1rem !important;
        margin: 0 0 0.5rem 0 !important;
        opacity: 0.9 !important;
        line-height: 1.4 !important;
    }
    .hero-meta {
        font-size: 1rem !important;
        opacity: 0.8 !important;
    }
    .search-section {
        background: var(--light-orange) !important;
        border-radius: 16px !important;
        padding: 2rem !important;
        margin-bottom: 2rem !important;
        border: 2px solid var(--primary-orange) !important;
        box-shadow: 0 8px 32px var(--shadow) !important;
    }
    .search-card:hover {
        transform: translateY(-5px) !important;
        box-shadow: 0 12px 48px rgba(0, 0, 0, 0.4) !important;
    }
    .card-title {
        color: var(--accent) !important;
        font-size: 1.5rem !important;
        font-weight: 700 !important;
        margin-bottom: 1rem !important;
        display: flex !important;
        align-items: center !important;
        gap: 0.5rem !important;
    }
    .card-desc {
        color: var(--text-muted) !important;
        margin-bottom: 1.5rem !important;
    }
    .results-card {
        background: var(--card-bg) !important;
        border-radius: 16px !important;
        border: 1px solid var(--border) !important;
        padding: 2rem !important;
        min-height: 600px !important;
    }
    .gallery-container {
        border-radius: 12px !important;
        overflow: hidden !important;
        background: rgba(255, 255, 255, 0.05) !important;
        padding: 1rem !important;
    }
    .tab-content {
        animation: fadeIn 0.5s ease-in !important;
    }
    .example-grid {
        display: grid !important;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)) !important;
        gap: 1rem !important;
        margin-top: 1rem !important;
    }
    .footer-column {
        background: rgba(255,255,255,0.04) !important;
        padding: 1rem !important;
        border-radius: 8px !important;
        border: 1px solid rgba(255,255,255,0.06) !important;
    }
    .footer-column p, .footer-column li {
        color: rgba(255,255,255,0.95) !important;
        margin: 0.35rem 0 !important;
        line-height: 1.5 !important;
        font-size: 0.98rem !important;
    }
    .footer-column ul {
        margin: 0.25rem 0 0 1rem !important;
    }
    /* Tăng cỡ chữ label của các input controls */
    label span {
        font-size: 1.15rem !important;
        font-weight: 500 !important;
    }
    /* Tăng cỡ chữ trong textbox input và status */
    .input-group textarea,
    .input-group input,
    .status-group textarea {
        font-size: 1.1rem !important;
        line-height: 1.5 !important;
    }
    .example-btn {
        background: var(--secondary-gradient) !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 0.75rem 1rem !important;
        color: white !important;
        cursor: pointer !important;
        transition: all 0.3s ease !important;
        text-align: center !important;
        font-weight: 500 !important;
    }
    .example-btn:hover {
        transform: scale(1.05) !important;
        box-shadow: 0 4px 20px rgba(245, 87, 108, 0.4) !important;
    }
    .status-card {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%) !important;
        border-radius: 12px !important;
        padding: 1.5rem !important;
        color: white !important;
        margin-top: 1rem !important;
    }
    .footer-card {
        background: var(--card-bg) !important;
        border-radius: 16px !important;
        border: 1px solid var(--border) !important;
        padding: 2rem !important;
        margin-top: 2rem !important;
        text-align: center !important;
    }
    @keyframes slideInDown {
        from { transform: translateY(-50px); opacity: 0; }
        to { transform: translateY(0); opacity: 1; }
    }
    @keyframes fadeIn {
        from { opacity: 0; }
        to { opacity: 1; }
    }
    .results-section {
        background: var(--light-blue) !important;
        border-radius: 16px !important;
        padding: 2rem !important;
        margin-bottom: 2rem !important;
        border: 2px solid var(--primary-blue) !important;
        box-shadow: 0 8px 32px rgba(30, 64, 175, 0.1) !important;
    }
    .info-section {
        background: var(--white) !important;
        border-radius: 16px !important;
        padding: 2rem !important;
        border: 2px solid var(--border-light) !important;
        box-shadow: 0 4px 20px rgba(0,0,0,0.05) !important;
    }
    .section-title {
        color: var(--primary-orange) !important;
        font-size: 3rem !important;
        font-weight: 700 !important;
        margin-bottom: 1rem !important;
        display: flex !important;
        align-items: center !important;
        gap: 0.5rem !important;
    }
    .section-desc {
        color: var(--text-muted) !important;
        margin-bottom: 1.5rem !important;
        font-size: 1.1rem !important;
    }
    .input-group {
        background: var(--white) !important;
        border-radius: 12px !important;
        padding: 1.5rem !important;
        margin-bottom: 1rem !important;
        border: 1px solid var(--border-light) !important;
        font-size: 1.5rem !important;
    }
    .button-row {
        display: flex !important;
        gap: 1rem !important;
        margin-top: 1rem !important;
    }
    .status-box {
        background: var(--light-blue) !important;
        border: 1px solid var(--primary-blue) !important;
        border-radius: 12px !important;
        padding: 1.5rem !important;
        margin-bottom: 1rem !important;
        color: var(--text-dark) !important;
    }
    .footer-section {
        background: linear-gradient(135deg, var(--primary-blue) 0%, var(--primary-orange) 100%) !important;
        color: var(--white) !important;
        border-radius: 12px !important;
        padding: 2rem 2rem !important;
        text-align: left !important;
        margin-top: 2rem !important;
        box-shadow: 0 6px 24px var(--shadow) !important;
        border: none !important;
    }
    .footer-section .gradio-group,
    .footer-section .gradio-markdown,
    .footer-section * {
        background: transparent !important;
        border: none !important;
        box-shadow: none !important;
        color: inherit !important;
        text-align: left !important;
    }
    .footer-section .section-title {
        color: var(--white) !important;
        font-size: 1.4rem !important;
        margin-bottom: 0.25rem !important;
    }
    .footer-section .section-desc {
        color: rgba(255, 255, 255, 0.92) !important;
        margin-bottom: 1rem !important;
    }
    .info-content p {
        line-height: 1.6 !important;
        margin-bottom: 1.2rem !important;
        font-size: 1rem !important;
        text-align: justify !important;
    }
    .info-content p:last-child {
        margin-bottom: 0 !important;
    }
    .info-footnote {
        font-size: 0.9rem !important;
        opacity: 0.8 !important;
        text-align: center !important;
        margin-top: 1.5rem !important;
        padding-top: 1rem !important;
        border-top: 1px solid rgba(255, 255, 255, 0.2) !important;
    }
    .info-content {
        display: flex !important;
        flex-direction: column !important;
        gap: 0.9rem !important;
        font-size: 1rem !important;
        line-height: 1.65 !important;
        color: rgba(255, 255, 255, 0.95) !important;
    }
    .info-footnote {
        margin-top: 1rem !important;
        font-size: 0.95rem !important;
        opacity: 0.85 !important;
    }
    .footer-title {
        font-size: 1.125rem !important;
        font-weight: 700 !important;
        margin-bottom: 0.5rem !important;
    }
    .footer-grid {
        display: grid !important;
        grid-template-columns: 1fr 1fr !important;
        gap: 1.5rem !important;
        margin-top: 0.75rem !important;
    }
    .footer-column h4 {
        color: var(--light-blue) !important;
        margin-bottom: 0.5rem !important;
    }
    .footer-column p {
        margin: 0.25rem 0 !important;
        opacity: 0.9 !important;
    }
    @media (max-width: 768px) {
        .hero-title { font-size: 2.5rem !important; }
        .footer-grid { grid-template-columns: 1fr !important; }
    }
    """
    
    # Theme mới với màu cam/xanh/trắng - Light mode
    theme = gr.themes.Soft(
        primary_hue="orange",
        secondary_hue="blue",
        neutral_hue="slate",
        font=gr.themes.GoogleFont("Inter")
    ).set(
        body_background_fill="#FFFFFF",
        body_background_fill_dark="#FFFFFF",
        button_primary_background_fill="linear-gradient(135deg, #FF6B35 0%, #1E40AF 100%)",
        button_primary_background_fill_hover="linear-gradient(135deg, #E55A2B 0%, #1E3A8A 100%)",
        button_primary_text_color="white",
        button_secondary_background_fill="#DBEAFE",
        button_secondary_background_fill_hover="#93C5FD",
        button_secondary_text_color="#1E40AF",
        input_background_fill="white",
        input_border_color="#E5E7EB",
        input_border_color_focus="#FF6B35",
        slider_color="#FF6B35",
        block_background_fill="white",
        block_border_color="#E5E7EB",
        block_title_text_color="#FF6B35",
        block_label_text_color="#333333",
        block_label_text_size="3rem"
    )
    
    with gr.Blocks(css=custom_css, theme=theme, title="🔍 AI Image Search") as app:
        # Hero Section
        with gr.Group(elem_classes="hero-section"):
            gr.Markdown("""
            <h1 class="hero-title">🔍 Hệ thống Tìm kiếm Hình ảnh AI</h1>
            <p class="hero-subtitle">Khám phá bộ sưu tập hình ảnh người Việt Nam với trí tuệ nhân tạo</p>
            <p class="hero-meta">🚀 Powered by OpenCLIP đa ngôn ngữ | Hỗ trợ Tiếng Việt & English</p>
            """)
        
        # Hero Section
        with gr.Group(elem_classes="hero-section"):
            gr.Markdown("""
            <div class="hero-content">
                <h1 class="hero-title">🔍 AI Image Search</h1>
                <p class="hero-subtitle">Tìm kiếm hình ảnh thông minh bằng mô tả văn bản</p>
                <div class="hero-features">
                    <div class="feature-item">🌐 Hỗ trợ đa ngôn ngữ</div>
                    <div class="feature-item">⚡ Tìm kiếm tức thời</div>
                    <div class="feature-item">🎯 Độ chính xác cao</div>
                </div>
            </div>
            """)
        
        # Search Section
        with gr.Group(elem_classes="search-section"):
            gr.Markdown("""
            <div class="section-title">📝 Tìm kiếm hình ảnh</div>
            <div class="section-desc">Nhập mô tả chi tiết bằng tiếng Việt hoặc tiếng Anh để tìm kiếm ảnh phù hợp.</div>
            """)
            
            with gr.Group(elem_classes="input-group"):
                query_input = gr.Textbox(
                    label="Mô tả hình ảnh",
                    placeholder="Ví dụ: người phụ nữ mặc áo dài đỏ, ngư dân đang đánh cá...",
                    lines=4,
                    max_lines=6,
                    show_label=False
                )
                
                with gr.Row():
                    top_k = gr.Slider(
                        minimum=3,
                        maximum=30,
                        value=12,
                        step=3,
                        label="📊 Số lượng kết quả",
                        info="Hiển thị bao nhiêu ảnh tối đa?"
                    )
                    min_similarity = gr.Slider(
                        minimum=0.0,
                        maximum=0.5,
                        value=0.05,
                        step=0.01,
                        label="🎯 Ngưỡng độ chính xác",
                        info="Độ tương đồng tối thiểu (0.0 = tất cả, 0.5 = rất chính xác)"
                    )
            
            with gr.Row(elem_classes="button-row"):
                search_btn = gr.Button("🚀 Tìm kiếm", variant="primary", size="lg")
                clear_btn = gr.Button("🗑️ Xóa", variant="secondary", size="lg")
                random_btn = gr.Button("🎲 Ví dụ ngẫu nhiên", variant="secondary", size="lg")
        
        # Results Section
        with gr.Group(elem_classes="results-section"):
            gr.Markdown("""
            <div class="section-title">🖼️ Kết quả tìm kiếm</div>
            <div class="section-desc">Ảnh phù hợp sẽ hiển thị ở đây kèm điểm số similarity.</div>
            """)
            
            with gr.Group(elem_classes="status-group"):
                status_text = gr.Textbox(
                    label="📊 Trạng thái hệ thống",
                    interactive=False,
                    lines=4,
                    show_copy_button=True,
                    value="Hệ thống sẵn sàng. Nhập mô tả và nhấn 'Tìm kiếm' để bắt đầu."
                )
            
            with gr.Group(elem_classes="gallery-group"):
                results_gallery = gr.Gallery(
                    label=None,
                    columns=4,
                    rows=3,
                    height=700,
                    object_fit="cover",
                    show_label=False,
                    preview=True,
                    show_download_button=True,
                    allow_preview=True
                )
        
        # Footer Section
        with gr.Group(elem_classes="footer-section"):
            gr.Markdown("""
            <div class="section-title">ℹ️ Về hệ thống</div>
            <div class="section-desc">Thông tin chi tiết về công nghệ và cách hoạt động.</div>
            """)
            
            with gr.Group(elem_classes="info-group"):
                gr.Markdown("""
                <div class="info-content">
                    <p>Hệ thống tìm kiếm hình ảnh AI này là một giải pháp tiên tiến được xây dựng trên nền tảng OpenCLIP đa ngôn ngữ với mô hình xlm-roberta-base-ViT-B-32, cho phép xử lý và so sánh hình ảnh với mô tả văn bản một cách chính xác và hiệu quả. Bộ dữ liệu bao gồm hơn 3000 hình ảnh chất lượng cao về người Việt Nam trong các bối cảnh đời sống hàng ngày, từ trẻ em nghịch ngợm, người cao tuổi với nón lá truyền thống, nông dân và ngư dân lao động cần cù, đến những khoảnh khắc gia đình ấm cúng và hoạt động mua bán sôi động tại chợ địa phương. Mỗi hình ảnh được mã hóa thành vector đặc trưng 512 chiều thông qua Vision Transformer, trong khi mô tả văn bản của người dùng cũng được chuyển đổi thành vector tương ứng bằng XLM-RoBERTa để đảm bảo khả năng hiểu ngữ cảnh đa ngôn ngữ. Thuật toán cosine similarity được áp dụng để tìm kiếm và xếp hạng các kết quả phù hợp nhất, với khả năng điều chỉnh ngưỡng độ chính xác và số lượng kết quả trả về để tối ưu hóa trải nghiệm. Hệ thống vận hành trên các framework hiện đại như PyTorch cho xử lý AI, Gradio cho giao diện web thân thiện, và NumPy cho tính toán hiệu quả, hỗ trợ đầy đủ cả tiếng Việt và tiếng Anh mà không cần dịch thuật bổ sung, mang lại trải nghiệm tìm kiếm nhanh chóng, trực quan và đáng tin cậy cho việc khám phá kho tàng hình ảnh văn hóa Việt Nam.</p>
                    <p class="info-footnote">© 2025 - AI Image Search System</p>
                </div>
                """)
        
        # Event handlers
        search_btn.click(
            fn=search_images,
            inputs=[query_input, top_k, min_similarity],
            outputs=[results_gallery, status_text]
        )
        query_input.submit(
            fn=search_images,
            inputs=[query_input, top_k, min_similarity],
            outputs=[results_gallery, status_text]
        )
        random_btn.click(
            fn=get_random_query,
            outputs=query_input
        )
        clear_btn.click(
            fn=clear_search,
            outputs=[query_input, results_gallery, status_text]
        )
    
    return app

def clear_search():
    """Xóa input và kết quả tìm kiếm"""
    return "", [], "Đã xóa. Nhập mô tả mới để tìm kiếm."

def main():
    """Main function"""
    
    # Khởi tạo hệ thống
    success = initialize_system()
    
    if not success:
        print("\n❌ Khởi tạo thất bại!")
        print("Vui lòng chạy: python download_data.py")
        return
    
    # Tạo và chạy app
    print("\n🚀 Đang khởi động Gradio app...\n")
    
    app = create_interface()
    
    app.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        show_error=True,
        inbrowser=True,
        favicon_path=None,
        show_api=False
    )

if __name__ == "__main__":
    main()
