"""
Utilities để visualize kết quả tìm kiếm
"""
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from typing import List
import math

def visualize_results(image_paths: List[str], scores: List[float], 
                     query: str, save_path: str = None, max_cols: int = 5):
    """
    Hiển thị kết quả tìm kiếm dưới dạng grid
    
    Args:
        image_paths: List đường dẫn ảnh
        scores: List similarity scores
        query: Text query
        save_path: Đường dẫn để lưu hình (optional)
        max_cols: Số cột tối đa
    """
    if not image_paths:
        print("No results to visualize!")
        return
    
    n_results = len(image_paths)
    n_cols = min(n_results, max_cols)
    n_rows = math.ceil(n_results / n_cols)
    
    # Tạo figure
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3, n_rows * 3.5))
    
    # Flatten axes nếu cần
    if n_results == 1:
        axes = np.array([axes])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)
    
    axes = axes.flatten()
    
    # Hiển thị từng ảnh
    for idx, (img_path, score) in enumerate(zip(image_paths, scores)):
        ax = axes[idx]
        
        try:
            img = Image.open(img_path)
            ax.imshow(img)
            ax.set_title(f"Score: {score:.3f}", fontsize=10)
            ax.axis('off')
        except Exception as e:
            ax.text(0.5, 0.5, f"Error loading image\n{str(e)}", 
                   ha='center', va='center', fontsize=8)
            ax.axis('off')
    
    # Ẩn các axes thừa
    for idx in range(n_results, len(axes)):
        axes[idx].axis('off')
    
    # Thêm title tổng
    fig.suptitle(f'Search Results for: "{query}"', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    # Lưu hoặc hiển thị
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✅ Saved visualization to: {save_path}")
    
    return fig

def create_comparison_view(query_results: dict, save_path: str = None):
    """
    So sánh kết quả của nhiều queries
    
    Args:
        query_results: Dict {query: (image_paths, scores)}
        save_path: Đường dẫn để lưu
    """
    n_queries = len(query_results)
    fig, axes = plt.subplots(n_queries, 3, figsize=(12, n_queries * 4))
    
    if n_queries == 1:
        axes = axes.reshape(1, -1)
    
    for idx, (query, (paths, scores)) in enumerate(query_results.items()):
        # Hiển thị top-3 kết quả
        for j in range(min(3, len(paths))):
            ax = axes[idx, j]
            
            try:
                img = Image.open(paths[j])
                ax.imshow(img)
                ax.set_title(f"Score: {scores[j]:.3f}", fontsize=9)
                ax.axis('off')
            except Exception as e:
                ax.text(0.5, 0.5, "Error", ha='center', va='center')
                ax.axis('off')
        
        # Label cho hàng
        axes[idx, 0].set_ylabel(f'"{query}"', fontsize=10, rotation=0, 
                                ha='right', va='center', labelpad=40)
    
    plt.suptitle('Query Comparison', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✅ Saved comparison to: {save_path}")
    
    return fig

def create_gradio_output(image_paths: List[str], scores: List[float]) -> List[tuple]:
    """
    Tạo output format cho Gradio Gallery
    
    Args:
        image_paths: List đường dẫn ảnh
        scores: List similarity scores
        
    Returns:
        List of (image_path, caption) tuples
    """
    results = []
    for path, score in zip(image_paths, scores):
        # Không hiển thị caption similarity
        results.append((path, None))
    
    return results

def plot_similarity_distribution(scores: List[float], query: str, save_path: str = None):
    """
    Vẽ phân phối similarity scores
    
    Args:
        scores: List similarity scores
        query: Text query
        save_path: Đường dẫn để lưu
    """
    plt.figure(figsize=(10, 5))
    
    # Histogram
    plt.subplot(1, 2, 1)
    plt.hist(scores, bins=20, edgecolor='black', alpha=0.7)
    plt.xlabel('Similarity Score')
    plt.ylabel('Frequency')
    plt.title('Score Distribution')
    plt.grid(True, alpha=0.3)
    
    # Box plot
    plt.subplot(1, 2, 2)
    plt.boxplot(scores, vert=True)
    plt.ylabel('Similarity Score')
    plt.title('Score Statistics')
    plt.grid(True, alpha=0.3)
    
    plt.suptitle(f'Similarity Analysis for: "{query}"', fontsize=12, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✅ Saved analysis to: {save_path}")
    
    return plt.gcf()

if __name__ == "__main__":
    print("Visualizer module loaded successfully!")
