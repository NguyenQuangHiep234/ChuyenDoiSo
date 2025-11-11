"""
Utilities để load và xử lý dữ liệu
"""
import json
import os
from pathlib import Path
from typing import List, Tuple
from PIL import Image
from tqdm import tqdm
import numpy as np
import pickle

class ImageDataset:
    """Class để quản lý dataset ảnh"""
    
    def __init__(self, data_dir: str = "data/processed"):
        """
        Khởi tạo dataset
        
        Args:
            data_dir: Thư mục chứa ảnh
        """
        self.data_dir = Path(data_dir)
        self.images_info = []
        self.image_paths = []
        
        if not self.data_dir.exists():
            print(f"⚠️ Thư mục {data_dir} không tồn tại!")
            return
        
        self._load_metadata()
        self._collect_images()
    
    def _load_metadata(self):
        """Load metadata từ file JSON"""
        metadata_file = self.data_dir / "sample_info.json"
        
        if metadata_file.exists():
            with open(metadata_file, 'r') as f:
                self.images_info = json.load(f)
            print(f"✅ Loaded metadata: {len(self.images_info)} images")
        else:
            print("⚠️ Không tìm thấy file metadata")
    
    def _collect_images(self):
        """Thu thập tất cả đường dẫn ảnh"""
        # Lấy tất cả file ảnh
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        
        for ext in image_extensions:
            self.image_paths.extend(self.data_dir.glob(f'*{ext}'))
        
        self.image_paths = sorted(self.image_paths)
        print(f"✅ Found {len(self.image_paths)} images")
    
    def __len__(self):
        """Trả về số lượng ảnh"""
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Tuple[Image.Image, str]:
        """
        Lấy ảnh tại index
        
        Returns:
            (image, image_path)
        """
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        return image, str(img_path)
    
    def get_all_images(self) -> List[Tuple[Image.Image, str]]:
        """
        Lấy tất cả ảnh
        
        Returns:
            List of (image, image_path)
        """
        images = []
        for i in range(len(self)):
            images.append(self[i])
        return images
    
    def load_embeddings(self, embeddings_file: str = None) -> np.ndarray:
        """
        Load embeddings đã tính toán trước
        
        Args:
            embeddings_file: Đường dẫn file embeddings
            
        Returns:
            numpy array của embeddings
        """
        if embeddings_file is None:
            embeddings_file = self.data_dir.parent / "image_embeddings.pkl"
        
        embeddings_file = Path(embeddings_file)
        
        if not embeddings_file.exists():
            print(f"⚠️ File embeddings không tồn tại: {embeddings_file}")
            return None
        
        with open(embeddings_file, 'rb') as f:
            data = pickle.load(f)
        
        print(f"✅ Loaded embeddings: {data['embeddings'].shape}")
        return data['embeddings'], data['image_paths']
    
    def save_embeddings(self, embeddings: np.ndarray, 
                       image_paths: List[str],
                       embeddings_file: str = None):
        """
        Lưu embeddings
        
        Args:
            embeddings: numpy array của embeddings
            image_paths: List đường dẫn ảnh tương ứng
            embeddings_file: Đường dẫn file để lưu
        """
        if embeddings_file is None:
            embeddings_file = self.data_dir.parent / "image_embeddings.pkl"
        
        embeddings_file = Path(embeddings_file)
        embeddings_file.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            'embeddings': embeddings,
            'image_paths': image_paths
        }
        
        with open(embeddings_file, 'wb') as f:
            pickle.dump(data, f)
        
        print(f"✅ Saved embeddings to: {embeddings_file}")

def precompute_embeddings(model, dataset: ImageDataset, 
                         batch_size: int = 32) -> Tuple[np.ndarray, List[str]]:
    """
    Tính toán embeddings cho tất cả ảnh trong dataset
    
    Args:
        model: CLIPModel instance
        dataset: ImageDataset instance
        batch_size: Kích thước batch
        
    Returns:
        (embeddings, image_paths)
    """
    print("\n" + "="*60)
    print("COMPUTING IMAGE EMBEDDINGS")
    print("="*60)
    
    all_embeddings = []
    all_paths = []
    
    # Process theo batch
    for i in tqdm(range(0, len(dataset), batch_size), desc="Encoding images"):
        batch_images = []
        batch_paths = []
        
        # Lấy batch
        for j in range(i, min(i + batch_size, len(dataset))):
            img, path = dataset[j]
            batch_images.append(img)
            batch_paths.append(path)
        
        # Encode batch
        embeddings = model.encode_image(batch_images)
        all_embeddings.append(embeddings)
        all_paths.extend(batch_paths)
    
    # Concatenate tất cả embeddings
    all_embeddings = np.vstack(all_embeddings)
    
    print(f"\n✅ Computed embeddings for {len(all_paths)} images")
    print(f"Embeddings shape: {all_embeddings.shape}")
    
    return all_embeddings, all_paths

if __name__ == "__main__":
    # Test dataset loader
    print("Testing ImageDataset...")
    
    dataset = ImageDataset()
    
    if len(dataset) > 0:
        print(f"\nDataset size: {len(dataset)}")
        
        # Test lấy 1 ảnh
        img, path = dataset[0]
        print(f"First image: {path}")
        print(f"Image size: {img.size}")
        
        print("\n✅ Dataset test passed!")
    else:
        print("\n⚠️ Dataset is empty. Run download_data.py first!")
