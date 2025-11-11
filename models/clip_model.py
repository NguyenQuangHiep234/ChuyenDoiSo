"""Mô hình CLIP để embedding ảnh và text"""

import torch
import open_clip
from PIL import Image
import numpy as np
from typing import List, Union, Optional


class CLIPModel:
    """Wrapper cho mô hình CLIP (open_clip)"""

    def __init__(
        self,
        model_name: str = "xlm-roberta-base-ViT-B-32",
        pretrained: str = "laion5b_s13b_b90k",
        device: Optional[str] = None,
    ) -> None:
        """Khởi tạo mô hình CLIP đa ngôn ngữ"""

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.model_name = model_name
        self.pretrained = pretrained

        print(f"Loading CLIP model: {model_name} ({pretrained})")
        print(f"Using device: {self.device}")

        self.model, _, preprocess_val = open_clip.create_model_and_transforms(
            model_name,
            pretrained=pretrained,
            device=self.device,
        )
        self.preprocess = preprocess_val
        self.model.eval()
        self.tokenizer = open_clip.get_tokenizer(model_name)

        print("✅ Model loaded successfully!")
    
    def encode_image(self, image: Union[Image.Image, List[Image.Image]]) -> np.ndarray:
        """
        Encode ảnh thành vector embedding
        
        Args:
            image: PIL Image hoặc list các PIL Images
            
        Returns:
            numpy array của image embeddings (normalized)
        """
        # Xử lý single image hoặc batch
        if isinstance(image, Image.Image):
            images = [image]
        else:
            images = image
        
        # Preprocess images
        image_inputs = torch.stack([self.preprocess(img) for img in images]).to(self.device)
        
        # Encode
        with torch.no_grad():
            image_features = self.model.encode_image(image_inputs)
            # Normalize
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        
        return image_features.cpu().numpy()
    
    def encode_text(self, text: Union[str, List[str]]) -> np.ndarray:
        """
        Encode text thành vector embedding
        
        Args:
            text: String hoặc list các strings
            
        Returns:
            numpy array của text embeddings (normalized)
        """
        # Xử lý single text hoặc batch
        if isinstance(text, str):
            texts = [text]
        else:
            texts = text

        # Tokenize (open_clip tự xử lý giới hạn độ dài)
        text_inputs = self.tokenizer(texts)
        text_inputs = text_inputs.to(self.device)

        # Encode
        with torch.no_grad():
            text_features = self.model.encode_text(text_inputs)
            # Normalize
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        return text_features.cpu().numpy()
    
    def compute_similarity(
        self,
        text_embedding: np.ndarray,
        image_embeddings: np.ndarray,
    ) -> np.ndarray:
        """
        Tính similarity giữa text và images
        
        Args:
            text_embedding: Text embedding vector (1, dim)
            image_embeddings: Image embedding vectors (N, dim)
            
        Returns:
            Array của similarity scores (N,)
        """
        # Cosine similarity (đã normalized nên chỉ cần dot product)
        similarities = np.dot(image_embeddings, text_embedding.T).squeeze()
        return similarities
    
    def search(
        self,
        query_text: str,
        image_embeddings: np.ndarray,
        top_k: int = 5,
    ) -> tuple:
        """
        Tìm kiếm top-k ảnh giống nhất với query text
        
        Args:
            query_text: Text query
            image_embeddings: Embeddings của tất cả ảnh (N, dim)
            top_k: Số lượng kết quả trả về
            
        Returns:
            (indices, scores): Indices và similarity scores của top-k ảnh
        """
        # Encode query
        text_embedding = self.encode_text(query_text)
        
        # Tính similarity
        similarities = self.compute_similarity(text_embedding, image_embeddings)
        
        # Lấy top-k
        top_indices = np.argsort(similarities)[::-1][:top_k]
        top_scores = similarities[top_indices]
        
        return top_indices, top_scores
    
    def batch_encode_images(
        self,
        images: List[Image.Image],
        batch_size: int = 32,
    ) -> np.ndarray:
        """
        Encode nhiều ảnh theo batch để tránh OOM
        
        Args:
            images: List các PIL Images
            batch_size: Kích thước batch
            
        Returns:
            numpy array của tất cả image embeddings
        """
        all_embeddings = []
        
        for i in range(0, len(images), batch_size):
            batch = images[i:i+batch_size]
            embeddings = self.encode_image(batch)
            all_embeddings.append(embeddings)
        
        return np.vstack(all_embeddings)

if __name__ == "__main__":
    # Test mô hình
    print("Testing CLIP model...")
    
    model = CLIPModel()
    
    # Test với text
    texts = ["a person wearing red shirt", "a dog running in the park"]
    text_embeddings = model.encode_text(texts)
    print(f"\nText embeddings shape: {text_embeddings.shape}")
    print(f"Text embedding sample: {text_embeddings[0][:5]}")
    
    # Test với ảnh dummy
    dummy_image = Image.new('RGB', (224, 224), color='red')
    image_embedding = model.encode_image(dummy_image)
    print(f"\nImage embedding shape: {image_embedding.shape}")
    print(f"Image embedding sample: {image_embedding[0][:5]}")
    
    # Test similarity
    similarity = model.compute_similarity(text_embeddings[0:1], image_embedding)
    print(f"\nSimilarity score: {similarity[0]:.4f}")
    
    print("\n✅ All tests passed!")
