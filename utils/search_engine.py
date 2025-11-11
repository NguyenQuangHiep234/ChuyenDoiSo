"""
Search engine để tìm kiếm ảnh theo text query
"""
import numpy as np
from typing import List, Tuple, Optional
from deep_translator import GoogleTranslator
import re

class ImageSearchEngine:
    """Engine tìm kiếm ảnh"""
    
    def __init__(self, clip_model, image_embeddings: np.ndarray, 
                 image_paths: List[str], enable_translation: bool = True):
        """
        Khởi tạo search engine
        
        Args:
            clip_model: CLIPModel instance
            image_embeddings: Embeddings của tất cả ảnh
            image_paths: Đường dẫn của các ảnh
            enable_translation: Có dịch tiếng Việt sang tiếng Anh không
        """
        self.model = clip_model
        self.image_embeddings = image_embeddings
        self.image_paths = image_paths
        self.enable_translation = enable_translation
        
        if enable_translation:
            try:
                self.translator = GoogleTranslator(source='vi', target='en')
                print("✅ Translation enabled (Vietnamese -> English)")
            except:
                self.translator = None
                print("⚠️ Translation not available")
        else:
            self.translator = None
        
        print(f"✅ Search engine initialized with {len(image_paths)} images")
    
    def _is_vietnamese(self, text: str) -> bool:
        """
        Kiểm tra xem text có phải tiếng Việt không
        """
        # Các ký tự đặc trưng tiếng Việt
        vietnamese_chars = 'àáảãạăằắẳẵặâầấẩẫậèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵđ'
        vietnamese_chars += vietnamese_chars.upper()
        
        for char in text:
            if char in vietnamese_chars:
                return True
        return False
    
    def _translate_query(self, query: str) -> str:
        """
        Dịch query từ tiếng Việt sang tiếng Anh nếu cần
        """
        if not self.enable_translation or self.translator is None:
            return query
        
        # Kiểm tra xem có phải tiếng Việt không
        if not self._is_vietnamese(query):
            return query
        
        try:
            translated = self.translator.translate(query)
            print(f"🔄 Translated: '{query}' -> '{translated}'")
            return translated
        except Exception as e:
            print(f"⚠️ Translation error: {e}")
            return query
    
    def search(
        self,
        query: str,
        top_k: int = 5,
        auto_translate: bool = True,
    ) -> Tuple[List[str], List[float]]:
        """
        Tìm kiếm ảnh theo text query
        
        Args:
            query: Text query (tiếng Việt hoặc tiếng Anh)
            top_k: Số lượng kết quả trả về
            auto_translate: Tự động dịch tiếng Việt
            
        Returns:
            (image_paths, similarity_scores)
        """
        # Dịch query nếu cần
        if auto_translate:
            processed_query = self._translate_query(query)
        else:
            processed_query = query
        
        # Tìm kiếm
        indices, scores = self.model.search(
            processed_query, 
            self.image_embeddings, 
            top_k=top_k
        )
        
        # Lấy đường dẫn ảnh
        result_paths = [self.image_paths[idx] for idx in indices]
        result_scores = scores.tolist()
        
        return result_paths, result_scores
    
    def search_with_filters(
        self,
        query: str,
        top_k: int = 5,
        min_similarity: float = 0.0,
        auto_translate: bool = True,
    ) -> Tuple[List[str], List[float]]:
        """
        Tìm kiếm với filter theo similarity threshold
        
        Args:
            query: Text query
            top_k: Số lượng kết quả tối đa
            min_similarity: Ngưỡng similarity tối thiểu
            auto_translate: Có dịch query sang tiếng Anh hay không
            
        Returns:
            (image_paths, similarity_scores)
        """
        # Tìm kiếm thông thường
        paths, scores = self.search(
            query,
            top_k=top_k,
            auto_translate=auto_translate,
        )
        
        # Lọc theo threshold
        filtered_results = [
            (path, score) for path, score in zip(paths, scores)
            if score >= min_similarity
        ]
        
        if filtered_results:
            paths, scores = zip(*filtered_results)
            return list(paths), list(scores)
        else:
            return [], []
    
    def get_statistics(self) -> dict:
        """Lấy thống kê của search engine"""
        return {
            'total_images': len(self.image_paths),
            'embedding_dim': self.image_embeddings.shape[1],
            'translation_enabled': self.enable_translation,
            'memory_usage_mb': self.image_embeddings.nbytes / (1024 * 1024)
        }

class QuerySuggester:
    """Gợi ý query cho người dùng"""
    
    @staticmethod
    def get_sample_queries() -> List[str]:
        """Trả về các query mẫu"""
        return [
            # Tiếng Anh
            "a person wearing red shirt",
            "woman with long hair",
            "man wearing glasses",
            "person holding umbrella",
            "child playing with ball",
            "people sitting on bench",
            
            # Tiếng Việt
            "người đàn ông đeo kính",
            "người phụ nữ mặc váy xanh",
            "trẻ em chơi bóng",
            "người cầm ô",
            "người mặc áo đỏ",
            "người ngồi trên ghế"
        ]
    
    @staticmethod
    def get_query_templates() -> List[str]:
        """Trả về các template để tạo query"""
        return [
            "a person wearing [color] [clothing]",
            "a [gender] with [attribute]",
            "person [action]",
            "người [hành động]",
            "người mặc [màu sắc] [quần áo]"
        ]

if __name__ == "__main__":
    print("Search engine module loaded successfully!")
    
    # Print sample queries
    print("\n📝 Sample queries:")
    for query in QuerySuggester.get_sample_queries():
        print(f"  - {query}")
