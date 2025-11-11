"""
Train/Fine-tune CLIP model với dataset người Việt Nam
"""
import numpy as np
from pathlib import Path
from tqdm import tqdm
import pickle
import json
import random
import re
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import GradScaler, autocast
from PIL import Image

from models.clip_model import CLIPModel
from utils.data_loader import ImageDataset


def _split_caption_text(text: str, max_words: int = 60) -> List[str]:
    """Tách caption dài thành nhiều câu ngắn phù hợp giới hạn CLIP"""
    if not text:
        return []

    sentences: List[str] = []

    # Tách theo dấu câu lớn
    for chunk in re.split(r'[.!?]+', text):
        chunk = chunk.strip()
        if not chunk:
            continue

        words = chunk.split()
        if len(words) <= max_words:
            if len(words) >= 3:
                sentences.append(chunk)
            continue

        # Nếu vẫn quá dài, chia thành đoạn nhỏ hơn
        for idx in range(0, len(words), max_words):
            sub_words = words[idx: idx + max_words]
            if len(sub_words) < 3:
                continue
            sentences.append(" ".join(sub_words))

    return sentences


def load_captions_from_file(file_path: Path) -> Dict[str, List[str]]:
    """Đọc file captions và trả về mapping ảnh -> danh sách mô tả"""
    if not file_path.exists():
        print(f"\n⚠️ Captions file not found: {file_path}")
        return {}

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
    except Exception as exc:
        print(f"\n❌ Không thể đọc captions từ {file_path}: {exc}")
        return {}

    captions_map: Dict[str, List[str]] = {}
    total_captions = 0

    for item in raw_data:
        image_name = item.get("image")
        captions = item.get("captions_vi") or []

        if not image_name:
            continue

        cleaned: List[str] = []
        for caption in captions:
            if not caption:
                continue
            parts = _split_caption_text(caption.strip())
            cleaned.extend(parts)

        if cleaned:
            captions_map[image_name] = cleaned
            total_captions += len(cleaned)

    if captions_map:
        print(
            f"\n🗂️ Loaded {total_captions} captions for {len(captions_map)} images "
            f"from {file_path.name}"
        )
    else:
        print(f"\n⚠️ Captions file {file_path.name} không có dữ liệu hợp lệ")

    return captions_map


@dataclass
class FineTuneConfig:
    epochs: int = 1
    batch_size: int = 16
    lr: float = 5e-6
    weight_decay: float = 0.01
    gradient_accumulation_steps: int = 1
    max_captions_per_image: int = 4
    num_workers: int = 0
    use_amp: bool = True
    save_every: int = 1
    shuffle_captions: bool = True


class CaptionImageDataset(Dataset):
    """Dataset ảnh-kèm-caption để fine-tune CLIP"""

    def __init__(
        self,
        image_dataset: ImageDataset,
        captions_map: Dict[str, List[str]],
        max_captions_per_image: int = 4,
        shuffle_captions: bool = True,
    ) -> None:
        self.dataset = image_dataset
        self.samples: List[tuple[str, str]] = []

        name_to_path = {
            Path(path).name: str(path)
            for path in image_dataset.image_paths
        }

        for image_name, captions in captions_map.items():
            if image_name not in name_to_path:
                continue

            captions_list = captions[:]
            if shuffle_captions:
                random.shuffle(captions_list)

            selected = captions_list[:max(1, max_captions_per_image)]
            for caption in selected:
                cleaned = caption.strip()
                if len(cleaned.split()) < 3:
                    continue
                self.samples.append((name_to_path[image_name], cleaned))

        if not self.samples:
            print("\n⚠️ Không tạo được cặp ảnh-caption hợp lệ để fine-tune")
        else:
            print(
                f"\n🧾 Prepared {len(self.samples)} image-caption pairs "
                f"from {len(captions_map)} captioned images"
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        image_path, caption = self.samples[idx]
        image = Image.open(image_path).convert("RGB")
        return image, caption


class CLIPTrainer:
    """Class để train/fine-tune CLIP model"""
    
    def __init__(self, model: CLIPModel, dataset: ImageDataset, 
                 output_dir: str = "trained_models"):
        """
        Khởi tạo trainer
        
        Args:
            model: CLIPModel instance
            dataset: ImageDataset instance
            output_dir: Thư mục lưu model và embeddings
        """
        self.model = model
        self.dataset = dataset
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"✅ Trainer initialized")
        print(f"📂 Output directory: {self.output_dir}")

    def _build_caption_dataloader(
        self,
        captions_map: Dict[str, List[str]],
        config: FineTuneConfig,
    ) -> Optional[DataLoader]:
        caption_dataset = CaptionImageDataset(
            image_dataset=self.dataset,
            captions_map=captions_map,
            max_captions_per_image=config.max_captions_per_image,
            shuffle_captions=config.shuffle_captions,
        )

        if len(caption_dataset) == 0:
            return None

        preprocess_fn = self.model.preprocess
        tokenizer = self.model.tokenizer

        def collate_fn(batch):
            images, texts = zip(*batch)
            tensors = []
            for img in images:
                tensors.append(preprocess_fn(img))
                if hasattr(img, "close"):
                    img.close()
            image_tensors = torch.stack(tensors)
            text_tokens = tokenizer(list(texts))
            return image_tensors, text_tokens

        loader = DataLoader(
            caption_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=config.num_workers,
            pin_memory=self.model.device == "cuda",
            drop_last=True if config.batch_size > 1 else False,
            collate_fn=collate_fn,
        )

        print(
            f"\n🧠 Caption dataloader ready: {len(loader.dataset)} pairs | "
            f"batch_size={config.batch_size}"
        )

        return loader

    def fine_tune_with_captions(
        self,
        captions_map: Dict[str, List[str]],
        config: FineTuneConfig,
    ) -> Optional[Dict[str, float]]:
        """Fine-tune CLIP bằng cặp ảnh-caption"""

        dataloader = self._build_caption_dataloader(captions_map, config)
        if dataloader is None:
            print("\n⏭️ Bỏ qua fine-tuning do không có dữ liệu hợp lệ")
            return None

        device = torch.device(self.model.device)
        model = self.model.model
        model.train()

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.lr,
            weight_decay=config.weight_decay,
        )

        scaler = GradScaler(enabled=config.use_amp and device.type == "cuda")

        total_steps = config.epochs * len(dataloader)
        global_step = 0
        metrics = {
            "epochs": config.epochs,
            "total_steps": total_steps,
            "avg_loss": 0.0,
        }

        for epoch in range(config.epochs):
            running_loss = 0.0
            progress = tqdm(
                dataloader,
                desc=f"Fine-tuning epoch {epoch + 1}/{config.epochs}",
                leave=False,
            )

            optimizer.zero_grad(set_to_none=True)
            accumulated = 0

            for step, (image_tensors, text_tokens) in enumerate(progress, start=1):
                image_tensors = image_tensors.to(device, non_blocking=True)
                text_tokens = text_tokens.to(device, non_blocking=True)

                with autocast(enabled=scaler.is_enabled()):
                    image_features = model.encode_image(image_tensors)
                    text_features = model.encode_text(text_tokens)

                    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                    text_features = text_features / text_features.norm(dim=-1, keepdim=True)

                    logit_scale = model.logit_scale.exp()
                    logits_per_image = logit_scale * image_features @ text_features.T
                    logits_per_text = logits_per_image.T

                    target = torch.arange(len(image_tensors), device=device)
                    loss_i = F.cross_entropy(logits_per_image, target)
                    loss_t = F.cross_entropy(logits_per_text, target)
                    loss = (loss_i + loss_t) * 0.5

                loss = loss / config.gradient_accumulation_steps
                scaler.scale(loss).backward()
                accumulated += 1

                if accumulated >= config.gradient_accumulation_steps:
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                    accumulated = 0

                    with torch.no_grad():
                        model.logit_scale.data = torch.clamp(model.logit_scale.data, max=np.log(100.0))

                running_loss += loss.item() * config.gradient_accumulation_steps
                global_step += 1

                if global_step % 10 == 0:
                    progress.set_postfix({"loss": f"{running_loss / step:.4f}"})

            if accumulated > 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                with torch.no_grad():
                    model.logit_scale.data = torch.clamp(model.logit_scale.data, max=np.log(100.0))

            epoch_loss = running_loss / max(1, len(dataloader))
            metrics.setdefault("epoch_losses", []).append(epoch_loss)
            print(
                f"\n📉 Epoch {epoch + 1}/{config.epochs} finished | "
                f"loss={epoch_loss:.4f}"
            )

            if config.save_every and ((epoch + 1) % config.save_every == 0):
                self.save_finetuned_weights(suffix=f"epoch_{epoch + 1}")

        model.eval()

        metrics["avg_loss"] = float(np.mean(metrics.get("epoch_losses", [0.0])))
        print(
            f"\n✅ Fine-tuning completed | epochs={config.epochs} | "
            f"avg_loss={metrics['avg_loss']:.4f}"
        )

        self.save_finetuned_weights(suffix="latest")

        return metrics

    def save_finetuned_weights(self, suffix: str = "latest") -> Path:
        """Lưu trọng số mô hình sau khi fine-tune"""

        weights_path = self.output_dir / f"fine_tuned_clip_{suffix}.pt"
        weights_path.parent.mkdir(parents=True, exist_ok=True)

        torch.save(
            {
                "model_name": self.model.model_name,
                "pretrained": self.model.pretrained,
                "state_dict": self.model.model.state_dict(),
                "logit_scale": float(self.model.model.logit_scale.exp().item()),
            },
            weights_path,
        )

        print(f"💾 Saved fine-tuned weights to: {weights_path}")
        return weights_path
    
    def prepare_embeddings(self, batch_size: int = 32, force_recompute: bool = False):
        """
        Chuẩn bị embeddings cho tất cả ảnh trong dataset
        
        Args:
            batch_size: Kích thước batch để encode
            force_recompute: Bắt buộc tính lại embeddings
        """
        embeddings_file = self.output_dir / "image_embeddings.pkl"

        # Kiểm tra xem đã có embeddings chưa
        existing_data = None
        if embeddings_file.exists():
            with open(embeddings_file, 'rb') as f:
                existing_data = pickle.load(f)

        if existing_data and not force_recompute:
            metadata = existing_data.get('metadata', {})
            current_signature = (self.model.model_name, self.model.pretrained)
            saved_signature = (
                metadata.get('model_name'),
                metadata.get('pretrained'),
            )

            if saved_signature == current_signature:
                print(f"\n✅ Loading existing embeddings from {embeddings_file}")
                return (
                    existing_data['embeddings'],
                    existing_data['image_paths'],
                    metadata,
                )

            print(
                "\n⚠️ Existing embeddings belong to a different model. "
                "Recomputing with current settings..."
            )
        
        print("\n" + "="*70)
        print("COMPUTING IMAGE EMBEDDINGS")
        print("="*70)
        
        all_embeddings = []
        all_paths = []
        
        # Tính embeddings theo batch
        n_images = len(self.dataset)
        n_batches = (n_images + batch_size - 1) // batch_size
        
        with tqdm(total=n_images, desc="Processing images") as pbar:
            for i in range(0, n_images, batch_size):
                batch_images = []
                batch_paths = []
                
                # Lấy batch
                for j in range(i, min(i + batch_size, n_images)):
                    try:
                        img, path = self.dataset[j]
                        batch_images.append(img)
                        batch_paths.append(path)
                    except Exception as e:
                        print(f"\n⚠️ Error loading image {j}: {e}")
                        continue
                
                if not batch_images:
                    continue
                
                # Encode batch
                try:
                    embeddings = self.model.encode_image(batch_images)
                    all_embeddings.append(embeddings)
                    all_paths.extend(batch_paths)
                    pbar.update(len(batch_images))
                except Exception as e:
                    print(f"\n⚠️ Error encoding batch: {e}")
                    continue
        
        # Concatenate tất cả embeddings
        if not all_embeddings:
            raise ValueError("No embeddings were computed!")
        
        all_embeddings = np.vstack(all_embeddings)
        
        print(f"\n✅ Computed embeddings for {len(all_paths)} images")
        print(f"📊 Embeddings shape: {all_embeddings.shape}")
        
        # Lưu embeddings
        metadata = {
            'n_images': len(all_paths),
            'embedding_dim': all_embeddings.shape[1],
            'model_name': self.model.model_name,
            'pretrained': self.model.pretrained,
            'dataset_path': str(self.dataset.data_dir)
        }
        
        data = {
            'embeddings': all_embeddings,
            'image_paths': all_paths,
            'metadata': metadata
        }
        
        with open(embeddings_file, 'wb') as f:
            pickle.dump(data, f)
        
        print(f"💾 Saved embeddings to: {embeddings_file}")
        
        return all_embeddings, all_paths, metadata
    
    def create_text_embeddings_cache(self, common_queries: list):
        """
        Tạo cache cho các query phổ biến
        
        Args:
            common_queries: List các query thường dùng
        """
        print("\n" + "="*70)
        print("CREATING TEXT EMBEDDINGS CACHE")
        print("="*70)
        
        cache_file = self.output_dir / "text_embeddings_cache.pkl"
        
        text_cache = {}
        
        print(f"\nProcessing {len(common_queries)} queries...")
        for query in tqdm(common_queries):
            try:
                embedding = self.model.encode_text(query)
                text_cache[query] = embedding
            except Exception as e:
                print(f"⚠️ Error encoding '{query}': {e}")
        
        # Lưu cache
        with open(cache_file, 'wb') as f:
            pickle.dump(text_cache, f)
        
        print(f"\n✅ Created cache for {len(text_cache)} queries")
        print(f"💾 Saved to: {cache_file}")
        
        return text_cache
    
    def save_training_config(self, config: dict):
        """
        Lưu cấu hình training
        
        Args:
            config: Dictionary chứa thông tin cấu hình
        """
        config_file = self.output_dir / "training_config.json"
        
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        
        print(f"💾 Saved training config to: {config_file}")
    
    def train(self, batch_size: int = 32, 
              common_queries: list = None,
              force_recompute: bool = False,
              config_extra: dict = None,
              captions_map: Optional[Dict[str, List[str]]] = None,
              fine_tune_config: Optional[FineTuneConfig] = None):
        """
        Main training pipeline
        
        Args:
            batch_size: Kích thước batch
            common_queries: List các query phổ biến để cache
            force_recompute: Bắt buộc tính lại embeddings
            config_extra: Thông tin bổ sung để lưu trong config
            captions_map: Mapping ảnh -> captions để fine-tune
            fine_tune_config: Cấu hình fine-tune
        """
        print("\n" + "="*70)
        print("🚀 STARTING TRAINING PIPELINE")
        print("="*70)

        fine_tune_metrics = None
        used_fine_tune_config: Optional[FineTuneConfig] = None

        if captions_map:
            print("\n🧑‍🏫 Step 0: Fine-tuning with captions...")
            ft_config = fine_tune_config or FineTuneConfig()
            used_fine_tune_config = ft_config
            fine_tune_metrics = self.fine_tune_with_captions(captions_map, ft_config)
            if fine_tune_metrics:
                force_recompute = True
            else:
                print("\n⚠️ Fine-tuning skipped due to missing caption pairs")
        
        # 1. Prepare image embeddings
        print("\n📸 Step 1: Preparing image embeddings...")
        embeddings, image_paths, metadata = self.prepare_embeddings(
            batch_size=batch_size,
            force_recompute=force_recompute
        )
        
        # 2. Create text embeddings cache (optional)
        if common_queries:
            print("\n📝 Step 2: Creating text embeddings cache...")
            text_cache = self.create_text_embeddings_cache(common_queries)
        else:
            print("\n⏭️ Step 2: Skipped (no common queries provided)")
            text_cache = None
        
        # 3. Save training config
        print("\n⚙️ Step 3: Saving training configuration...")
        config = {
            'model_name': self.model.model_name,
            'pretrained': self.model.pretrained,
            'dataset_path': str(self.dataset.data_dir),
            'n_images': len(image_paths),
            'embedding_dim': embeddings.shape[1],
            'batch_size': batch_size,
            'device': self.model.device,
            'text_cache_size': len(text_cache) if text_cache else 0
        }
        if config_extra:
            config.update(config_extra)
        if fine_tune_metrics and used_fine_tune_config:
            config['fine_tune'] = {
                **asdict(used_fine_tune_config),
                'metrics': fine_tune_metrics
            }
        self.save_training_config(config)
        
        print("\n" + "="*70)
        print("✅ TRAINING COMPLETED!")
        print("="*70)
        print(f"\n📊 Summary:")
        print(f"  - Images processed: {len(image_paths)}")
        print(f"  - Embedding dimension: {embeddings.shape[1]}")
        print(f"  - Model saved to: {self.output_dir}")
        print(f"  - Memory usage: {embeddings.nbytes / 1024 / 1024:.2f} MB")
        
        if text_cache:
            print(f"  - Text cache: {len(text_cache)} queries")
        if fine_tune_metrics:
            print(
                f"  - Fine-tuned for {fine_tune_metrics['epochs']} epochs | "
                f"avg loss: {fine_tune_metrics['avg_loss']:.4f}"
            )
        
        return {
            'embeddings': embeddings,
            'image_paths': image_paths,
            'metadata': metadata,
            'text_cache': text_cache,
            'output_dir': str(self.output_dir),
            'fine_tune_metrics': fine_tune_metrics,
            'fine_tune_config': asdict(used_fine_tune_config) if used_fine_tune_config else None,
        }

def get_vietnamese_queries():
    """Lấy các query tiếng Việt phổ biến"""
    return [
        # Mô tả chung
        "người đàn ông",
        "người phụ nữ",
        "trẻ em",
        "người già",
        
        # Trang phục
        "người mặc áo đỏ",
        "người mặc áo xanh",
        "người mặc áo trắng",
        "người mặc áo đen",
        "người mặc váy",
        "người mặc quần jean",
        "người mặc áo dài",
        "người mặc đồng phục",
        
        # Phụ kiện
        "người đeo kính",
        "người đội mũ",
        "người đeo túi xách",
        "người cầm ô",
        "người đeo ba lô",
        
        # Đặc điểm
        "người tóc dài",
        "người tóc ngắn",
        "người có râu",
        "người béo",
        "người gầy",
        "người cao",
        "người thấp",
        
        # Hành động
        "người đang đi bộ",
        "người đang chạy",
        "người đang ngồi",
        "người đang đứng",
        "người đang nằm",
        "người đang ăn",
        "người đang uống",
        "người đang cười",
        
        # Địa điểm
        "người ở công viên",
        "người ở bãi biển",
        "người ở trong nhà",
        "người ở ngoài trời",
        
        # Số lượng
        "một người",
        "hai người",
        "nhiều người",
        "nhóm người"
    ]

def main():
    """Main training function"""
    print("\n" + "="*70)
    print("HỆ THỐNG HUẤN LUYỆN MÔ HÌNH TÌM KIẾM HÌNH ẢNH")
    print("="*70)
    
    # Kiểm tra dataset - sử dụng data/processed trực tiếp
    dataset = ImageDataset(data_dir="data/processed")
    
    if len(dataset) == 0:
        print("\n❌ Dataset trống!")
        print("Vui lòng đảm bảo thư mục data/processed chứa ảnh .jpg/.png.")
        return
    
    print(f"\n✅ Dataset loaded: {len(dataset)} images")
    
    # Load model
    print("\n🤖 Loading CLIP model...")
    model = CLIPModel(
        model_name="xlm-roberta-base-ViT-B-32",
        pretrained="laion5b_s13b_b90k",
    )
    
    # Khởi tạo trainer
    trainer = CLIPTrainer(
        model=model,
        dataset=dataset,
        output_dir="trained_models"
    )
    
    # Load captions nếu có
    captions_file = None
    captions_map: Dict[str, List[str]] = {}
    for candidate_name in ["captions_draft.json", "captions_draft (2).json", "captions_draft_final.json"]:
        candidate_path = Path("data") / candidate_name
        if candidate_path.exists():
            captions_map = load_captions_from_file(candidate_path)
            captions_file = candidate_path
            if captions_map:
                break

    fine_tune_cfg = None

    if captions_map:
        # Sử dụng toàn bộ caption làm cache truy vấn
        vietnamese_queries = sorted({cap for caps in captions_map.values() for cap in caps})
        print(f"\n📝 Prepared {len(vietnamese_queries)} unique captions for caching")

        fine_tune_cfg = FineTuneConfig(
            epochs=1,
            batch_size=16,
            lr=5e-6,
            weight_decay=0.01,
            max_captions_per_image=4,
            gradient_accumulation_steps=1,
            num_workers=0,
            use_amp=(model.device == "cuda"),
            save_every=0,
            shuffle_captions=True,
        )
    else:
        vietnamese_queries = get_vietnamese_queries()
        print(f"\n📝 Captions file không khả dụng, dùng {len(vietnamese_queries)} query mặc định")

    config_extra = {
        'captions_count': len(vietnamese_queries) if captions_map else 0,
        'captions_images': len(captions_map) if captions_map else 0,
        'text_cache_source': 'captions' if captions_map else 'default_queries',
        'fine_tune_enabled': bool(captions_map)
    }
    if captions_file:
        config_extra['captions_file'] = str(captions_file)
    
    # Train
    results = trainer.train(
        batch_size=32,
        common_queries=vietnamese_queries,
        force_recompute=False,  # Đặt True nếu muốn tính lại
        config_extra=config_extra,
        captions_map=captions_map if captions_map else None,
        fine_tune_config=fine_tune_cfg,
    )
    
    print("\n" + "="*70)
    print("🎉 HOÀN TẤT HUẤN LUYỆN!")
    print("="*70)
    print("\nBây giờ bạn có thể chạy ứng dụng web:")
    print("  python app.py")
    print("\n")

if __name__ == "__main__":
    main()
