import os
import random
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional

try:
    import torch
    from torch.utils.data import Dataset
    from torchvision import transforms
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from PIL import Image, ImageEnhance, ImageFilter, ImageOps

DEFAULT_DATASET_DIR = Path(__file__).resolve().parent.parent.parent.parent / "datasets"

def setup_dataset_directories(base_dir: Optional[Path] = None) -> Dict[str, Path]:
    """
    Creates and returns the dataset directory structure:
    datasets/
      train/ (genuine, ai_generated)
      validation/ (genuine, ai_generated)
      test/ (genuine, ai_generated)
      tampered/
    """
    root = base_dir or DEFAULT_DATASET_DIR
    paths = {
        "train_genuine": root / "train" / "genuine",
        "train_ai": root / "train" / "ai_generated",
        "val_genuine": root / "validation" / "genuine",
        "val_ai": root / "validation" / "ai_generated",
        "test_genuine": root / "test" / "genuine",
        "test_ai": root / "test" / "ai_generated",
        "tampered": root / "tampered",
    }
    for p in paths.values():
        p.mkdir(parents=True, exist_ok=True)
    return paths


class RealisticAugmentation:
    """
    Realistic image augmentations for identity document training:
    - JPEG compression (quality 40-95)
    - Resizing / Downsampling (WhatsApp-style compression)
    - Brightness / Contrast jitter
    - Gaussian blur
    - Mild rotation (-5 to +5 deg)
    """
    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() > self.p:
            return img

        # 1. Mild rotation
        if random.random() < 0.3:
            angle = random.uniform(-5, 5)
            img = img.rotate(angle, resample=Image.BICUBIC, expand=False)

        # 2. Brightness & Contrast
        if random.random() < 0.4:
            enhancer = ImageEnhance.Brightness(img)
            img = enhancer.enhance(random.uniform(0.8, 1.2))
            enhancer = ImageEnhance.Contrast(img)
            img = enhancer.enhance(random.uniform(0.8, 1.2))

        # 3. Gaussian Blur
        if random.random() < 0.3:
            radius = random.uniform(0.5, 1.5)
            img = img.filter(ImageFilter.GaussianBlur(radius=radius))

        # 4. JPEG Compression simulation
        if random.random() < 0.4:
            import io
            quality = random.randint(40, 90)
            buffer = io.BytesIO()
            img.save(buffer, format="JPEG", quality=quality)
            buffer.seek(0)
            img = Image.open(buffer).convert("RGB")

        return img


if HAS_TORCH:
    class DocumentAIDataset(Dataset):
        """
        PyTorch Dataset for document AI detection.
        Loads images from dataset directory with label 0=genuine, 1=ai_generated.
        """
        def __init__(
            self,
            image_paths: List[Path],
            labels: List[int],
            img_size: Tuple[int, int] = (224, 224),
            augment: bool = False,
        ):
            self.image_paths = image_paths
            self.labels = labels
            self.img_size = img_size
            self.augment = augment
            self.augmentor = RealisticAugmentation(p=0.7)

            self.to_tensor_transform = transforms.Compose([
                transforms.Resize(img_size),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])

        def __len__(self) -> int:
            return len(self.image_paths)

        def __getitem__(self, idx: int):
            path = self.image_paths[idx]
            label = self.labels[idx]
            
            try:
                img = Image.open(path).convert("RGB")
                if self.augment:
                    img = self.augmentor(img)
                tensor = self.to_tensor_transform(img)
            except Exception:
                tensor = torch.zeros((3, self.img_size[0], self.img_size[1]), dtype=torch.float32)
                
            return tensor, torch.tensor(label, dtype=torch.float32)


def get_split_files(split_dir: Path) -> Tuple[List[Path], List[int]]:
    """
    Returns image paths and labels (0 for genuine, 1 for ai_generated) in a given split directory.
    """
    genuine_dir = split_dir / "genuine"
    ai_dir = split_dir / "ai_generated"

    valid_exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    paths = []
    labels = []

    if genuine_dir.exists():
        for p in genuine_dir.iterdir():
            if p.suffix.lower() in valid_exts and p.is_file():
                paths.append(p)
                labels.append(0)

    if ai_dir.exists():
        for p in ai_dir.iterdir():
            if p.suffix.lower() in valid_exts and p.is_file():
                paths.append(p)
                labels.append(1)

    return paths, labels
