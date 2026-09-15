import os
import shutil
import random
from pathlib import Path
from typing import Dict, List, Tuple

BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "datasets"

CATEGORIES = ["real", "ai_generated", "altered", "ai_altered"]

def setup_dataset_directories():
    """Creates the dataset directory hierarchy for ML model training and evaluation."""
    for category in CATEGORIES:
        for split in ["train", "validation", "test"]:
            path = DATASET_DIR / split / category
            path.mkdir(parents=True, exist_ok=True)
    logger_msg = f"[ML Pipeline] Dataset directory hierarchy created at {DATASET_DIR}"
    print(logger_msg)

def split_dataset(
    source_dir: Path,
    target_base: Path,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    seed: int = 42
) -> Dict[str, int]:
    """
    Splits image files in source_dir into train/validation/test sets.
    Group-splits by base image identity to prevent data leakage.
    """
    random.seed(seed)
    images = list(source_dir.glob("*.jpg")) + list(source_dir.glob("*.png")) + list(source_dir.glob("*.jpeg"))
    
    if not images:
        print(f"No image files found in {source_dir}")
        return {"train": 0, "validation": 0, "test": 0}

    # Group by base identifier before file extensions to avoid leakage of cropped/augmented duplicates
    groups: Dict[str, List[Path]] = {}
    for img in images:
        base_name = img.stem.split("_aug")[0].split("_crop")[0]
        groups.setdefault(base_name, []).append(img)

    group_keys = list(groups.keys())
    random.shuffle(group_keys)

    n_groups = len(group_keys)
    n_train = int(n_groups * train_ratio)
    n_val = int(n_groups * val_ratio)

    train_groups = set(group_keys[:n_train])
    val_groups = set(group_keys[n_train:n_train + n_val])

    counts = {"train": 0, "validation": 0, "test": 0}

    category_name = source_dir.name
    for g_key, file_list in groups.items():
        if g_key in train_groups:
            split_name = "train"
        elif g_key in val_groups:
            split_name = "validation"
        else:
            split_name = "test"

        dest_dir = target_base / split_name / category_name
        dest_dir.mkdir(parents=True, exist_ok=True)

        for img_path in file_list:
            shutil.copy2(img_path, dest_dir / img_path.name)
            counts[split_name] += 1

    print(f"Category '{category_name}' split complete: {counts}")
    return counts

if __name__ == "__main__":
    setup_dataset_directories()
