"""
SENTINEL AI — ConvNeXt-Tiny AI Document Detector Training Pipeline

Binary classification: REAL (0) vs AI_GENERATED (1)

Dataset structure:
    dataset/ai_detector/train/real/
    dataset/ai_detector/train/ai_generated/
    dataset/ai_detector/validation/real/
    dataset/ai_detector/validation/ai_generated/
    dataset/ai_detector/test/real/
    dataset/ai_detector/test/ai_generated/

Output:
    models/convnext_tiny_ai_detector_v1.pth
    models/convnext_tiny_ai_detector_v1_metadata.json

CRITICAL:
- Never saves untrained weights as if they were trained
- Never fabricates metrics
- Only saves metadata with trained=true after actual training
"""

from pathlib import Path
import json
import random
import hashlib
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)

BASE_DIR = Path(__file__).resolve().parents[2]
DATASET_DIR = BASE_DIR / "dataset" / "ai_detector"
TRAIN_DIR = DATASET_DIR / "train"
VAL_DIR = DATASET_DIR / "validation"
TEST_DIR = DATASET_DIR / "test"

MODEL_DIR = BASE_DIR / "backend" / "app" / "models"
if not MODEL_DIR.exists():
    MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = MODEL_DIR / "convnext_base_ai_detector_v1.pth"
METADATA_PATH = MODEL_DIR / "convnext_base_ai_detector_v1_metadata.json"

IMAGE_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 1e-4
SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Expected class folders
EXPECTED_CLASSES = ["ai_generated", "real"]


def set_seed(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# Document-preserving augmentations: small rotation, mild jitter.
# NO aggressive crops, flips, or distortions that destroy forensic characteristics.
train_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomRotation(degrees=3),
    transforms.ColorJitter(brightness=0.08, contrast=0.08, saturation=0.05),
    transforms.RandomGrayscale(p=0.02),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

eval_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def create_datasets():
    """
    Load ImageFolder datasets from disk.
    Expected structure: train/{real,ai_generated}/, validation/{...}/, test/{...}/
    """
    for split, path in [("train", TRAIN_DIR), ("validation", VAL_DIR), ("test", TEST_DIR)]:
        if not path.exists():
            raise FileNotFoundError(
                f"{split} dataset not found at {path}. "
                f"Create the directory structure:\n"
                f"  {path}/real/\n"
                f"  {path}/ai_generated/\n"
                f"and populate with document images."
            )

    train_dataset = datasets.ImageFolder(TRAIN_DIR, transform=train_transform)
    val_dataset = datasets.ImageFolder(VAL_DIR, transform=eval_transform)
    test_dataset = datasets.ImageFolder(TEST_DIR, transform=eval_transform)

    # Validate class names
    if sorted(train_dataset.classes) != sorted(EXPECTED_CLASSES):
        raise ValueError(
            f"Expected classes {EXPECTED_CLASSES}, found {train_dataset.classes}. "
            f"Dataset must have 'real/' and 'ai_generated/' subdirectories."
        )
    if train_dataset.classes != val_dataset.classes:
        raise ValueError("Train and validation class mappings differ.")
    if train_dataset.classes != test_dataset.classes:
        raise ValueError("Train and test class mappings differ.")

    # Identify which class index is ai_generated
    ai_idx = train_dataset.class_to_idx.get("ai_generated")
    real_idx = train_dataset.class_to_idx.get("real")
    print(f"Class mapping: real={real_idx}, ai_generated={ai_idx}")
    print(f"Train: {len(train_dataset)} | Val: {len(val_dataset)} | Test: {len(test_dataset)}")

    return train_dataset, val_dataset, test_dataset, ai_idx


def create_model():
    """
    Create ConvNeXt-Base with binary classification head (~88M parameters).
    Uses ImageNet pretrained backbone for transfer learning.
    """
    weights = models.ConvNeXt_Base_Weights.DEFAULT
    model = models.convnext_base(weights=weights)

    # Replace classifier head: [LayerNorm, Flatten, Linear(1024, 1000)] → [LayerNorm, Flatten, Linear(1024, 1)]
    in_features = model.classifier[2].in_features
    model.classifier[2] = nn.Linear(in_features, 1)

    return model


def train_one_epoch(model, loader, criterion, optimizer, ai_class_idx):
    """Train for one epoch with binary BCE loss."""
    model.train()
    total_loss = 0.0
    total = 0
    correct = 0

    for images, labels in loader:
        images = images.to(DEVICE)
        # Convert multi-class labels to binary: 1 if ai_generated, 0 if real
        binary_labels = (labels == ai_class_idx).float().to(DEVICE)

        optimizer.zero_grad()
        logits = model(images).squeeze(1)  # [B, 1] → [B]
        loss = criterion(logits, binary_labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        predictions = (torch.sigmoid(logits) >= 0.5).float()
        correct += (predictions == binary_labels).sum().item()
        total += labels.size(0)

    return total_loss / max(total, 1), correct / max(total, 1)


@torch.no_grad()
def evaluate(model, loader, criterion, ai_class_idx):
    """Evaluate model and compute real metrics. Never fabricates."""
    model.eval()
    total_loss = 0.0
    total = 0
    all_labels = []
    all_predictions = []
    all_probabilities = []

    for images, labels in loader:
        images = images.to(DEVICE)
        binary_labels = (labels == ai_class_idx).float().to(DEVICE)

        logits = model(images).squeeze(1)
        loss = criterion(logits, binary_labels)
        probabilities = torch.sigmoid(logits)
        predictions = (probabilities >= 0.5).float()

        total_loss += loss.item() * images.size(0)
        total += labels.size(0)
        all_labels.extend(binary_labels.cpu().numpy())
        all_predictions.extend(predictions.cpu().numpy())
        all_probabilities.extend(probabilities.cpu().numpy())

    y_true = np.array(all_labels)
    y_pred = np.array(all_predictions)
    y_prob = np.array(all_probabilities)

    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )

    try:
        roc_auc = roc_auc_score(y_true, y_prob)
    except Exception:
        roc_auc = None

    try:
        pr_auc = average_precision_score(y_true, y_prob)
    except Exception:
        pr_auc = None

    cm = confusion_matrix(y_true, y_pred).tolist()

    # Compute FPR and FNR
    tn = cm[0][0] if len(cm) > 1 else 0
    fp = cm[0][1] if len(cm) > 1 and len(cm[0]) > 1 else 0
    fn = cm[1][0] if len(cm) > 1 else 0
    tp = cm[1][1] if len(cm) > 1 and len(cm[1]) > 1 else 0

    fpr = fp / max(fp + tn, 1)
    fnr = fn / max(fn + tp, 1)

    return {
        "loss": total_loss / max(total, 1),
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "roc_auc": float(roc_auc) if roc_auc is not None else None,
        "pr_auc": float(pr_auc) if pr_auc is not None else None,
        "confusion_matrix": cm,
        "false_positive_rate_on_genuine": round(fpr, 4),
        "false_negative_rate_on_ai": round(fnr, 4),
    }


def file_sha256(path: Path):
    sha = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            sha.update(chunk)
    return sha.hexdigest()


def train_ai_detector():
    """
    Main training entrypoint.
    Uses ConvNeXt-Tiny with ImageNet pretrained backbone.
    Early stopping on validation AUC.
    """
    set_seed()
    print(f"Using device: {DEVICE}")
    print(f"Architecture: ConvNeXt-Base (~88M parameters)")

    train_dataset, val_dataset, test_dataset, ai_class_idx = create_datasets()

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0
    )
    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0
    )

    model = create_model().to(DEVICE)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_val_auc = -1.0
    best_state = None
    best_epoch = -1
    patience = 5
    no_improve_count = 0
    history = []

    for epoch in range(EPOCHS):
        train_loss, train_accuracy = train_one_epoch(
            model, train_loader, criterion, optimizer, ai_class_idx
        )
        val_metrics = evaluate(model, val_loader, criterion, ai_class_idx)
        scheduler.step()

        val_auc = val_metrics.get("roc_auc") or val_metrics["f1"]

        print(
            f"Epoch {epoch + 1}/{EPOCHS} "
            f"train_loss={train_loss:.4f} "
            f"train_acc={train_accuracy:.4f} "
            f"val_auc={val_auc:.4f} "
            f"val_f1={val_metrics['f1']:.4f}"
        )

        history.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "validation": val_metrics,
        })

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_epoch = epoch + 1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve_count = 0
        else:
            no_improve_count += 1
            if no_improve_count >= patience:
                print(f"Early stopping at epoch {epoch + 1} (no improvement for {patience} epochs)")
                break

    if best_state is None:
        raise RuntimeError(
            "No trained model state was produced. Training failed. "
            "NOT saving untrained weights."
        )

    # Load best checkpoint and evaluate on test set
    model.load_state_dict(best_state)
    test_metrics = evaluate(model, test_loader, criterion, ai_class_idx)

    # Save model weights
    torch.save(model.state_dict(), MODEL_PATH)

    # Save metadata with honest training status
    metadata = {
        "model_name": "ConvNeXt-Base",
        "model_version": "convnext_base_ai_detector_v1",
        "architecture": "ConvNeXt-Base",
        "task": "real_vs_ai_generated_document",
        "num_classes": 1,
        "class_mapping": {"0": "real", "1": "ai_generated"},
        "image_size": IMAGE_SIZE,
        "trained": True,
        "calibrated": False,
        "calibration_method": None,
        "seed": SEED,
        "epochs_completed": best_epoch,
        "max_epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "device": str(DEVICE),
        "best_validation_auc": round(best_val_auc, 4),
        "test_metrics": test_metrics,
        "training_history": history,
        "weights_sha256": file_sha256(MODEL_PATH),
    }

    with open(METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print("\n" + "=" * 60)
    print("Training completed successfully.")
    print(f"Model saved: {MODEL_PATH}")
    print(f"Metadata saved: {METADATA_PATH}")
    print(f"Best epoch: {best_epoch}")
    print(f"Best validation AUC: {best_val_auc:.4f}")
    print("\nTest metrics:")
    print(json.dumps(test_metrics, indent=2))
    print("=" * 60)


if __name__ == "__main__":
    train_ai_detector()
