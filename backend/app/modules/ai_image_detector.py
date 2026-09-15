"""
SENTINEL AI — ConvNeXt-Tiny AI-Generated Document Image Detector

Binary classifier: REAL (0) vs AI_GENERATED (1)
Produces a single canonical `ai_probability` via sigmoid.

Never runs inference on untrained/random weights.
Returns ai_probability=None when model is unavailable.
"""

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, Optional

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

from app.modules.patch_detector import predict_patch_ai_probabilities

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent.parent
MODEL_DIR = BASE_DIR / "backend" / "app" / "models"
if not MODEL_DIR.exists():
    MODEL_DIR = Path(__file__).resolve().parent.parent / "models"

# ConvNeXt-Base checkpoint paths
MODEL_PATH = MODEL_DIR / "convnext_base_ai_detector_v1.pth"
METADATA_PATH = MODEL_DIR / "convnext_base_ai_detector_v1_metadata.json"

# Legacy fallback paths (deprecated EfficientNet/Tiny)
LEGACY_MODEL_PATH_V2 = MODEL_DIR / "convnext_tiny_ai_detector_v1.pth"
LEGACY_MODEL_PATH_V1 = MODEL_DIR / "ai_detector_v1.pth"

DEFAULT_MODEL_VERSION = "convnext_base_ai_detector_v1"
IMAGE_SIZE = 224

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ConvNeXtBaseAIDetector(nn.Module):
    """
    ConvNeXt-Base binary classifier for document authenticity (~88M parameters).
    Output: single logit → sigmoid → P(AI_GENERATED).
    """

    def __init__(self):
        super().__init__()
        self.backbone = models.convnext_base(weights=None)
        # ConvNeXt-Base classifier head: Sequential(LayerNorm, Flatten, Linear)
        # The Linear layer is at index 2 with in_features=1024
        in_features = self.backbone.classifier[2].in_features
        self.backbone.classifier[2] = nn.Linear(in_features, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


class AIImageDetector:
    """
    Singleton detector service.

    Safety guarantees:
    - Never runs inference with untrained weights
    - Returns ai_probability=None when model unavailable
    - Reports consistent model_version from metadata
    """

    _instance: Optional["AIImageDetector"] = None

    def __init__(
        self,
        model_path: Optional[Path] = None,
        metadata_path: Optional[Path] = None,
    ):
        self.model_path = Path(model_path) if model_path else MODEL_PATH
        self.metadata_path = Path(metadata_path) if metadata_path else METADATA_PATH
        self.model = None
        self.model_loaded = False
        self.model_version = DEFAULT_MODEL_VERSION
        self.is_trained = False
        self.is_calibrated = False
        self.calibration_method = None

        self.transform = transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

        self._load_metadata()
        self._load_model()

    @classmethod
    def get_instance(cls) -> "AIImageDetector":
        if cls._instance is None:
            cls._instance = cls()
        elif not cls._instance.model_loaded:
            cls._instance._load_metadata()
            cls._instance._load_model()
        return cls._instance

    def _load_metadata(self):
        """Load and validate model metadata JSON."""
        if not self.metadata_path.exists():
            logger.warning(f"AI detector metadata missing at {self.metadata_path}")
            return

        try:
            with open(self.metadata_path, "r", encoding="utf-8") as f:
                meta = json.load(f)

            self.model_version = meta.get("model_version", DEFAULT_MODEL_VERSION)
            self.is_trained = bool(meta.get("trained", False))
            self.is_calibrated = bool(meta.get("calibrated", False))
            self.calibration_method = meta.get("calibration_method")

            logger.info(
                f"AI detector metadata loaded: version={self.model_version}, "
                f"trained={self.is_trained}, calibrated={self.is_calibrated}"
            )
        except Exception as e:
            logger.warning(f"Failed loading metadata from {self.metadata_path}: {e}")

    def _load_model(self):
        """
        Load ConvNeXt-Tiny checkpoint.
        CRITICAL: Refuses to load if metadata says trained=false or is absent.
        """
        if not self.model_path.exists():
            logger.warning(
                f"AI detector checkpoint missing at {self.model_path}. "
                f"Model will be unavailable."
            )
            self.model_loaded = False
            return

        if not self.is_trained:
            logger.warning(
                f"AI detector checkpoint exists at {self.model_path} but metadata "
                f"indicates model is NOT trained (trained={self.is_trained}). "
                f"Refusing to load untrained weights."
            )
            self.model_loaded = False
            return

        try:
            state = torch.load(
                self.model_path, map_location=DEVICE, weights_only=True
            )

            model = ConvNeXtBaseAIDetector()
            model.load_state_dict(state)
            model.to(DEVICE)
            model.eval()

            self.model = model
            self.model_loaded = True
            logger.info(
                f"ConvNeXt-Base AI detector loaded successfully from "
                f"{self.model_path} (version={self.model_version})"
            )
        except Exception as e:
            logger.error(f"Failed loading AI detector weights from {self.model_path}: {e}")
            self.model = None
            self.model_loaded = False

    def predict(self, image_bgr: np.ndarray) -> Dict[str, Any]:
        """
        Run inference on a single document image.

        Returns a dict with canonical `ai_probability` field.
        Returns ai_probability=None when model is unavailable.
        """
        if image_bgr is None or image_bgr.size == 0:
            return self._unavailable("IMAGE_MISSING")

        if self.model is None or not self.model_loaded:
            return self._unavailable("MODEL_NOT_READY")

        t0 = time.time()
        try:
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(image_rgb)
            tensor = self.transform(pil_img).unsqueeze(0).to(DEVICE)

            with torch.no_grad():
                logit = self.model(tensor)

            # Binary sigmoid → P(AI_GENERATED)
            raw_prob = float(torch.sigmoid(logit).squeeze().cpu().item())
            ai_probability = round(max(0.0, min(1.0, raw_prob)), 4)

            # Multi-scale patch inference
            patch_res = predict_patch_ai_probabilities(
                image_bgr, self.model, DEVICE
            )

            latency = round((time.time() - t0) * 1000, 2)

            return {
                "status": "READY",
                "model_version": self.model_version,
                "model_loaded": True,
                "model_name": "ConvNeXt-Base",
                "trained": self.is_trained,
                "calibrated": self.is_calibrated,
                "calibration_method": self.calibration_method,
                # Canonical AI probability — THE single authoritative field
                "ai_probability": ai_probability,
                "global_probability": ai_probability,
                # Patch statistics
                "patch_mean_probability": patch_res.get("patch_ai_mean"),
                "patch_median_probability": patch_res.get("patch_ai_median"),
                "patch_topk_probability": patch_res.get("patch_ai_topk"),
                "patch_count": patch_res.get("patch_count", 0),
                "successful_patch_count": patch_res.get("successful_patch_count", 0),
                "patch_evidence_available": patch_res.get("evidence_available", False),
                # Inference metadata
                "is_ai_generated": bool(ai_probability >= 0.5),
                "latency_ms": latency,
                "reasons": [],
            }
        except Exception as e:
            logger.error(f"AI detector inference error: {e}")
            return self._unavailable("MODEL_INFERENCE_ERROR")

    def _unavailable(self, reason: str) -> Dict[str, Any]:
        """Return safe unavailable result with ai_probability=None."""
        return {
            "status": reason,
            "model_version": self.model_version,
            "model_loaded": False,
            "model_name": "ConvNeXt-Base",
            "trained": self.is_trained,
            "calibrated": self.is_calibrated,
            "calibration_method": self.calibration_method,
            "ai_probability": None,
            "global_probability": None,
            "patch_mean_probability": None,
            "patch_median_probability": None,
            "patch_topk_probability": None,
            "patch_count": 0,
            "successful_patch_count": 0,
            "patch_evidence_available": False,
            "is_ai_generated": False,
            "latency_ms": 0.0,
            "reasons": [reason],
        }


# Public aliases
AIDetectorService = AIImageDetector


def predict_ai_image_probability(image_bgr: np.ndarray) -> Dict[str, Any]:
    """Entry point function for AI image detector."""
    detector = AIImageDetector.get_instance()
    return detector.predict(image_bgr)
