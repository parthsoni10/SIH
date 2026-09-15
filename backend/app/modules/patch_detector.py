"""
SENTINEL AI — Multi-Scale Patch AI Detection (ConvNeXt-Tiny)

Extracts document patches and runs binary ConvNeXt-Tiny inference per patch.
Aggregation: top-k median of patch AI probabilities.

Returns ai_probability=None when model is unavailable.
"""

import cv2
import torch
import numpy as np
import logging

from typing import Dict, Any, List, Optional
from torchvision import transforms

logger = logging.getLogger(__name__)


PATCH_TRANSFORM = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


def extract_document_patches(
    image_bgr: np.ndarray,
    patch_size: int = 224,
) -> List[np.ndarray]:
    """
    Extract a 3x3 grid of patches from the document image.
    Returns up to 9 patches covering corners, edges, and center.
    """
    if image_bgr is None or image_bgr.size == 0:
        return []

    if len(image_bgr.shape) != 3:
        return []

    h, w = image_bgr.shape[:2]

    if h < patch_size or w < patch_size:
        resized = cv2.resize(
            image_bgr,
            (patch_size, patch_size),
            interpolation=cv2.INTER_AREA,
        )
        return [resized]

    y_coords = [
        0,
        max(0, (h - patch_size) // 2),
        max(0, h - patch_size),
    ]

    x_coords = [
        0,
        max(0, (w - patch_size) // 2),
        max(0, w - patch_size),
    ]

    y_coords = list(dict.fromkeys(y_coords))
    x_coords = list(dict.fromkeys(x_coords))

    patches = []

    for y in y_coords:
        for x in x_coords:
            patch = image_bgr[
                y:y + patch_size,
                x:x + patch_size,
            ]

            if (
                patch.shape[0] == patch_size
                and patch.shape[1] == patch_size
            ):
                patches.append(patch.copy())

    return patches


def _extract_ai_probability(output: torch.Tensor) -> float:
    """
    Extract AI probability from binary classifier output.
    Expects shape [1, 1] (single logit) → sigmoid → P(AI_GENERATED).
    """
    if output is None:
        raise ValueError("Model returned None")

    if output.ndim != 2:
        raise ValueError(f"Invalid output shape: {tuple(output.shape)}")

    if output.shape[0] != 1:
        raise ValueError("Patch inference expects batch size 1")

    if output.shape[1] == 1:
        # Binary classifier: single logit → sigmoid
        ai_prob = float(torch.sigmoid(output[0, 0]).item())
        return max(0.0, min(1.0, ai_prob))

    if output.shape[1] == 2:
        # 2-class softmax fallback
        probs = torch.softmax(output, dim=1)[0]
        return float(probs[1].item())

    if output.shape[1] == 3:
        # Legacy 3-class fallback (ai_generated is index 1)
        probs = torch.softmax(output, dim=1)[0]
        return float(probs[1].item())

    raise ValueError(
        f"Unexpected output dimension: {output.shape[1]}. "
        f"Expected 1 (binary), 2, or 3 classes."
    )


def predict_patch_ai_probabilities(
    image_bgr: np.ndarray,
    model: Optional[torch.nn.Module] = None,
    device: Optional[torch.device] = None,
    top_k: int = 3,
) -> Dict[str, Any]:
    """
    Run ConvNeXt-Tiny inference on document patches.

    Returns aggregated patch statistics.
    All probability fields are None when evidence is unavailable.
    """
    patches = extract_document_patches(image_bgr)

    result = {
        "patch_count": len(patches),
        "successful_patch_count": 0,
        "patch_ai_mean": None,
        "patch_ai_median": None,
        "patch_ai_topk": None,
        "max_patch_probability": None,
        "patch_probabilities": [],
        "evidence_available": False,
        "inference_error": None,
    }

    if not patches:
        result["inference_error"] = "NO_VALID_PATCHES"
        return result

    if model is None:
        result["inference_error"] = "AI_MODEL_UNAVAILABLE"
        logger.error("AI patch model unavailable")
        return result

    try:
        model.eval()

        if device is None:
            try:
                device = next(model.parameters()).device
            except StopIteration:
                device = torch.device("cpu")

        ai_probs = []

        with torch.inference_mode():
            for patch in patches:
                try:
                    rgb_patch = cv2.cvtColor(
                        patch, cv2.COLOR_BGR2RGB
                    )
                    tensor = PATCH_TRANSFORM(
                        rgb_patch
                    ).unsqueeze(0).to(device)

                    output = model(tensor)
                    ai_prob = _extract_ai_probability(output)
                    ai_probs.append(ai_prob)

                except Exception:
                    logger.exception("Patch inference failed")

        if not ai_probs:
            result["inference_error"] = "ALL_PATCH_INFERENCES_FAILED"
            return result

        sorted_ai = sorted(ai_probs, reverse=True)
        k = min(max(1, top_k), len(sorted_ai))
        top_k_mean = float(np.mean(sorted_ai[:k]))

        result.update({
            "successful_patch_count": len(ai_probs),
            "patch_ai_mean": round(float(np.mean(ai_probs)), 4),
            "patch_ai_median": round(float(np.median(ai_probs)), 4),
            "patch_ai_topk": round(top_k_mean, 4),
            "max_patch_probability": round(float(max(ai_probs)), 4),
            "patch_probabilities": [
                round(float(x), 4) for x in ai_probs
            ],
            "evidence_available": True,
            "inference_error": None,
        })

        return result

    except Exception as exc:
        logger.exception("Patch detector failed")
        result["inference_error"] = f"PATCH_INFERENCE_FAILED: {exc}"
        return result
