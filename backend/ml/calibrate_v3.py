#!/usr/bin/env python3
"""
SENTINEL V3 — Model Sanity Check + Domain Threshold Calibration
================================================================

Run this BEFORE touching any other code. It answers the question your
dashboard could not: is the ConvNeXt detector actually broken, or just
mis-thresholded?

    python ml/calibrate_v3.py --real dataset/id_domain/real \
                              --synth dataset/id_domain/ai_generated \
                              --model app/models/convnext_base_ai_detector_v1.pth \
                              --out app/models/thresholds_v3.json

What it reports
---------------
1. MODEL SANITY. Feeds every image through the detector and reports the output
   distribution. The diagnostic that matters is output STANDARD DEVIATION:

       std < 0.05  -> DEAD. The head has collapsed to a constant. Retrain.
                      (Your production numbers -- global 0.557, patch top-k
                       0.556 on two visually different images -- are the
                       signature of exactly this.)
       AUC  < 0.60 -> The model learned nothing transferable to ID cards.
                      Retrain on domain data; do not calibrate a coin flip.
       AUC  > 0.85 -> Healthy. Calibration will fix the thresholds.

2. TEXTURE THRESHOLDS. Fits the per-class medians that
   modules/texture_forensics.py interpolates between, on YOUR data rather
   than on borrowed constants.

3. PROVENANCE AUC. Usually the highest single-feature AUC in this domain, and
   it needs no training at all.

Dataset requirement -- this is the part people skip and it is why the model
died. Both folders must be captured the SAME WAY:

    dataset/id_domain/real/          photographs of physical ID cards,
                                     taken on phones, with glare, tilt,
                                     lamination, at 3-8 MP
    dataset/id_domain/ai_generated/  synthetic ID cards, then RE-CAPTURED
                                     or degraded with the recapture
                                     augmentation below

If the real folder is phone photos and the synth folder is clean 1024x1024
PNGs, the model learns "JPEG artefacts vs no JPEG artefacts" and collapses the
moment it meets a screenshot. Aim for >= 800 images per class to start.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import cv2
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from app.modules.provenance import analyse_provenance                 # noqa: E402
from app.modules.texture_forensics import analyse_texture             # noqa: E402

IMG_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff", ".heic"}


# --------------------------------------------------------------------------
# Recapture augmentation. Apply to synthetic training images so the model is
# forced to learn content/texture cues instead of compression cues.
# --------------------------------------------------------------------------
def recapture_augment(bgr: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    h, w = bgr.shape[:2]

    # 1. perspective tilt, as if held at an angle
    m = rng.uniform(0.01, 0.045)
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    dst = np.float32([[rng.uniform(0, m) * w, rng.uniform(0, m) * h],
                      [w - rng.uniform(0, m) * w, rng.uniform(0, m) * h],
                      [w - rng.uniform(0, m) * w, h - rng.uniform(0, m) * h],
                      [rng.uniform(0, m) * w, h - rng.uniform(0, m) * h]])
    bgr = cv2.warpPerspective(bgr, cv2.getPerspectiveTransform(src, dst), (w, h),
                              borderMode=cv2.BORDER_REPLICATE)

    # 2. optical blur
    bgr = cv2.GaussianBlur(bgr, (0, 0), rng.uniform(0.4, 1.6))

    # 3. laminate glare: a soft bright ellipse
    if rng.random() < 0.6:
        glare = np.zeros((h, w), np.float32)
        cx, cy = rng.integers(0, w), rng.integers(0, h)
        cv2.ellipse(glare, (int(cx), int(cy)),
                    (int(w * rng.uniform(0.15, 0.4)), int(h * rng.uniform(0.1, 0.3))),
                    rng.uniform(0, 180), 0, 360, 1.0, -1)
        glare = cv2.GaussianBlur(glare, (0, 0), min(h, w) * 0.05)
        bgr = np.clip(bgr.astype(np.float32) +
                      glare[..., None] * rng.uniform(20, 70), 0, 255).astype(np.uint8)

    # 4. sensor noise
    bgr = np.clip(bgr.astype(np.float32) +
                  rng.normal(0, rng.uniform(1.5, 6.0), bgr.shape), 0, 255).astype(np.uint8)

    # 5. JPEG re-compression at phone-like quality
    q = int(rng.integers(62, 93))
    ok, enc = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), q])
    if ok:
        bgr = cv2.imdecode(enc, cv2.IMREAD_COLOR)
    return bgr


def _auc(pos: np.ndarray, neg: np.ndarray) -> float:
    """Mann-Whitney U / rank AUC. pos = synthetic, neg = real."""
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    allv = np.concatenate([pos, neg])
    order = allv.argsort()
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(allv) + 1)
    rp = ranks[:len(pos)].sum()
    return float((rp - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def _load_detector(model_path: str | None):
    if not model_path or not pathlib.Path(model_path).exists():
        return None
    try:
        import torch
        import torchvision
        net = torchvision.models.convnext_base()
        net.classifier[2] = torch.nn.Linear(1024, 2)
        state = torch.load(model_path, map_location="cpu")
        net.load_state_dict(state.get("state_dict", state), strict=False)
        net.eval()
        return net
    except Exception as exc:
        print(f"  [!] could not load detector: {exc}")
        return None


def _detector_prob(net, bgr: np.ndarray) -> float:
    import torch
    rgb = cv2.cvtColor(cv2.resize(bgr, (224, 224)), cv2.COLOR_BGR2RGB)
    x = torch.from_numpy(rgb).permute(2, 0, 1).float().div(255.)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    x = ((x - mean) / std).unsqueeze(0)
    with torch.no_grad():
        return float(torch.softmax(net(x), 1)[0, 1])


def scan(folder: pathlib.Path, net):
    rows = []
    for p in sorted(folder.rglob("*")):
        if p.suffix.lower() not in IMG_EXT:
            continue
        raw = p.read_bytes()
        bgr = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
        if bgr is None:
            continue
        prov = analyse_provenance(raw)
        tex = analyse_texture(bgr)
        rows.append({
            "path": str(p),
            "prov": prov.synthetic_provenance_score,
            "local_var": tex.local_variance_median,
            "hf_ratio": tex.hf_lf_energy_ratio,
            "residual": tex.highpass_residual_std,
            "lap": tex.laplacian_variance,
            "model": _detector_prob(net, bgr) if net is not None else float("nan"),
        })
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--real", required=True)
    ap.add_argument("--synth", required=True)
    ap.add_argument("--model", default=None)
    ap.add_argument("--out", default="app/models/thresholds_v3.json")
    a = ap.parse_args()

    net = _load_detector(a.model)
    print("Scanning real   ...")
    real = scan(pathlib.Path(a.real), net)
    print("Scanning synth  ...")
    synth = scan(pathlib.Path(a.synth), net)
    print(f"  real={len(real)}  synth={len(synth)}\n")
    if not real or not synth:
        print("ERROR: both folders need images.")
        return 1

    def col(rows, k):
        return np.array([r[k] for r in rows], float)

    print("=" * 66)
    print("1. MODEL SANITY")
    print("=" * 66)
    verdict_retrain = False
    if net is None:
        print("  detector not loaded -- skipping")
    else:
        mr, ms = col(real, "model"), col(synth, "model")
        allm = np.concatenate([mr, ms])
        std = float(np.nanstd(allm))
        auc = _auc(ms, mr)
        print(f"  output mean {np.nanmean(allm):.4f}   std {std:.4f}")
        print(f"  real  mean  {np.nanmean(mr):.4f}    synth mean {np.nanmean(ms):.4f}")
        print(f"  AUC         {auc:.4f}")
        if std < 0.05:
            print("  >> VERDICT: DEAD HEAD. Output is near-constant.")
            print("     Causes, in order of likelihood:")
            print("       a) classifier head never unfrozen during training")
            print("       b) learning rate collapsed the head to the class prior")
            print("       c) checkpoint saved before the head converged")
            print("       d) load_state_dict(strict=False) silently dropped the head")
            print("     ACTION: retrain. Verify train accuracy exceeds 0.95 on a")
            print("             200-image subset BEFORE full training (overfit test).")
            verdict_retrain = True
        elif auc < 0.60:
            print("  >> VERDICT: NO TRANSFER. Model learned out-of-domain features.")
            print("     ACTION: rebuild the dataset with recapture_augment() and retrain.")
            verdict_retrain = True
        elif auc < 0.85:
            print("  >> VERDICT: WEAK. Usable only as a corroborating signal.")
        else:
            print("  >> VERDICT: HEALTHY. Proceed to calibration.")

    print()
    print("=" * 66)
    print("2. HANDCRAFTED FEATURE AUC  (no training required)")
    print("=" * 66)
    for key, label in [("prov", "provenance score"),
                       ("local_var", "local variance (inverted)"),
                       ("hf_ratio", "HF/LF ratio (inverted)"),
                       ("residual", "highpass residual")]:
        vr, vs = col(real, key), col(synth, key)
        auc = _auc(vs, vr)
        if key in ("local_var", "hf_ratio"):
            auc = 1.0 - auc
        print(f"  {label:<30} AUC {auc:.4f}   "
              f"real med {np.median(vr):8.3f}   synth med {np.median(vs):8.3f}")

    thresholds = {
        "local_var_real_median":   float(np.median(col(real, "local_var"))),
        "local_var_synth_median":  float(np.median(col(synth, "local_var"))),
        "hf_ratio_real_median":    float(np.median(col(real, "hf_ratio"))),
        "hf_ratio_synth_median":   float(np.median(col(synth, "hf_ratio"))),
        "residual_real_median":    float(np.median(col(real, "residual"))),
        "residual_synth_median":   float(np.median(col(synth, "residual"))),
        "min_laplacian_for_texture": float(np.percentile(col(real, "lap"), 10)),
        "max_jpeg_blockiness": 0.35,
        "_meta": {
            "n_real": len(real),
            "n_synth": len(synth),
            "model_requires_retrain": verdict_retrain,
        },
    }
    out = pathlib.Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(thresholds, indent=2))
    print(f"\nWrote {out}")

    if verdict_retrain:
        print("\n*** Set is_calibrated=False until the model is retrained. ***")
        print("*** forensic_fusion_v3 will exclude it and the pipeline will   ***")
        print("*** still reach verdicts via PROVENANCE + SEMANTIC + TEXTURE.  ***")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
