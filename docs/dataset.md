# SENTINEL AI — AI Image Dataset Guidelines

## Dataset Structure

```text
datasets/
  train/
    genuine/
    ai_generated/
  validation/
    genuine/
    ai_generated/
  test/
    genuine/
    ai_generated/
  tampered/
```

---

## Data Leakage Prevention

- **Group Splitting**: Images are split at the document/source level (e.g. all images or crops originating from a specific card photo remain strictly in the same split).
- Near-duplicates or re-compressed variants of the same physical document are NEVER permitted across different splits.

---

## Realistic Augmentations

During training, `RealisticAugmentation` applies:
- Mild rotation (-5° to +5°)
- Brightness & Contrast jitter (0.8x to 1.2x)
- Gaussian blur (radius 0.5 to 1.5)
- Simulated JPEG compression (quality 40 to 90)
