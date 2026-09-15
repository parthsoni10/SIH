# SENTINEL AI — Target System Architecture

## Overview

SENTINEL AI is a dual-tier identity document verification platform combining local forensic computer vision algorithms, deep learning neural networks, and multimodal generative AI.

```
IMAGE
 │
 ├── DOCUMENT FORENSICS (ELA, EXIF, Noise) ──► Tampering Score
 │
 └── AI IMAGE DETECTOR (EfficientNet-B0) ──► AI Probability
       │
       ▼
  FORENSIC FUSION ENGINE (FFT Frequency, Noise Floor, Metadata, Microtext)
       │
       ▼
  2-SIGNAL CORROBORATION RULE
       │
       ▼
  FEATURE ENGINEERING (16-feature vector)
       │
       ▼
  RANDOM FOREST V2 CLASSIFIER
       │
       ▼
  DECISION ENGINE ──► GENUINE / REVIEW / FRAUDULENT
       │
       ▼
  MULTIMODAL GEMINI EXPLANATION
       │
       ▼
  DB AUDIT LOG & REACT DASHBOARD
```

---

## Separation of Edit-Tampering vs AI-Generation

- **Document Tampering (`tampering.py`)**: Asks *"was a genuine physical document edited or altered after camera capture?"* (analyzes Error Level Analysis diffs, EXIF software tags, block noise variance).
- **AI-Generation Detection (`ai_image_detector.py` & `synthetic_detection.py`)**: Asks *"is this image a synthetic AI rendering that was never a real physical object?"* (analyzes EfficientNet-B0 probability, 2D FFT magnitude spectrum, noise floor smoothness, microtext density).

---

## Decision Tiers

- **GENUINE**: Clear document with low risk score (<30%) and clean signal integrity.
- **REVIEW**: Single uncorroborated suspicious signal, weak OCR, minor blur, or medium risk (30% - 79%) requiring officer review.
- **FRAUDULENT**: Corroborated AI generation (≥2 strong signals), confirmed edit tampering, blacklisted ID, or high risk (≥80%).
