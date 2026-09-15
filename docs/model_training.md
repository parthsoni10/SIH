# SENTINEL AI — Model Training Instructions

## 1. Train EfficientNet-B0 AI Image Detector

```bash
python training/train_ai_detector.py
```
- Outputs weights to `backend/app/models/ai_detector_v1.pth`.
- Saves training metadata to `backend/app/models/ai_detector_v1_metadata.json`.

---

## 2. Train Random Forest V2 Risk Model

```bash
python training/train_risk_model_v2.py
```
- Ingests 16-feature vector including `ai_probability`, `frequency_score`, `synthetic_noise_score`, and `forensic_consistency_score`.
- Saves model to `backend/app/models/risk_model_v2.pkl`.
- Preserves `risk_model.pkl` as baseline.
