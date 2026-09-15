# SENTINEL AI — Baseline System Report (Phase 0)

**Date**: 2026-09-14
**System**: SENTINEL AI Document Verification Platform
**Status**: Baseline Inspection Complete

---

## 1. System Architecture

SENTINEL AI is a hybrid document verification system that combines local computer-vision algorithms, machine-learning risk models, and multimodal LLM analysis (Gemini) for end-to-end identity document screening.

### Key Operational Characteristics
- **Backend Framework**: FastAPI (running on Uvicorn)
- **Frontend Framework**: React + Vite + TailwindCSS
- **Database**: SQLite (`screening.db`) via SQLAlchemy ORM
- **Local Machine Learning**: Scikit-Learn Random Forest Classifier (`risk_model.pkl`), YuNet OpenCV Face Detector (`face_detection_yunet_2023mar.onnx`)
- **Multimodal AI**: Google Gemini (`google-genai`) for unified field extraction, visual verification, and officer explanations

---

## 2. Dependencies & Environment

### Backend Dependencies (`backend/requirements.txt`)
- `fastapi`, `uvicorn[standard]`, `python-multipart`, `sqlalchemy`
- `paddleocr`, `paddlepaddle`, `opencv-python`, `Pillow`, `scikit-image`
- `scikit-learn==1.6.1`, `joblib`, `pandas`, `numpy`
- `deepface`, `tensorflow`, `mistralai`, `python-dateutil`, `python-dotenv`, `pytest`

### Frontend Dependencies (`Frontend/package.json`)
- React 18, Vite, TailwindCSS, Lucide React icons, Axios

---

## 3. Pipeline Flow & Verification Orchestration

Document verification is orchestrated in `backend/app/routers/documents.py` via `POST /api/documents/verify`:

```
User Upload (Front Image + Optional Back Image + Optional Live Capture)
  │
  ├── 1. Preprocessing (Resize, EXIF Extraction, Quality Checks)
  ├── 2. OCR & Document Layout Normalization (PaddleOCR / MRZ Parser)
  ├── 3. Synthetic Signals Pass 1 (Garbled microtext, QR bimodality, EXIF AI tools, Noise floor)
  ├── 4. Validation Rules (Regex, Expiry, Checksums, Blacklist) & Tampering Detection (ELA, EXIF, Noise)
  ├── 5. Face Verification (YuNet / OpenCV face match against live capture)
  ├── 6. Initial Risk Scoring (Random Forest v1 on 12-feature vector)
  ├── 7. Unified Multimodal LLM Analysis (Gemini field extraction & visual score)
  ├── 8. Synthetic Detection Pass 2 (Recalculates synthetic score with Gemini visual score)
  ├── 9. Final Risk & Hard Override Re-evaluation
  └── 10. Audit Logging (SQLite) & JSON API Response
```

---

## 4. Current ML & Heuristic Models

### Risk Model (`risk_model_v1`)
- **Location**: `backend/app/models/risk_model.pkl`
- **Type**: Scikit-Learn `RandomForestClassifier` (12 features)
- **Features**: `ocr_confidence`, `validation_pass_rate`, `id_checksum_valid`, `expiry_valid`, `tampering_score`, `face_match_score`, `blacklist_hit`, `document_type_Aadhaar`, `document_type_Driving License`, `document_type_PAN Card`, `document_type_Passport`, `document_type_Visa`
- **Role**: Calculates general fraud probability (0–100%).

### Face Detection Model
- **Location**: `backend/app/models/face_detection_yunet_2023mar.onnx`
- **Type**: OpenCV YuNet ONNX detector for face detection and alignment.

---

## 5. Current Database Schema (`audit_logs`)

Defined in `backend/app/db/schema.py`:
- `id`, `created_at`, `document_type`, `filename`
- `risk_score`, `prediction`, `probability`, `explanation`
- `ocr_confidence`, `validation_pass_rate`, `id_checksum_valid`, `expiry_valid`, `tampering_score`, `face_match_score`, `face_match_missing`, `blacklist_hit`
- JSON fields: `extracted_fields`, `failed_rules`, `tampering_signals`, `feature_vector`, `raw_text`, `llm_validation`, `preprocessing_warnings`, `tampering_flags`, `synthetic_reasons`, `layout_anomalies`
- Synthetic & Layout fields: `synthetic_generation_score`, `layout_score`

---

## 6. Current Limitations & Upgrade Objectives

1. **No Dedicated Deep-Learning AI Image Detector**: The current system relies on heuristic signals (QR bimodality, microtext fuzzy match, EXIF tool tags) and LLM visual judgment. It lacks a trained CNN/ViT baseline (e.g. EfficientNet-B0) dedicated to detecting synthetic/AI-generated document images.
2. **Feature Vector Gaps**: `risk_model_v1` does not ingest deep-learning AI image probability, frequency spectrum features, or dedicated synthetic scores.
3. **Decision Engine & Corroboration**: Needs a standalone `decision_engine.py` producing calibrated `GENUINE`, `REVIEW`, or `FRAUDULENT` decisions based on a explicit 2-signal corroboration rule.
4. **Dataset & Calibration**: Lacks a structured dataset pipeline (`datasets/train`, `datasets/val`, `datasets/test`) and formal threshold calibration artifacts (`models/thresholds_v1.json`).

---

## 7. Baseline Test Verification
- Ran full test suite (`python -m pytest tests/ -v`).
- **Pass Rate**: 54/54 passed (100%).
