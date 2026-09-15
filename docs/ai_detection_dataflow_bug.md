# Sentinel AI — AI Detection Data-Flow Bug Root Cause Report

**Date**: 2026-09-14
**Issue Reported**: Inconsistency on Result Page:
- Model Signals Summary: `AI Deep Learning Probability = 45.8%`
- AI Image Detection & Forensics Card: `Deep Learning AI Probability = 0.0%`, `2D FFT Score = 0.0%`, `Noise Floor = 0.0%`, `Strong Signals = 0 / 5`.

---

## Complete Pipeline Data-Flow Trace

1. **Inference & Calculation (`detect_synthetic_document`)**:
   - `ai_image_detector.py` runs EfficientNet-B0 inference on the image tensor, returning `ai_probability = 0.458`.
   - `frequency_analysis.py` computes 2D FFT magnitude spectrum, returning `frequency_score = 0.32`.
   - `synthetic_detection.py` computes spatial noise floor and microtext density.
   - `forensic_fusion.py` fuses these signals and returns `synthetic_analysis` dict.

2. **Backend Orchestration (`documents.py`)**:
   - `documents.py` creates an instance of `AuditLog` ORM model.
   - It sets `audit_entry.ai_probability = 0.458`, which maps to the database column `ai_probability` in SQLite.
   - It then dynamically called `setattr(audit_entry, "synthetic_analysis", synthetic_analysis)`.

3. **Database Schema & Pydantic Serialization Gap (`schema.py` & `document.py`)**:
   - `AuditLog` in `backend/app/db/schema.py` did **NOT** contain `synthetic_analysis` as a persistent `Column(JSON)`.
   - When SQLAlchemy saved `audit_entry` to SQLite and refreshed the object, or when audit logs were fetched, dynamic `setattr` attributes were dropped.
   - When FastAPI converted `AuditLog` to `VerificationResponse` via Pydantic (`from_attributes = True`), `resultData.synthetic_analysis` was serialized as `null` / `undefined`.

4. **Frontend Consumer Mismatch (`ResultPage.jsx` & `AiImageAnalysisCard.jsx`)**:
   - `ResultPage.jsx` read `resultData.ai_probability` directly from the top-level database column, displaying `45.8%` in the sidebar summary.
   - `AiImageAnalysisCard.jsx` read `syntheticAnalysis?.ai_probability ?? 0.0`, `syntheticAnalysis?.frequency_score ?? 0.0`, `syntheticAnalysis?.noise_score ?? 0.0`, which evaluated to `null ?? 0.0` -> `0.0%` for all metrics.

---

## Root Cause Summary

The bug was caused by a **missing database JSON column (`synthetic_analysis`)** and **missing canonical serialization** in the Pydantic response schema, forcing the frontend component to fall back to `0.0` default values while the sidebar read a separate top-level database column.

---

## Resolution Strategy

1. Add `synthetic_analysis = Column(JSON, nullable=True)` to `AuditLog` in `schema.py` with auto-migration in `main.py`.
2. Add `synthetic_analysis: Optional[Dict[str, Any]]` to `VerificationResponse` in `document.py`.
3. Standardize the canonical `synthetic_analysis` payload returned by the backend:
   ```json
   synthetic_analysis = {
     "ai_probability": 0.458,
     "ai_prediction": "likely_ai_generated",
     "fft_score": 0.32,
     "noise_score": 0.15,
     "metadata_score": 0.0,
     "tampering_score": 0.12,
     "strong_signal_count": 2,
     "confidence": 0.85,
     "reasons": ["deep_learning_ai_detector_high", "frequency_spectral_anomaly"],
     "model_version": "ai_detector_v1"
   }
   ```
4. Update `AiImageAnalysisCard.jsx` to read exclusively from `resultData.synthetic_analysis`.
