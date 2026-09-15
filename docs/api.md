# SENTINEL AI — API Documentation

## `POST /api/documents/verify`

Uploads identity document image(s) for verification.

### Parameters (Form-Data)
- `document_type` (string, required): `"Passport"`, `"Visa"`, `"Aadhaar"`, `"PAN Card"`, `"Driving License"`, or `"Other"`
- `document_file` (file, required): Front side image file (JPG, PNG, WebP)
- `document_back_file` (file, optional): Back side image file
- `live_file` (file, optional): Live camera selfie capture

---

### Example Response Payload (Phase 12)

```json
{
  "id": 142,
  "created_at": "2026-09-14T18:00:00",
  "document_type": "PAN Card",
  "filename": "pan_sample.jpg",
  "risk_score": 92,
  "prediction": "fraudulent",
  "probability": 0.92,
  "explanation": "Document flagged due to corroborated synthetic AI generation cues.",
  "ocr_confidence": 0.95,
  "validation_pass_rate": 1.0,
  "id_checksum_valid": true,
  "expiry_valid": true,
  "tampering_score": 0.15,
  "face_match_score": null,
  "blacklist_hit": false,
  "risk": {
    "score": 92,
    "probability": 0.92,
    "tier": "high"
  },
  "synthetic_analysis": {
    "score": 0.94,
    "prediction": "likely_ai_generated",
    "ai_probability": 0.94,
    "frequency_score": 0.88,
    "noise_score": 0.81,
    "metadata_score": 0.20,
    "strong_signal_count": 3
  },
  "tampering_analysis": {
    "score": 0.15,
    "signals": { "ela_score": 0.08, "metadata_score": 0.20, "noise_inconsistency": 0.0 }
  },
  "decision": {
    "status": "fraudulent",
    "requires_manual_review": false,
    "reason_codes": ["SYNTHETIC_AI_IMAGE_CORROBORATED", "HIGH_FRAUD_RISK_SCORE"]
  },
  "model_versions": {
    "risk_model": "risk_model_v2",
    "ai_detector": "ai_detector_v1"
  }
}
```
