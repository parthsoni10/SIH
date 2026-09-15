# SENTINEL AI — Target Architecture V2 Specification

## 1. Core Philosophy & Separation of Concerns

The SENTINEL AI Target Architecture V2 enforces strict decoupling across 5 independent forensic dimensions:

1. **DOCUMENT_VALIDITY**: Structural format, regex patterns, checksum algorithms, ICAO MRZ check digits, expiration, and layout template matching.
2. **IMAGE_AUTHENTICITY / AI_GENERATION**: Deep learning ConvNeXt-Base binary classifier (~88M parameters, ~350MB model size, global + multi-scale patch inference), 2D FFT normalized spectral frequency anomaly score, spatial noise anomaly score, camera EXIF tag inspection, and 2-signal corroboration.
3. **DIGITAL_TAMPERING**: Error Level Analysis (ELA), grid noise variance inconsistency, copy-move feature duplication (ORB keypoint matching), splicing edge boundary discontinuities, and software editing signatures.
4. **IMAGE_QUALITY**: Gating factor evaluating blur (Laplacian variance), brightness, contrast, resolution, JPEG compression quality, and noise level.
5. **FINAL_DECISION**: Multi-stage decision matrix mapping forensic signals into definitive verdicts: `GENUINE`, `AI_GENERATED`, `ALTERED`, `AI_GENERATED_AND_ALTERED`, `SUSPICIOUS`, and `MANUAL_REVIEW`.

---

## 2. 20-Stage Screening Pipeline

```
DOCUMENT IMAGE ──▶ STAGE 1: Upload
                      │
                      ▼
               STAGE 2: Image Quality Assessment & Gating
                      │
                      ▼
               STAGE 3: Image Normalization
                      │
                      ▼
               STAGE 4: Document Type Detection / Validation
                      │
                      ▼
               STAGE 5: OCR Text Extraction
                      │
                      ▼
               STAGE 6: Document Field Parsing
                      │
                      ▼
               STAGE 7: Document Number Validation
                      │
                      ▼
               STAGE 8: Layout & Geometry Validation
                      │
                      ▼
               STAGE 9: Document Structural Validity
                      │
          ┌───────────┼───────────┐
          ▼           ▼           ▼
     STAGE 10     STAGE 11-14  STAGE 15
    Digital Edit   AI Detector   Face Verification
    Tampering      Global + Patch (Null if omitted)
    Forensics      FFT + Noise
          │           │           │
          └───────────┼───────────┘
                      ▼
               STAGE 16: Multi-Signal Forensic Evidence Fusion
                      │
                      ▼
               STAGE 17: Calibrated Decision Engine
                      │
                      ▼
               STAGE 18: Multimodal Gemini Explanation (Advisory)
                      │
                      ▼
               STAGE 19: Audit Database Save (SQLite)
                      │
                      ▼
               STAGE 20: Canonical API Response (React 18 SPA)
```

---

## 3. Calibrated Fail-Closed Final Decisions

| Status | Condition |
|---|---|
| **`GENUINE`** | Valid document structure, AI probability < 0.35, 0 strong signals, tampering probability < 0.50, acceptable quality, trained & calibrated model. |
| **`AI_GENERATED`** | AI probability $\ge 0.65$ AND corroborated by $\ge 2$ strong signals. |
| **`ALTERED`** | Digital tampering probability $\ge 0.75$ (high ELA, copy-move, or splicing edge anomaly). |
| **`AI_GENERATED_AND_ALTERED`** | Both AI probability $\ge 0.65$ AND tampering probability $\ge 0.75$. |
| **`SUSPICIOUS`** | Invalid document format, checksum failure, or blacklisted ID match without strong AI/tampering corroboration. |
| **`MANUAL_REVIEW`** | Inconclusive AI evidence (0.35–0.65), uncalibrated/unloaded AI model, quality score < 0.60 (`QUALITY_TOO_LOW_FOR_FORENSICS`), or missing evidence. |

---

## 4. Safety Guardrails

1. **Blur, JPEG Compression & Missing EXIF**: Never automatically label an image as `AI_GENERATED`. Degraded images output `quality_too_low_for_forensics = True` and route to `MANUAL_REVIEW`.
2. **Missing Live Capture Photo**: `face_match_score` is set to `null` (`available: false`, `status: "NOT_PROVIDED"`). It is **never** assigned an artificial default score of 0.93.
3. **Random Forest Classifier**: Completely removed from active final classification usage.
4. **Advisory Gemini Role**: Gemini provides officer report explanations and field verification; it cannot override deterministic forensic logic.