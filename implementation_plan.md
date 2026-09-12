# AI-Based Fake Identity & Document Screening System — Implementation Plan
### SIH 2026 | Stack: React + Tailwind | FastAPI | PaddleOCR | DeepFace | Gemini AI | Pre-trained RandomForest

This document is written as a set of **self-contained module specs**. Each one can be handed
individually to a coding agent — it states the goal, exact inputs/outputs, file layout, dependencies,
step-by-step tasks, and acceptance criteria, so a module can be built and tested in isolation before
wiring it into the pipeline.

---

## 0. Architecture Overview

Single backend now (no separate Node layer) — **FastAPI does everything**: file handling, calling each
AI module, orchestration, and serving the React frontend's API calls.

```
                         ┌─────────────────────────┐
                         │   React + Tailwind SPA   │
                         │  (Upload / Result / Log) │
                         └────────────┬────────────┘
                                      │ REST (JSON)
                         ┌────────────▼────────────┐
                         │        FastAPI           │
                         │  (single backend, all    │
                         │   modules as services)    │
                         └───┬─────┬─────┬─────┬────┘
                             │     │     │     │
                   ┌─────────┘     │     │     └─────────┐
                   ▼               ▼     ▼               ▼
            M2: OCR          M3: Validation M4: Tampering  M5: Face Match
           (PaddleOCR)          (rules)      (OpenCV/ELA)    (DeepFace)
                   │               │             │               │
                   └───────┬───────┴──────┬──────┴───────┬───────┘
                           ▼               ▼              ▼
                   M6: Risk Scoring (risk_model.pkl, 12 features)
                           │
                           ▼
                   M7: Gemini Explanation Layer
                           │
                           ▼
                   M8: Response + Audit Log (DB)
```

**Processing flow per document:** upload → preprocess → OCR (parallel with tampering check) →
validation (uses OCR output) → face match (if live capture provided) → assemble 12-feature vector →
risk model prediction → Gemini turns the raw scores into a one-line officer explanation → response +
audit log write.

### Repo layout

```
ai-border-screening/
├── backend/
│   ├── app/
│   │   ├── main.py                    # FastAPI app, routers mounted here
│   │   ├── config.py                  # env vars, model paths, Gemini key
│   │   ├── routers/
│   │   │   ├── documents.py           # upload + full pipeline endpoint
│   │   │   └── audit.py               # history/audit endpoints
│   │   ├── modules/
│   │   │   ├── preprocessing.py       # Module 1
│   │   │   ├── ocr.py                 # Module 2 (PaddleOCR)
│   │   │   ├── mrz.py                 # Module 2b (MRZ parser)
│   │   │   ├── validation.py          # Module 3
│   │   │   ├── tampering.py           # Module 4
│   │   │   ├── face_match.py          # Module 5 (DeepFace)
│   │   │   ├── risk_scoring.py        # Module 6 (loads risk_model.pkl)
│   │   │   └── explain.py             # Module 7 (Gemini)
│   │   ├── models/                    # risk_model.pkl lives here
│   │   ├── db/
│   │   │   ├── database.py            # SQLite/Postgres session
│   │   │   └── schema.py              # SQLAlchemy models
│   │   └── schemas/                   # Pydantic request/response models
│   ├── requirements.txt
│   └── tests/
│       ├── test_ocr.py
│       ├── test_tampering.py
│       ├── test_face_match.py
│       └── test_risk_scoring.py
│
├── frontend/
│   ├── src/
│   │   ├── pages/                     # Upload, Result, AuditTrail
│   │   ├── components/                # RiskBadge, Uploader, FieldTable, FaceCapture
│   │   ├── services/api.js
│   │   └── App.jsx
│   ├── tailwind.config.js
│   └── package.json
│
└── docs/
    └── implementation_plan.md         # this file
```

---

## Module 1 — Preprocessing

**Goal:** normalize every uploaded image before any downstream module touches it, so OCR, tampering,
and face modules all see consistent input.

**Location:** `backend/app/modules/preprocessing.py`

**Input:** raw uploaded file (JPEG/PNG/PDF page) as bytes
**Output:** `PreprocessedImage` — normalized `np.ndarray` (BGR), plus metadata dict (`original_size`,
`format`, `exif_present: bool`)

**Dependencies:** `opencv-python`, `Pillow`, `pdf2image` (if PDFs are allowed)

**Tasks:**
1. Accept bytes → decode via `cv2.imdecode`; if PDF, rasterize first page via `pdf2image`.
2. Auto-orient using EXIF orientation tag (`Pillow.ImageOps.exif_transpose`) **before** stripping EXIF
   elsewhere — tampering module needs the original EXIF separately, so preprocessing should return the
   EXIF dict rather than discard it.
3. Resize longest edge to a fixed max (e.g. 2000px) to bound downstream compute time; never upscale.
4. Run a basic blur/quality check (Laplacian variance) and flag `low_quality: bool` if below threshold —
   surfaced later to the officer rather than silently failing OCR.
5. Return the normalized image array + metadata dict; do **not** mutate/save the original file yet
   (original gets saved separately, untouched, for the audit trail).

**Acceptance criteria:** given a rotated, EXIF-tagged phone photo, output is correctly oriented,
resized, and returns `exif_present: True` with the raw EXIF dict intact for Module 4.

---

## Module 2 — OCR Extraction (PaddleOCR)

**Goal:** extract every text field from the document plus a confidence score per field.

**Location:** `backend/app/modules/ocr.py` (+ `mrz.py` for passport MRZ)

**Input:** preprocessed image array, `document_type` (string, one of `Passport|Visa|Aadhaar|PAN
Card|Driving License`)
**Output:**
```json
{
  "fields": {"name": {"value": "JOHN DOE", "confidence": 0.97}, "dob": {"value": "1990-05-12", "confidence": 0.95}, "...": "..."},
  "ocr_confidence": 0.96,
  "mrz": {"present": true, "raw_lines": ["P<IND..."], "checksum_valid": true}
}
```

**Dependencies:** `paddleocr`, `paddlepaddle` (CPU build unless GPU is available)

**Tasks:**
1. Initialize `PaddleOCR(use_angle_cls=True, lang='en')` once at app startup (not per-request — it's
   expensive to load), store on app state.
2. Run OCR to get raw `(bbox, text, confidence)` triples.
3. Build a **per-document-type field extractor**: since layouts are known (Aadhaar, PAN, Passport,
   Visa, DL each have fixed field positions/labels), map raw OCR boxes to named fields using either
   (a) relative bbox position heuristics, or (b) regex/keyword matching on labels near each value
   (e.g. text following "DOB" or "Date of Birth" → dob field). Start with (b), it's more robust to
   scan variance.
4. `ocr_confidence` = mean confidence across all extracted fields (this becomes a risk-model feature —
   keep the exact name `ocr_confidence`).
5. For Passport/Visa only: if an MRZ-shaped region (two or three lines of `<` and alphanumerics at the
   bottom) is detected, run it through `mrz.py`:
   - Parse per ICAO 9303 field positions (document number, DOB, expiry, nationality).
   - Recompute each field's embedded check digit and compare to the MRZ's own checksum digit.
   - Return `id_checksum_valid: bool` — this is your cheapest, highest-signal fraud detector; build
     and test this in isolation first.
6. For Aadhaar/PAN: no MRZ, but Aadhaar has a Verhoeff checksum on the 12-digit number and PAN has a
   fixed alphanumeric pattern (`[A-Z]{5}[0-9]{4}[A-Z]{1}`) — implement both as the `id_checksum_valid`
   equivalent for those types.

**Acceptance criteria:** unit test with 3 sample images (one per major doc type) returns correctly
mapped fields with `confidence > 0`, and a deliberately mismatched-checksum MRZ returns
`checksum_valid: False`.

---

## Module 3 — Document Validation

**Goal:** turn extracted fields into pass/fail rule checks — this feeds `validation_pass_rate` and
`expiry_valid` directly into the risk model.

**Location:** `backend/app/modules/validation.py`

**Input:** OCR output from Module 2
**Output:**
```json
{"validation_pass_rate": 0.83, "expiry_valid": true, "failed_rules": ["dob_format"], "blacklist_hit": false}
```

**Dependencies:** none beyond stdlib + `python-dateutil`

**Tasks:**
1. Define a rules registry per document type: required-field presence, date format validity, date
   logical checks (`dob < today`, `expiry > today` → feeds `expiry_valid` directly), field-length/
   pattern checks (PAN regex, Aadhaar 12-digit, passport number pattern).
2. `validation_pass_rate` = (rules passed) / (total rules run) for that document type.
3. Blacklist check: query a `blacklist` table (document number, name+DOB hash) in the DB — this is a
   **mock table** for the demo, seed it with a handful of fake blacklisted IDs; returns
   `blacklist_hit: bool`, which the risk model treats as a near-automatic high-risk signal.
4. Keep `failed_rules` as a list — this is what Module 7 (Gemini) turns into a human sentence, so don't
   discard the detail, only the numeric score.

**Acceptance criteria:** a document with an expired date returns `expiry_valid: False` and a reduced
`validation_pass_rate`; a blacklisted seeded ID returns `blacklist_hit: True`.

---

## Module 4 — Tampering Detection

**Goal:** produce `tampering_score` (0 = clean, 1 = highly suspicious) from image-level forensic signals
— no trained CNN, this is the heuristic MVP as scoped earlier.

**Location:** `backend/app/modules/tampering.py`

**Input:** original (untouched) uploaded bytes + preprocessed array + EXIF dict from Module 1
**Output:** `{"tampering_score": 0.15, "signals": {"ela_score": 0.12, "metadata_score": 0.2, "noise_inconsistency": 0.1}}`

**Dependencies:** `opencv-python`, `Pillow`, `scikit-image`, `numpy`

**Tasks:**
1. **Error Level Analysis (ELA):** re-save the image at a fixed JPEG quality (e.g. 90), diff against
   original, amplify — regions that were spliced/edited recompress differently and light up in the
   diff. Reduce to a single scalar via mean intensity in the diff map, normalize to 0–1.
2. **Metadata analysis:** check EXIF for inconsistencies — missing EXIF entirely on a claimed "camera
   photo," software tags indicating editing tools (Photoshop/GIMP in `Software` tag), mismatched
   `DateTimeOriginal` vs file metadata. Score this as a 0–1 scalar too.
3. **Noise inconsistency (optional but cheap):** split image into blocks, compute local noise variance
   per block (`scikit-image`'s `estimate_sigma`); spliced regions often have different noise
   characteristics than the surrounding image — flag high variance-of-variance across blocks.
4. Combine the three signals into `tampering_score` via a simple weighted average (document this
   weighting explicitly — it's a heuristic, say so in the pitch, don't call it "AI-detected").
5. Keep the individual `signals` dict in the response — feeds Gemini's explanation ("flagged: EXIF
   software tag indicates image editing software").

**Acceptance criteria:** running this on a genuine-looking scan vs. a deliberately re-saved/cropped
version of the same image shows the tampered version scoring meaningfully higher.

---

## Module 5 — Face Verification (DeepFace)

**Goal:** confirm the person in the document photo matches a live capture — produces
`face_match_score`.

**Location:** `backend/app/modules/face_match.py`

**Input:** document image (crop of photo region), live capture image (from frontend camera widget)
**Output:** `{"face_match_score": 0.91, "face_detected_in_doc": true, "face_detected_live": true}`

**Dependencies:** `deepface`, `tensorflow` (DeepFace pulls this in), `retina-face` or default detector

**Tasks:**
1. Crop the document photo region — either from a known fixed layout position (fast, per document
   type) or via DeepFace's own face detector run on the full document image as fallback.
2. Call `DeepFace.verify(img1_path=doc_face, img2_path=live_capture, model_name="Facenet",
   enforce_detection=False)` — `enforce_detection=False` matters because document photos are small/
   low-res and strict detection will throw on legitimate images.
3. `DeepFace.verify` returns a distance; convert to a 0–1 **similarity** score (`1 - normalized
   distance`) since the risk model expects higher = more trustworthy, not a raw distance.
4. Handle the "no live capture provided" case gracefully (e.g. batch/offline verification mode) —
   return `face_match_score: null` and let the risk-scoring module substitute a neutral default rather
   than crashing the pipeline.
5. Cache/load the DeepFace model once at startup, same reasoning as PaddleOCR — cold-loading per
   request will make the demo visibly slow.

**Acceptance criteria:** same-person pair scores high similarity; mismatched pair scores low; missing
live capture doesn't throw an unhandled exception.

---

## Module 6 — Risk Scoring (your trained model)

**Goal:** assemble the exact 12-feature vector your `risk_model.pkl` expects and get a prediction +
probability.

**Location:** `backend/app/modules/risk_scoring.py`, model file at `backend/app/models/risk_model.pkl`

**Confirmed model contract** (inspected directly from your uploaded file — a scikit-learn
`RandomForestClassifier`, 200 trees, max_depth=10, `class_weight='balanced'`):

```python
FEATURE_ORDER = [
    "ocr_confidence", "validation_pass_rate", "id_checksum_valid", "expiry_valid",
    "tampering_score", "face_match_score", "blacklist_hit",
    "document_type_Aadhaar", "document_type_Driving License",
    "document_type_PAN Card", "document_type_Passport", "document_type_Visa",
]
# classes_: [0, 1]  → 0 = genuine, 1 = fraudulent
```

**Input:** outputs from Modules 2–5 + `document_type` string
**Output:** `{"risk_score": 78, "prediction": "fraudulent", "probability": 0.78, "feature_vector": {...}}`

**Dependencies:** `scikit-learn==1.6.1` (**pin this exact version** — the model was pickled under 1.6.1;
loading it under a newer version throws an `InconsistentVersionWarning` and risks silently different
tree behavior), `joblib`, `pandas`, `numpy`

**Tasks:**
1. Load the model once at startup: `joblib.load("app/models/risk_model.pkl")`.
2. Build a single-row `pandas.DataFrame` with columns in **exactly** `FEATURE_ORDER` above:
   - Booleans (`id_checksum_valid`, `expiry_valid`, `blacklist_hit`) → cast to `int` (0/1).
   - `document_type` → one-hot manually (set the matching `document_type_*` column to 1, all others 0)
     rather than using `pd.get_dummies` on a single row, which would silently drop missing categories.
   - Any missing/null feature (e.g. `face_match_score` when no live capture) → substitute a documented
     neutral default (e.g. dataset mean, or `0.5`) — **do not** silently pass `NaN`, RandomForest in
     sklearn doesn't accept it.
3. `model.predict_proba(row)[0][1]` → fraud probability; `risk_score = round(probability * 100)`.
4. Return both the score and the full feature vector used — Module 7 needs the vector, and it's useful
   for the audit log / "why was this flagged" view either way.
5. Write a small startup self-test: run one known-genuine and one known-fraudulent synthetic vector
   through the loaded model and assert the predictions land on the expected side — catches a silently
   broken model load before it reaches a demo.

**Acceptance criteria:** feeding the model the synthetic "genuine" and "fraudulent" example rows from
your earlier notebook reproduces the same class predictions here.

---

## Module 7 — Gemini Explanation Layer

**Goal:** turn the numeric outputs into one officer-readable sentence — the human-usability layer on
top of the raw score.

**Location:** `backend/app/modules/explain.py`

**Input:** risk score + feature vector + `failed_rules` (Module 3) + `signals` (Module 4)
**Output:** `{"explanation": "Flagged: MRZ checksum mismatch on date of birth, face similarity low (42%)."}`

**Dependencies:** `google-genai` (official Gemini SDK)

**Tasks:**
1. Store `GEMINI_API_KEY` in env config, never hardcode.
2. Build a compact prompt template — pass only the numeric signals and failed-rule names, **not** raw
   PII beyond what's needed (avoid sending full name/DOB to the LLM if a redacted summary suffices):
   ```
   Given: risk_score=78, failed_rules=[dob_checksum], tampering_signals={ela: 0.6, metadata: 0.3},
   face_match_score=0.42, blacklist_hit=False.
   Write one concise sentence for a border officer explaining why this document was flagged, in plain language.
   ```
3. Call `gemini-2.0-flash` (or current fast tier) for low latency — this runs per-document in a live
   demo, don't reach for a slower/larger model here.
4. Add a fallback: if the Gemini call fails/times out (no network, rate limit), fall back to a
   template-generated sentence built directly from `failed_rules`/`signals` so the pipeline never hard-
   fails on the explanation step alone.
5. Cap response length in the prompt (e.g. "one sentence, under 30 words") — this is a dashboard badge,
   not a report.

**Acceptance criteria:** a high-risk and a low-risk input each produce a distinct, accurate one-line
explanation; a simulated Gemini API failure falls back gracefully without breaking the request.

---

## Module 8 — FastAPI Orchestration + Audit Log

**Goal:** wire Modules 1–7 into one request/response cycle and persist every verification for the audit
trail.

**Location:** `backend/app/routers/documents.py`, `backend/app/db/`

**Endpoints:**
```
POST /api/documents/verify
  multipart form: document_file, live_capture_file (optional), document_type
  → runs full pipeline synchronously, returns full result JSON

GET  /api/audit
  → paginated list of past verifications (for the audit-trail dashboard page)

GET  /api/audit/{id}
  → single verification detail, including feature vector + explanation
```

**Dependencies:** `fastapi`, `uvicorn`, `sqlalchemy`, `python-multipart`, `sqlite` (swap for Postgres
later if needed — SQLite is fine for a hackathon demo)

**Tasks:**
1. Define the `VerificationResult` and `AuditLog` SQLAlchemy models: store document type, extracted
   fields (as JSON), all module scores, final risk score/prediction, explanation text, timestamp — this
   *is* the "digital trail for investigations" line from your problem statement, make sure it's
   genuinely queryable, not just a raw dump.
2. In the `/verify` endpoint: call Modules 1→2→(3 and 4 in parallel via `asyncio.gather` since they're
   independent)→5→6→7 in sequence, catching and logging per-module failures without crashing the whole
   request (e.g. if face match fails because no live capture was sent, continue with the rest).
3. Write the full result to the DB before returning it to the client — audit log must be async but
   awaited before response, not fire-and-forget, or a crash right after response would lose the record.
4. Add basic request validation (file size limits, allowed mime types) via FastAPI's `UploadFile` +
   Pydantic — reject obviously invalid uploads before they hit PaddleOCR/DeepFace.
5. CORS middleware configured for the React dev server origin.

**Acceptance criteria:** a full end-to-end call with a real test image returns a complete JSON response
in a few seconds and a matching row appears in the audit DB.

---

## Module 9 — React + Tailwind Frontend

**Goal:** officer-facing dashboard: upload, see risk result, browse audit trail.

**Location:** `frontend/src/`

**Pages:**
- `UploadPage` — document upload + optional live face capture (browser camera via `getUserMedia`) +
  document-type selector → calls `POST /api/documents/verify`.
- `ResultPage` — risk score badge (color-coded: green <30, amber 30–70, red >70), extracted field
  table with per-field confidence, Gemini's one-line explanation prominently displayed, expandable
  "raw signals" section for the technically curious judge.
- `AuditTrailPage` — paginated table from `GET /api/audit`, filterable by risk level/document type.

**Components:** `DocumentUploader`, `FaceCaptureWidget`, `RiskScoreBadge`, `FieldConfidenceTable`,
`ExplanationCard`.

**Dependencies:** `react`, `vite`, `tailwindcss`, `axios` (or `fetch`), `react-router-dom`

**Tasks:**
1. Scaffold with Vite (`npm create vite@latest frontend -- --template react`), add Tailwind per its
   Vite integration docs.
2. `services/api.js` — thin axios wrapper around the two backend endpoints, with typed response
   shapes matching Module 8's JSON contract.
3. `FaceCaptureWidget` — request camera permission, capture a still frame to a `Blob`, attach to the
   multipart form alongside the document file.
4. `RiskScoreBadge` — pure presentational component taking `{score, prediction}`, no business logic.
5. Loading/error states on `UploadPage` — the pipeline takes a few seconds (OCR + DeepFace are not
   instant), show a spinner with stage labels if feasible ("Reading document…", "Checking for
   tampering…", "Matching face…") rather than a bare spinner.

**Acceptance criteria:** full flow — upload a document image, optionally capture a live photo, submit,
see a populated result page — works against the running FastAPI backend with no console errors.

---

## Build order (recommended)

1. **Module 6 first, standalone** — you already have the trained model; write and unit-test the
   feature-vector assembly against your own synthetic notebook rows before anything else depends on it.
2. **Module 2 (OCR + MRZ checksum)** — highest-value, most convincing single feature.
3. **Module 3 (validation)** — cheap, no external deps, unlocks two more risk-model features.
4. **Module 4 (tampering)** — independent, can be built in parallel by a second team member.
5. **Module 5 (face match)** — independent, third team member, heaviest dependency install (TensorFlow).
6. **Module 8 (orchestration)** — wire 2–6 together once each is unit-tested alone.
7. **Module 7 (Gemini)** — bolt on last, it's a thin layer over already-working scores.
8. **Module 9 (frontend)** — can start in parallel from day one against a mocked API response shaped
   like Module 8's contract, then swap to the real backend once it's live.

---

## Environment / requirements.txt (backend)

```
fastapi
uvicorn[standard]
python-multipart
sqlalchemy
paddleocr
paddlepaddle
opencv-python
Pillow
scikit-image
scikit-learn==1.6.1
joblib
pandas
numpy
deepface
tensorflow
google-genai
python-dateutil
```

Pin `scikit-learn==1.6.1` specifically — see Module 6 note on the version-mismatch warning from the
provided `risk_model.pkl`.
