# SENTINEL — Root Cause Analysis & V3 Fix Plan

Diagnosis of verification #177 (PAN Card, 9/15/2026), which returned
`MANUAL_REVIEW / UNCALIBRATED MODEL` on a genuine document.

---

## Part 1 — What actually broke

### 1.1 The detector head is collapsed, not merely uncalibrated

Your dashboard reported:

```
AI Probability          55.7%
Patch Consistency (TopK) 55.6%
Corroborating Signals   0 / 4
Model Loaded YES · Trained YES · Calibration NOT CALIBRATED
```

Global and patch top-k agreeing to within 0.1 percentage points is the
signature of a **dead classifier head**. Multi-scale patch aggregation exists
precisely to produce *variance* across regions — the photo area, the QR block,
and the guilloche background have completely different statistics. If every
patch returns ~0.556, the network is not looking at the input. It has
collapsed to the class prior.

Four causes, in order of likelihood:

1. The classifier head was never unfrozen during fine-tuning.
2. Learning rate drove the head to a constant that minimises loss on an
   imbalanced set.
3. The checkpoint was saved before head convergence.
4. `load_state_dict(..., strict=False)` silently discarded the head weights
   because of a key-name mismatch — this one is common and produces exactly
   a random-init head with a near-constant softmax.

**Run `ml/calibrate_v3.py` first.** It reports output standard deviation. If
`std < 0.05`, cause is confirmed and no amount of threshold tuning will help.

Before any full retrain, run the **overfit test**: take 200 images, train
until train accuracy > 0.95. If the model cannot memorise 200 images, the
training loop itself is broken and a 5,000-image run will only waste a day.

### 1.2 Domain mismatch

Generic AI-detector datasets teach "photos of the world vs diffusion art". A
photograph of a laminated ID card under office lighting is neither. If your
`real/` folder is phone photos and your `ai_generated/` folder is clean
1024×1024 PNGs, the model learns *"JPEG artefacts vs no JPEG artefacts"* —
which is a compression detector wearing an AI-detector label, and it collapses
the instant it meets a re-photographed synthetic card.

Fix: `recapture_augment()` in `ml/calibrate_v3.py`. Apply it to every
synthetic training image — perspective tilt, optical blur, laminate glare,
sensor noise, JPEG re-encode at Q62–93. This forces the model onto content and
texture cues instead of container cues.

### 1.3 Recapture destroyed the signals your score depended on

Your pipeline also logged `Image is blurry (Laplacian variance: 79.7)` and
`HIGH_JPEG_COMPRESSION`, with image quality at 58%.

Blur and JPEG both suppress high-frequency energy. So does synthesis. Without
conditioning on measured degradation, the feature is **not identifiable** — a
blurry genuine card and a sharp synthetic card produce the same raw number.
That is why a real 2.1× class separation in local variance was reported as a
15% frequency anomaly.

### 1.4 The fusion rule could never fire

The "2 strong signals" rule counted `global_prob`, `patch_topk`, and
`noise_anomaly` as three independent votes. They are not — all three measure
high-frequency surface statistics. It is one physical measurement counted
three times. Meanwhile the uncalibrated model still carried 50% of the weight
and pinned the fused score at 0.557, the exact centre of the decision space
where no verdict is reachable.

### 1.5 The decision engine was degenerate

With `is_calibrated=False`, **every** document routes to `MANUAL_REVIEW`:
`GENUINE` required positive genuine evidence that nothing could emit (hence
`AUTHENTICITY_NOT_ESTABLISHED`), `AI_GENERATED` required corroboration the
collapsed model blocked, and the 0.35–0.80 inconclusive band swallowed the
constant 0.557.

A fail-closed system that closes on 100% of traffic provides no security.
Officers learn to rubber-stamp the review queue, and you are worse off than
with no system at all.

---

## Part 2 — What separates your two reference images

Measured directly on the uploaded pair:

| Signal | Real (iPhone) | Gemini render | Separation |
|---|---|---|---|
| Container / mode | MPO / RGB | PNG / RGBA | decisive |
| EXIF | `Make=Apple`, `Model=iPhone 14 Pro`, full optical block | **none at all** | decisive |
| Ancillary data | ICC + XMP + MPO index | none | decisive |
| Local variance (5×5 median) | 4.49 | 2.16 | **2.1×** |
| HF/LF energy ratio (windowed) | 6.79 | 6.37 | 1.07× (weak) |
| QR decodes | ✗ (too blurry) | ✗ | no signal |
| ConvNeXt probability | 0.557 | ~0.55 | none |

Two findings matter most.

**Provenance is your strongest signal and it needs no training.** The AI image
has zero metadata; the real one carries a complete optical capture chain. Your
V2 pipeline scored this as a 35% `metadata_anomaly` and then capped it as a
lone signal — throwing away the best evidence you had.

**Your QR code check is missing, and it is the definitive one.** A current PAN
card carries a QR encoding a digitally signed demographic payload. A diffusion
model renders a QR as *texture* — convincing finder squares, garbage
Reed–Solomon blocks. The probability of a generative model emitting a valid
codeword across a version-20+ symbol is effectively zero. It will never decode.

The honest caveat: on your genuine capture the real QR *also* failed to decode
(Laplacian 84 on the crop, laminate glare). So non-decode is only admissible as
evidence when the crop is sharp enough that a genuine symbol would have
decoded. `qr_integrity.py` gates on exactly that:

```
decode succeeds           -> strong POSITIVE genuineness evidence
decode fails, crop sharp  -> strong synthesis evidence
decode fails, crop soft   -> NO evidence; escalate capture quality instead
```

This asymmetry is the whole design. It is also why improving your **capture
protocol** buys you more accuracy than any model change: get the QR in focus
and you get a near-deterministic answer.

---

## Part 3 — The V3 architecture

Four **physically independent** signal families replace the old five-signal sum:

| Family | Measures | Survives recapture? | Weight |
|---|---|---|---|
| **A. PROVENANCE** | container, EXIF, ICC, quantisation history | ✅ yes | 0.35 |
| **B. SEMANTIC** | QR codeword validity, payload↔print cross-check | ✅ yes | 0.35 |
| **C. TEXTURE** | local variance, spectral rolloff *(one family)* | ⚠️ degrades | 0.20 |
| **D. LEARNED** | ConvNeXt global + patch *(one family)* | ⚠️ degrades | 0.10 |

Three rules make this work:

1. **Corroboration requires ≥2 different families, and ≥1 must be A or B.**
   A and B survive blur and compression; C and D do not. This is what stops a
   badly-photographed genuine card from being labelled AI_GENERATED.

2. **An uncalibrated model is excluded, not down-weighted.** `is_calibrated`
   and `is_weights_loaded` become distinct states. A collapsed model
   contributes nothing rather than dragging the score to 0.5.

3. **Unavailable families are renormalised out, never scored 0.0.** V2 treated
   "could not measure" as "looks genuine".

The decision engine gains a third state. V2 had verdict-or-review; V3
distinguishes `CONCLUSIVE` / `INSUFFICIENT` / `CONTRADICTORY`, and reports
`blocking_gaps` so the officer knows *what to fix* rather than just seeing
"requires review". Two independent routes reach `GENUINE`, so one offline
module cannot deadlock the pipeline.

### Verified on your reference pair

```
GENUINE (iPhone)      provenance 0.000  texture 0.002  ->  AI 0.001
                      families 2/4, no strong signals
                      VERDICT: GENUINE [CONCLUSIVE] conf 0.896
                      AUTHENTICITY_CONFIRMED_BY_PROVENANCE

SYNTHETIC (Gemini)    provenance 0.780  texture 0.999  ->  AI 0.860
                      strong: [PROVENANCE, TEXTURE], anchor present
                      VERDICT: AI_GENERATED [CONCLUSIVE] conf 0.860
```

Both correct — **with the detector still excluded as uncalibrated.** The
pipeline reaches confident verdicts without the broken model, which is the
point.

> ⚠️ **Read this before quoting those numbers.** The texture thresholds were
> fitted on these same two images, so the 0.001 / 0.999 margin is overfit by
> construction. It demonstrates the architecture is sound. It is **not** an
> accuracy figure. Real thresholds must come from `calibrate_v3.py` on a
> proper dataset, and your benchmark needs far more than 8 samples before any
> claim is defensible.

---

## Part 4 — Order of work

**Day 1 — stop the bleeding.** Drop in the four modules. Set
`is_calibrated=False` explicitly. Your pipeline starts clearing genuine
documents again on provenance + structure alone.

**Day 2 — machine-readable integrity.** `pip install pyzbar` plus
`libzbar0`. Wire `qr_integrity.py` in after OCR so it can cross-check the
payload against the printed number. Cross-check catches the ALTERED case that
ELA misses on a re-photographed card.

**Day 3 — capture protocol.** Add a client-side quality gate to `UploadPage.jsx`:
reject below Laplacian 120, warn on glare, require the QR region in focus.
This is the single highest-leverage change and it costs no ML work.

**Day 4–5 — dataset.** Minimum 800 images per class, *captured the same way*.
Photograph real cards. Generate synthetics, then run every one through
`recapture_augment()`. Never mix clean PNGs into the synthetic class.

**Day 6 — retrain.** Overfit test on 200 images first. If train accuracy stays
below 0.95, fix the loop before scaling up.

**Day 7 — calibrate.** `calibrate_v3.py` writes real thresholds and reports
per-feature AUC. Only set `is_calibrated=True` when the model clears AUC 0.85.

---

## Part 5 — For the SIH demo

Two things will be asked, so prepare them:

**"What if the attacker strips EXIF?"** Then family A goes quiet and you fall
back to B + C. Say so openly — it demonstrates you understand your own failure
modes, which scores better than claiming 100%. The honest answer is that
stripped EXIF is itself weak evidence, which is exactly why the corroboration
rule exists.

**"What is your false-positive rate?"** Your current benchmark is 8 samples,
which cannot support a 0.00% claim. Either enlarge it or present it as a
smoke test. A panel that catches an overclaimed metric will discount
everything else you say.

Also: strengthen the ALTERED path. The two cards differ in printed PAN number
and in signature, and neither ELA nor the AI detector reliably catches a field
edit on a re-photographed card. The QR payload cross-check does, and it is the
most defensible answer you can give on document alteration.

One last note — you are testing with a real PAN card carrying live personal
data. Use synthetic or consented test fixtures in your repo and demo, and keep
the real card out of version control and out of any dataset you share.
