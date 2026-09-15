# SENTINEL AI — AI-Generated Image Detection

## 1. EfficientNet-B0 Backbone (`ai_image_detector.py`)

- **Input**: 224×224 RGB image normalized with ImageNet mean and std.
- **Model**: Pretrained `EfficientNet-B0` fine-tuned with binary cross-entropy loss (`BCEWithLogitsLoss`).
- **Feature Constraint**: Evaluates purely pixel-level visual features. Filenames, EXIF strings, and generator names are NEVER used as classification features.

---

## 2. 2D FFT Frequency Analysis (`frequency_analysis.py`)

- Converts document image to grayscale and computes 2D Fast Fourier Transform (`np.fft.fft2`).
- Measures radial energy distributions across low-frequency (<=15%), mid-frequency (15%-50%), and high-frequency (>50%) bands.
- Flags unnatural high-frequency grid spikes and anomalous spectral entropy characteristic of generative AI latent space upsamplers.

---

## 3. 2-Signal Corroboration Rule (`forensic_fusion.py`)

A high-risk synthetic classification (`LIKELY_AI_GENERATED` / fraud override) requires at least **two independent strong signals** (`strong_signal_count >= 2`):

1. Deep Learning AI Detector Probability (`ai_probability >= 0.65`)
2. 2D FFT Spectral Frequency Anomaly (`frequency_score >= 0.45`)
3. Spatial Noise Floor Smoothness (`noise_score >= 0.70`)
4. EXIF AI Generator/Editor Software Tag (`metadata_score >= 0.80`)
5. Mangled Microtext Density (`ocr_layout_score >= 0.45`)
6. Gemini Visual Authenticity Judgment (`visual_score < 0.35`)

> **Safeguard**: Missing EXIF alone NEVER triggers a synthetic fraud override.
