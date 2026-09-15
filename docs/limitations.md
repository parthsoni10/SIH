# SENTINEL AI — Known Limitations & Operational Guardrails

1. **No 100% Accuracy Guarantee**: No automated system can detect 100% of adversarial AI generations or physical document spoofing. High-value transactions should combine automated screening with human officer verification.
2. **Web Browser EXIF Stripping**: Modern web browsers, messaging apps (WhatsApp, Telegram), and social media strip EXIF metadata from uploaded images. Missing EXIF is treated as ambiguous supporting evidence and NEVER causes a fraud classification alone.
3. **Scan Flatness vs Low Noise Floor**: Scanners produce low noise variance across flat background regions. Thresholds are calibrated (`median_noise < 2.0`) to avoid false positives on clean scans.
4. **Bilingual OCR Character Noise**: Secondary Hindi/regional language labels adjacent to English text can produce OCR noise fragments. Fuzzy label matching prevents these from triggering synthetic microtext flags.
5. **CPU / GPU Hardware Scaling**: EfficientNet-B0 inference runs on CPU by default (~15-40ms latency) and automatically leverages CUDA GPU acceleration when available.
