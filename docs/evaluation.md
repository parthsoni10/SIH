# SENTINEL AI — Model Evaluation & Benchmark Instructions

## 1. Evaluate AI Image Detector

```bash
python training/evaluate_ai_detector.py
```
Calculates Accuracy, Precision, Recall, F1, ROC-AUC, FPR, FNR, and Confusion Matrix on `datasets/test/`.

---

## 2. Run Model Benchmark (V1 vs V2)

```bash
python benchmark/benchmark_models.py
```
Generates comparison reports:
- `benchmark/benchmark_report.json`
- `benchmark/benchmark_report.md`
