import sys
import json
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix
)

root_path = Path(__file__).resolve().parent.parent
backend_path = root_path / "backend"
sys.path.insert(0, str(root_path))
sys.path.insert(0, str(backend_path))

from app.modules.risk_scoring import predict_risk, predict_risk_v2
from training.train_risk_model_v2 import generate_synthetic_training_data


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("benchmark_models")

BENCHMARK_DIR = Path(__file__).resolve().parent
REPORT_JSON = BENCHMARK_DIR / "benchmark_report.json"
REPORT_MD = BENCHMARK_DIR / "benchmark_report.md"


def compute_metrics(y_true, y_pred, y_prob):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel() if cm.shape == (2, 2) else (0, 0, 0, 0)
    fpr = float(fp / max(fp + tn, 1))
    fnr = float(fn / max(fn + tp, 1))

    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(y_true, y_prob)), 4),
        "fpr": round(fpr, 4),
        "fnr": round(fnr, 4),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}
    }


def run_benchmark():
    logger.info("Generating evaluation dataset for model benchmark...")
    df, y_true = generate_synthetic_training_data(n_samples=500)

    # Evaluate Risk Model V1 Baseline
    v1_preds = []
    v1_probs = []
    for idx, row in df.iterrows():
        doc_type = "Passport"
        for dt in ["Aadhaar", "Driving License", "PAN Card", "Passport", "Visa"]:
            if row.get(f"document_type_{dt}", 0.0) == 1.0:
                doc_type = dt

        res_v1 = predict_risk(
            ocr_confidence=row["ocr_confidence"],
            validation_pass_rate=row["validation_pass_rate"],
            id_checksum_valid=bool(row["id_checksum_valid"]),
            expiry_valid=bool(row["expiry_valid"]),
            tampering_score=row["tampering_score"],
            face_match_score=row["face_match_score"],
            blacklist_hit=bool(row["blacklist_hit"]),
            document_type=doc_type,
            synthetic_generation_score=row["synthetic_generation_score"],
        )
        v1_preds.append(1 if res_v1["prediction"] == "fraudulent" else 0)
        v1_probs.append(res_v1["probability"])

    metrics_v1 = compute_metrics(y_true, v1_preds, v1_probs)

    # Evaluate Risk Model V2 Upgraded
    v2_preds = []
    v2_probs = []
    for idx, row in df.iterrows():
        doc_type = "Passport"
        for dt in ["Aadhaar", "Driving License", "PAN Card", "Passport", "Visa"]:
            if row.get(f"document_type_{dt}", 0.0) == 1.0:
                doc_type = dt

        res_v2 = predict_risk_v2(
            ocr_confidence=row["ocr_confidence"],
            validation_pass_rate=row["validation_pass_rate"],
            id_checksum_valid=bool(row["id_checksum_valid"]),
            expiry_valid=bool(row["expiry_valid"]),
            tampering_score=row["tampering_score"],
            face_match_score=row["face_match_score"],
            blacklist_hit=bool(row["blacklist_hit"]),
            document_type=doc_type,
            synthetic_generation_score=row["synthetic_generation_score"],
            ai_probability=row["ai_probability"],
            frequency_score=row["frequency_score"],
            synthetic_noise_score=row["synthetic_noise_score"],
            forensic_consistency_score=row["forensic_consistency_score"],
        )
        v2_preds.append(1 if res_v2["prediction"] == "fraudulent" else 0)
        v2_probs.append(res_v2["probability"])

    metrics_v2 = compute_metrics(y_true, v2_preds, v2_probs)

    report = {
        "evaluation_samples": len(df),
        "risk_model_v1": metrics_v1,
        "risk_model_v2": metrics_v2,
        "delta": {
            "accuracy_improvement": round(metrics_v2["accuracy"] - metrics_v1["accuracy"], 4),
            "f1_improvement": round(metrics_v2["f1"] - metrics_v1["f1"], 4),
            "roc_auc_improvement": round(metrics_v2["roc_auc"] - metrics_v1["roc_auc"], 4),
            "fpr_reduction": round(metrics_v1["fpr"] - metrics_v2["fpr"], 4),
        }
    }

    with open(REPORT_JSON, "w") as f:
        json.dump(report, f, indent=2)

    md_content = f"""# SENTINEL AI — Model Benchmark Report

Evaluating **Risk Model V1 (Baseline)** vs **Risk Model V2 (Upgraded Architecture)** across {len(df)} evaluation samples.

| Metric | Risk Model V1 (Baseline) | Risk Model V2 (Target Architecture) | Delta |
|---|---|---|---|
| **Accuracy** | {metrics_v1['accuracy']:.4f} | {metrics_v2['accuracy']:.4f} | +{report['delta']['accuracy_improvement']:.4f} |
| **Precision** | {metrics_v1['precision']:.4f} | {metrics_v2['precision']:.4f} | - |
| **Recall** | {metrics_v1['recall']:.4f} | {metrics_v2['recall']:.4f} | - |
| **F1 Score** | {metrics_v1['f1']:.4f} | {metrics_v2['f1']:.4f} | +{report['delta']['f1_improvement']:.4f} |
| **ROC-AUC** | {metrics_v1['roc_auc']:.4f} | {metrics_v2['roc_auc']:.4f} | +{report['delta']['roc_auc_improvement']:.4f} |
| **False Positive Rate (FPR)** | {metrics_v1['fpr']:.4f} | {metrics_v2['fpr']:.4f} | -{report['delta']['fpr_reduction']:.4f} |
| **False Negative Rate (FNR)** | {metrics_v1['fnr']:.4f} | {metrics_v2['fnr']:.4f} | - |
"""
    with open(REPORT_MD, "w") as f:
        f.write(md_content)

    logger.info(f"Benchmark completed successfully! Reports saved to {REPORT_JSON} and {REPORT_MD}")
    print(md_content)
    return report


if __name__ == "__main__":
    run_benchmark()
