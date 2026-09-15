import sys
import joblib
import numpy as np
import pandas as pd
import logging
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

backend_path = Path(__file__).resolve().parent.parent / "backend"
sys.path.insert(0, str(backend_path))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("train_risk_model_v2")

MODEL_DIR = backend_path / "app" / "models"
MODEL_V2_PATH = MODEL_DIR / "risk_model_v2.pkl"

FEATURE_ORDER_V2 = [
    "ocr_confidence",
    "validation_pass_rate",
    "id_checksum_valid",
    "expiry_valid",
    "tampering_score",
    "face_match_score",
    "blacklist_hit",
    "synthetic_generation_score",
    "ai_probability",
    "frequency_score",
    "synthetic_noise_score",
    "forensic_consistency_score",
    "document_type_Aadhaar",
    "document_type_Driving License",
    "document_type_PAN Card",
    "document_type_Passport",
    "document_type_Visa",
]


def generate_synthetic_training_data(n_samples: int = 1200):
    """
    Generates synthetic training dataset covering genuine, tampered, AI-generated,
    and blacklisted document feature distributions.
    """
    np.random.seed(42)
    data = []
    labels = []

    doc_types = ["Aadhaar", "Driving License", "PAN Card", "Passport", "Visa"]

    for _ in range(n_samples):
        # 0 = genuine (60%), 1 = fraud (40%)
        is_fraud = np.random.rand() < 0.4
        labels.append(1 if is_fraud else 0)

        dt = np.random.choice(doc_types)
        doc_one_hot = {f"document_type_{d}": 1.0 if dt == d else 0.0 for d in doc_types}

        if not is_fraud:
            # Genuine photo profile
            ocr_conf = np.random.uniform(0.85, 0.99)
            val_pass = np.random.uniform(0.90, 1.0)
            id_chk = 1.0 if np.random.rand() > 0.02 else 0.0
            exp_val = 1.0 if np.random.rand() > 0.02 else 0.0
            tamp_score = np.random.uniform(0.02, 0.22)
            face_score = np.random.uniform(0.85, 0.99)
            bl_hit = 0.0
            synth_score = np.random.uniform(0.0, 0.25)
            ai_prob = np.random.uniform(0.0, 0.20)
            freq_score = np.random.uniform(0.0, 0.25)
            noise_score = np.random.uniform(0.0, 0.30)
            consistency = np.random.uniform(0.85, 1.0)
        else:
            # Fraud profile (tampered or synthetic)
            is_ai_synthetic = np.random.rand() > 0.5
            bl_hit = 1.0 if np.random.rand() < 0.25 else 0.0

            if is_ai_synthetic:
                ocr_conf = np.random.uniform(0.70, 0.95)
                val_pass = np.random.uniform(0.60, 1.0)
                id_chk = 1.0 if np.random.rand() > 0.2 else 0.0
                exp_val = 1.0 if np.random.rand() > 0.2 else 0.0
                tamp_score = np.random.uniform(0.05, 0.35)
                face_score = np.random.uniform(0.30, 0.90)
                synth_score = np.random.uniform(0.65, 0.99)
                ai_prob = np.random.uniform(0.70, 0.99)
                freq_score = np.random.uniform(0.50, 0.95)
                noise_score = np.random.uniform(0.60, 0.95)
                consistency = np.random.uniform(0.10, 0.50)
            else:
                # Digital tampering
                ocr_conf = np.random.uniform(0.40, 0.80)
                val_pass = np.random.uniform(0.30, 0.75)
                id_chk = 0.0 if np.random.rand() < 0.6 else 1.0
                exp_val = 0.0 if np.random.rand() < 0.5 else 1.0
                tamp_score = np.random.uniform(0.50, 0.95)
                face_score = np.random.uniform(0.20, 0.65)
                synth_score = np.random.uniform(0.10, 0.40)
                ai_prob = np.random.uniform(0.05, 0.35)
                freq_score = np.random.uniform(0.10, 0.40)
                noise_score = np.random.uniform(0.10, 0.40)
                consistency = np.random.uniform(0.20, 0.60)

        row = {
            "ocr_confidence": ocr_conf,
            "validation_pass_rate": val_pass,
            "id_checksum_valid": id_chk,
            "expiry_valid": exp_val,
            "tampering_score": tamp_score,
            "face_match_score": face_score,
            "blacklist_hit": bl_hit,
            "synthetic_generation_score": synth_score,
            "ai_probability": ai_prob,
            "frequency_score": freq_score,
            "synthetic_noise_score": noise_score,
            "forensic_consistency_score": consistency,
            **doc_one_hot,
        }
        data.append(row)

    df = pd.DataFrame(data)[FEATURE_ORDER_V2]
    return df, np.array(labels)


def train_risk_model_v2():
    logger.info("Generating synthetic training and validation dataset for Risk Model V2...")
    X, y = generate_synthetic_training_data(n_samples=1600)

    train_size = 1200
    X_train, X_val = X.iloc[:train_size], X.iloc[train_size:]
    y_train, y_val = y[:train_size], y[train_size:]

    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        class_weight="balanced"
    )
    model.fit(X_train, y_train)

    # Evaluate on validation split
    val_preds = model.predict(X_val)
    val_probs = model.predict_proba(X_val)[:, 1]

    acc = accuracy_score(y_val, val_preds)
    prec = precision_score(y_val, val_preds)
    rec = recall_score(y_val, val_preds)
    f1 = f1_score(y_val, val_preds)
    roc_auc = roc_auc_score(y_val, val_probs)

    logger.info(
        f"[Risk Model V2 Metrics] Acc: {acc:.4f} | Prec: {prec:.4f} | Rec: {rec:.4f} | F1: {f1:.4f} | ROC-AUC: {roc_auc:.4f}"
    )

    joblib.dump(model, MODEL_V2_PATH)
    logger.info(f"Successfully saved Risk Model V2 to {MODEL_V2_PATH}")


if __name__ == "__main__":
    train_risk_model_v2()
