from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score


def compute_auc_ap(y_true: np.ndarray, y_score: np.ndarray) -> tuple[float, float]:
    labels = np.asarray(y_true, dtype=np.uint8).reshape(-1)
    scores = np.asarray(y_score, dtype=np.float32).reshape(-1)
    auc = float("nan") if np.unique(labels).size < 2 else float(roc_auc_score(labels, scores))
    ap = 0.0 if int(labels.sum()) == 0 else float(average_precision_score(labels, scores))
    return auc, ap


def metrics_at_threshold(y_true: np.ndarray, y_score: np.ndarray, threshold: float) -> dict[str, float]:
    labels = np.asarray(y_true, dtype=np.uint8).reshape(-1)
    scores = np.asarray(y_score, dtype=np.float32).reshape(-1)
    preds = (scores >= float(threshold)).astype(np.uint8)
    tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0, 1]).ravel()
    precision = 0.0 if (tp + fp) == 0 else float(tp / (tp + fp))
    recall = 0.0 if (tp + fn) == 0 else float(tp / (tp + fn))
    f1 = 0.0 if (precision + recall) == 0 else float(2.0 * precision * recall / (precision + recall))
    fpr = 0.0 if (fp + tn) == 0 else float(fp / (fp + tn))
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": float(tp),
        "fp": float(fp),
        "tn": float(tn),
        "fn": float(fn),
        "fpr": fpr,
    }
