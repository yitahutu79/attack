from __future__ import annotations

import numpy as np


def threshold_from_benign_fpr(benign_scores: np.ndarray, target_fpr: float) -> float:
    scores = np.asarray(benign_scores, dtype=np.float32).reshape(-1)
    if len(scores) == 0:
        raise ValueError("Cannot calibrate threshold from empty benign score set")
    q = float(np.clip(1.0 - float(target_fpr), 0.0, 1.0))
    return float(np.quantile(scores, q, method="linear"))
