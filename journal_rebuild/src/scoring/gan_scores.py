from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from journal_rebuild.src.models.tcn_adversarial import TCNDiscriminator, WindowDataset


def _minmax_norm(values: np.ndarray, low: float, high: float) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32).reshape(-1)
    denom = max(float(high - low), 1e-12)
    return np.clip((arr - float(low)) / denom, 0.0, 1.0).astype(np.float32)


def _loader(features: np.ndarray, starts: np.ndarray, window_size: int, batch_size: int) -> DataLoader:
    labels = np.zeros(len(starts), dtype=np.uint8)
    return DataLoader(WindowDataset(features, starts, window_size, labels), batch_size=int(batch_size), shuffle=False, num_workers=0)


def embed_windows(
    discriminator: TCNDiscriminator,
    *,
    features: np.ndarray,
    starts: np.ndarray,
    window_size: int,
    batch_size: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    discriminator.eval()
    loader = _loader(features, starts, window_size, batch_size)
    embeddings: list[np.ndarray] = []
    sd_raw: list[np.ndarray] = []
    with torch.no_grad():
        for batch_x, _ in loader:
            batch_x = batch_x.to(device)
            feats = discriminator.forward_features(batch_x)
            emb = feats.mean(dim=2)
            logits = discriminator(batch_x).squeeze(1)
            normality = torch.sigmoid(logits)
            anomaly_sd = 1.0 - normality
            embeddings.append(emb.cpu().numpy().astype(np.float32))
            sd_raw.append(anomaly_sd.cpu().numpy().astype(np.float32))
    return np.concatenate(embeddings, axis=0), np.concatenate(sd_raw, axis=0)


def feature_deviation_scores(emb: np.ndarray, ref: np.ndarray) -> np.ndarray:
    emb = np.asarray(emb, dtype=np.float32)
    ref = np.asarray(ref, dtype=np.float32)
    mu = ref.mean(axis=0, keepdims=True)
    diff = emb - mu
    return np.sqrt(np.maximum((diff * diff).sum(axis=1), 0.0)).astype(np.float32)


def score_windows(
    discriminator: TCNDiscriminator,
    *,
    features: np.ndarray,
    starts: np.ndarray,
    ref_features: np.ndarray,
    ref_starts: np.ndarray,
    window_size: int,
    batch_size: int,
    alpha: float,
    device: torch.device,
) -> dict[str, np.ndarray]:
    ref_emb, ref_sd_raw = embed_windows(
        discriminator,
        features=ref_features,
        starts=ref_starts,
        window_size=window_size,
        batch_size=batch_size,
        device=device,
    )
    emb, sd_raw = embed_windows(
        discriminator,
        features=features,
        starts=starts,
        window_size=window_size,
        batch_size=batch_size,
        device=device,
    )
    ref_sf_raw = feature_deviation_scores(ref_emb, ref_emb)
    sf_raw = feature_deviation_scores(emb, ref_emb)
    sd_normalized = _minmax_norm(sd_raw, float(np.min(ref_sd_raw)), float(np.max(ref_sd_raw)))
    sf_normalized = _minmax_norm(sf_raw, float(np.min(ref_sf_raw)), float(np.max(ref_sf_raw)))
    fused_score = float(alpha) * sd_normalized + (1.0 - float(alpha)) * sf_normalized
    return {
        "SD_raw": np.asarray(sd_raw, dtype=np.float32),
        "SF_raw": np.asarray(sf_raw, dtype=np.float32),
        "SD_normalized": np.asarray(sd_normalized, dtype=np.float32),
        "SF_normalized": np.asarray(sf_normalized, dtype=np.float32),
        "fused_score": np.asarray(fused_score, dtype=np.float32),
    }
