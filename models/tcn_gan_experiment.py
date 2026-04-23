#!/usr/bin/env python3
"""CICIDS2017：基于 TCN-GAN 的异常检测训练与评估脚本。"""

from __future__ import annotations

import argparse
import glob
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Iterable


def _print_dep_install_hint(missing: list[str]) -> None:
    exe = sys.executable or "python"
    venv = os.environ.get("VIRTUAL_ENV") or os.environ.get("CONDA_PREFIX")
    venv_hint = f"\n当前环境：{venv}" if venv else ""
    missing_text = "、".join(missing)
    print(
        "\n".join(
            [
                "依赖未安装，脚本无法运行。",
                f"缺少：{missing_text}",
                f"当前 Python：{exe}{venv_hint}",
                "",
                "请用同一个 Python 执行下面命令安装依赖：",
                f"{exe} -m pip install -U pip",
                f"{exe} -m pip install numpy pandas scikit-learn torch",
                "",
                "如果你是 Mac 芯片（M1/M2/M3），PyTorch 安装完成后脚本会自动优先使用 MPS（Metal）加速。",
            ]
        )
    )


def _require_deps_or_exit() -> None:
    missing: list[str] = []
    try:
        import numpy as _  # noqa: F401
    except ModuleNotFoundError:
        missing.append("numpy")
    try:
        import pandas as _  # noqa: F401
    except ModuleNotFoundError:
        missing.append("pandas")
    try:
        import sklearn as _  # noqa: F401
    except ModuleNotFoundError:
        missing.append("scikit-learn")
    try:
        import torch as _  # noqa: F401
    except ModuleNotFoundError:
        missing.append("torch")

    if missing:
        _print_dep_install_hint(missing)
        raise SystemExit(2)


_require_deps_or_exit()

_ATTACK_ROOT = Path(__file__).resolve().parents[1]
if str(_ATTACK_ROOT) not in sys.path:
    sys.path.insert(0, str(_ATTACK_ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split  # noqa: E402
from sklearn.preprocessing import MinMaxScaler  # noqa: E402
from torch.utils.data import DataLoader, Dataset  # noqa: E402

from dataset_loaders import list_dataset_files, load_windowed_dataset, normalize_dataset_name  # noqa: E402

try:  # PyTorch 新版本建议使用 parametrizations.weight_norm
    from torch.nn.utils.parametrizations import weight_norm as _weight_norm  # type: ignore[attr-defined]
except Exception:  # pragma: no cover
    from torch.nn.utils import weight_norm as _weight_norm  # type: ignore[no-redef]

if os.environ.get("TCN_GAN_DISABLE_WEIGHT_NORM", "").strip().lower() in {"1", "true", "yes"}:
    # Some macOS/PyTorch wheel combinations segfault during backward through
    # parametrized weight_norm. Keep an explicit escape hatch for reproducible
    # CPU smoke tests and cross-dataset bring-up runs.
    def _weight_norm(layer):  # type: ignore[no-redef]
        return layer

try:
    from tqdm.auto import tqdm  # type: ignore
except Exception:  # pragma: no cover
    tqdm = None  # type: ignore


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_cicids(dataset_dir: str) -> pd.DataFrame:
    csv_paths = sorted(glob.glob(os.path.join(dataset_dir, "*.csv")))
    if not csv_paths:
        raise FileNotFoundError(f"目录中未找到 CSV 文件：{dataset_dir}")
    frames = []
    for path in csv_paths:
        # CICIDS2017 常见问题：列名/字段前后有空格，用 skipinitialspace 先处理一部分。
        df = pd.read_csv(path, skipinitialspace=True, low_memory=False)
        df = df.dropna(axis=0)
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def load_cicids_files(dataset_dir: str) -> dict[str, pd.DataFrame]:
    csv_paths = sorted(glob.glob(os.path.join(dataset_dir, "*.csv")))
    if not csv_paths:
        raise FileNotFoundError(f"目录中未找到 CSV 文件：{dataset_dir}")
    out: dict[str, pd.DataFrame] = {}
    for path in csv_paths:
        name = os.path.basename(path)
        out[name] = pd.read_csv(path, skipinitialspace=True, low_memory=False)
    return out


def _find_label_col(columns: list[str]) -> str | None:
    for c in columns:
        if str(c).strip().lower() == "label":
            return c
    return None


def preprocess_raw_frame(df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    # 清理缺失/无穷，并标准化列名；返回“未归一化”的数值特征表与二分类标签。
    df = df.replace([np.inf, -np.inf], np.nan).dropna(axis=0).copy()
    df.columns = [str(c).strip() for c in df.columns]
    label_col = _find_label_col(list(df.columns))
    if label_col is None:
        raise ValueError("数据集必须包含 'Label' 列（可能是列名前后空格导致未识别）")
    labels = df[label_col].astype(str).str.strip().str.upper()
    binary_label = (labels != "BENIGN").astype(np.uint8).to_numpy()
    df = df.drop(columns=[label_col], errors="ignore")
    numeric = df.select_dtypes(include=[np.number])
    if numeric.empty:
        raise ValueError("未找到数值型特征列（无法训练）")
    return numeric, binary_label


def fit_scaler_from_train(
    train_numeric_frames: list[pd.DataFrame],
    train_labels: list[np.ndarray],
    benign_only: bool,
) -> MinMaxScaler:
    parts: list[pd.DataFrame] = []
    for numeric, labels in zip(train_numeric_frames, train_labels, strict=True):
        if benign_only:
            mask = labels == 0
            if np.any(mask):
                parts.append(numeric.loc[mask])
        else:
            parts.append(numeric)
    if not parts:
        raise ValueError("训练集没有可用于拟合归一化器的样本（可能是训练文件里没有 BENIGN）")
    scaler = MinMaxScaler()
    scaler.fit(pd.concat(parts, ignore_index=True).astype(np.float32))
    return scaler


def preprocess_frame(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    df = df.replace([np.inf, -np.inf], np.nan).dropna(axis=0)
    # 统一清理列名：去掉前后空格，避免 ' Label' / 'Label ' 这种情况。
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]

    label_col = None
    for c in df.columns:
        if str(c).strip().lower() == "label":
            label_col = c
            break
    if label_col is None:
        raise ValueError("数据集必须包含 'Label' 列（可能是列名前后空格导致未识别）")

    labels = df[label_col].astype(str).str.strip().str.upper()
    binary_label = (labels != "BENIGN").astype(np.uint8).to_numpy()
    df = df.drop(columns=[label_col], errors="ignore")
    numeric = df.select_dtypes(include=[np.number])
    if numeric.empty:
        raise ValueError("未找到数值型特征列（无法训练）")
    # 归一化只用正常样本拟合，减少攻击样本对缩放范围的影响，也能降低数据泄漏风险。
    benign_mask = binary_label == 0
    if not np.any(benign_mask):
        raise ValueError("没有正常（BENIGN）样本，无法拟合归一化器")
    scaler = MinMaxScaler()
    scaler.fit(numeric.loc[benign_mask].astype(np.float32))
    features = scaler.transform(numeric.astype(np.float32))
    return features.astype(np.float32), binary_label


def build_sequences(
    features: np.ndarray,
    labels: np.ndarray,
    window_size: int,
    stride: int,
    anomaly_ratio: float,
    *,
    progress: bool = False,
    desc: str = "",
) -> tuple[np.ndarray, np.ndarray]:
    sequences: list[np.ndarray] = []
    seq_labels: list[int] = []
    total = len(features) - window_size + 1
    if total <= 0:
        raise ValueError("window_size 大于可用样本数量")
    it: Iterable[int] = range(0, total, stride)
    if progress and tqdm is not None:
        it = tqdm(
            it,
            total=int((total + stride - 1) // stride),
            desc=desc or "build_sequences",
            leave=False,
            file=sys.stderr,
        )
    for start in it:
        window = features[start : start + window_size]
        sequences.append(window)
        window_label = labels[start : start + window_size].sum() / window_size
        seq_labels.append(int(window_label >= anomaly_ratio))
    return np.stack(sequences), np.array(seq_labels, dtype=np.uint8)


def split_sequences(
    sequences: np.ndarray,
    labels: np.ndarray,
    test_size: float,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    benign_idx = np.where(labels == 0)[0]
    anomaly_idx = np.where(labels == 1)[0]
    if len(benign_idx) == 0:
        raise ValueError("没有可用于训练的正常（BENIGN）序列")
    if len(anomaly_idx) == 0:
        raise ValueError("没有异常序列（无法评估）")
    train_idx, benign_test_idx = train_test_split(
        benign_idx, test_size=test_size, random_state=random_state
    )
    test_idx = np.concatenate([benign_test_idx, anomaly_idx])
    rng = np.random.default_rng(random_state)
    rng.shuffle(test_idx)
    return sequences[train_idx], sequences[test_idx], labels[test_idx]


class SequenceDataset(Dataset):
    def __init__(self, sequences: np.ndarray, labels: np.ndarray | None = None) -> None:
        self.sequences = torch.from_numpy(sequences).float()
        self.labels = (
            torch.from_numpy(labels).long() if labels is not None else torch.zeros(len(sequences), dtype=torch.long)
        )

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.sequences[idx], self.labels[idx]


class Chomp1d(nn.Module):
    def __init__(self, chomp_size: int) -> None:
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.chomp_size <= 0:
            return x
        return x[:, :, :-self.chomp_size]


class TemporalBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, dilation, dropout):
        super().__init__()
        padding = (kernel_size - 1) * dilation
        self.conv1 = _weight_norm(
            nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
            )
        )
        self.chomp1 = Chomp1d(padding)
        self.conv2 = _weight_norm(
            nn.Conv1d(
                out_channels,
                out_channels,
                kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
            )
        )
        self.chomp2 = Chomp1d(padding)
        self.dropout = nn.Dropout(dropout)
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(x)
        out = self.chomp1(out)
        out = F.relu(out)
        out = self.dropout(out)
        out = self.conv2(out)
        out = self.chomp2(out)
        out = F.relu(out)
        out = self.dropout(out)
        residual = x if self.downsample is None else self.downsample(x)
        return F.relu(out + residual)


class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=3, dropout=0.2):
        super().__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            layers.append(
                TemporalBlock(
                    in_channels,
                    out_channels,
                    kernel_size,
                    stride=1,
                    dilation=dilation_size,
                    dropout=dropout,
                )
            )
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class TCNGenerator(nn.Module):
    def __init__(self, seq_len: int, feat_dim: int, latent_dim: int, hidden_channels: Iterable[int], dropout: float):
        super().__init__()
        channels = list(hidden_channels)
        self.seq_len = seq_len
        self.initial = nn.Linear(latent_dim, channels[0])
        self.tcn = TemporalConvNet(channels[0], channels, dropout=dropout)
        self.to_feature = nn.Conv1d(channels[-1], feat_dim, kernel_size=1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = self.initial(z).unsqueeze(-1)
        x = x.expand(-1, -1, self.seq_len)
        x = self.tcn(x)
        out = self.to_feature(x)
        return torch.sigmoid(out.permute(0, 2, 1))


class TCNDiscriminator(nn.Module):
    def __init__(
        self,
        seq_len: int,
        feat_dim: int,
        hidden_channels: Iterable[int],
        dropout: float,
        pooling: str = "mean",
    ):
        super().__init__()
        channels = list(hidden_channels)
        self.pooling = str(pooling).lower().strip()
        self.tcn = TemporalConvNet(feat_dim, channels, dropout=dropout)
        self.classifier = nn.Conv1d(channels[-1], 1, kernel_size=1)
        # Lightweight temporal attention: produce per-time weights from features, then do weighted pooling.
        self.attn = nn.Conv1d(channels[-1], 1, kernel_size=1) if self.pooling in ("attn", "attention") else None

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B,T,F] -> [B,F,T] -> [B,C,T]
        x = x.permute(0, 2, 1)
        return self.tcn(x)

    def forward_time_logits(self, features: torch.Tensor) -> torch.Tensor:
        # features: [B,C,T] -> logits_t: [B,T]
        return self.classifier(features).squeeze(1)

    def forward_attn_weights_from_features(self, features: torch.Tensor) -> torch.Tensor | None:
        if self.attn is None:
            return None
        # attn_logits: [B,1,T] -> weights: [B,T]
        attn_logits = self.attn(features).squeeze(1)
        return torch.softmax(attn_logits, dim=1)

    def forward_attn_weights(self, x: torch.Tensor) -> torch.Tensor | None:
        feats = self.forward_features(x)
        return self.forward_attn_weights_from_features(feats)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.forward_features(x)
        logits_t = self.forward_time_logits(features)  # [B,T]
        if self.attn is not None and self.pooling in ("attn", "attention"):
            w = self.forward_attn_weights_from_features(features)  # [B,T]
            assert w is not None
            logits = (w * logits_t).sum(dim=1, keepdim=True)  # [B,1]
            return logits
        logits = logits_t.mean(dim=1, keepdim=True)  # [B,1]
        return logits


def train_one_epoch(
    generator: nn.Module,
    discriminator: nn.Module,
    train_loader: DataLoader,
    device: torch.device,
    g_optimizer: torch.optim.Optimizer,
    d_optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    latent_dim: int,
    gan_loss: str = "vanilla",
    gp_lambda: float = 10.0,
    n_critic: int = 1,
    *,
    progress: bool = False,
    desc: str = "",
) -> tuple[float, float]:
    generator.train()
    discriminator.train()
    g_loss_avg = 0.0
    d_loss_avg = 0.0
    gan_loss = str(gan_loss).lower().strip()
    n_critic = max(int(n_critic), 1)
    it = train_loader
    if progress and tqdm is not None:
        it = tqdm(it, desc=desc or "train", leave=False, file=sys.stderr)
    for real_seq, _ in it:
        real_seq = real_seq.to(device)
        batch_size = real_seq.size(0)
        real_labels = torch.ones(batch_size, 1, device=device)
        fake_labels = torch.zeros(batch_size, 1, device=device)

        # Discriminator/Critic updates
        d_steps = n_critic if gan_loss in ("wgan-gp", "wgangan-gp", "wgan_gp", "wgangp") else 1
        for _ in range(d_steps):
            noise = torch.randn(batch_size, latent_dim, device=device)
            fake_seq = generator(noise)
            d_real = discriminator(real_seq)
            d_fake = discriminator(fake_seq.detach())
            if gan_loss in ("wgan-gp", "wgan_gp", "wgangp"):
                # Wasserstein critic: maximize D(real) - D(fake) => minimize D(fake) - D(real)
                d_loss = (d_fake.mean() - d_real.mean())
                # Gradient penalty
                eps = torch.rand(batch_size, 1, 1, device=device)
                x_hat = eps * real_seq + (1.0 - eps) * fake_seq.detach()
                x_hat.requires_grad_(True)
                d_hat = discriminator(x_hat).sum()
                grads = torch.autograd.grad(
                    outputs=d_hat,
                    inputs=x_hat,
                    create_graph=True,
                    retain_graph=True,
                    only_inputs=True,
                )[0]
                grads = grads.reshape(batch_size, -1)
                gp = ((grads.norm(2, dim=1) - 1.0) ** 2).mean()
                d_loss = d_loss + float(gp_lambda) * gp
            else:
                d_loss = criterion(d_real, real_labels) + criterion(d_fake, fake_labels)
            d_optimizer.zero_grad()
            d_loss.backward()
            d_optimizer.step()

        # Generator update
        noise = torch.randn(batch_size, latent_dim, device=device)
        fake_seq = generator(noise)
        d_fake = discriminator(fake_seq)
        if gan_loss in ("wgan-gp", "wgan_gp", "wgangp"):
            g_loss = -d_fake.mean()
        else:
            g_loss = criterion(d_fake, real_labels)
        g_optimizer.zero_grad()
        g_loss.backward()
        g_optimizer.step()

        d_loss_avg += d_loss.item()
        g_loss_avg += g_loss.item()
        if progress and tqdm is not None:
            try:
                it.set_postfix({"d": float(d_loss.item()), "g": float(g_loss.item())})
            except Exception:
                pass

    num_batches = len(train_loader)
    return d_loss_avg / num_batches, g_loss_avg / num_batches


def evaluate_model(
    discriminator: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
    *,
    progress: bool = False,
    desc: str = "",
) -> tuple[float, float, np.ndarray, np.ndarray]:
    discriminator.eval()
    scores: list[float] = []
    labels: list[int] = []
    with torch.no_grad():
        it = test_loader
        if progress and tqdm is not None:
            it = tqdm(it, desc=desc or "eval", leave=False, file=sys.stderr)
        for seq, label in it:
            seq = seq.to(device)
            logits = discriminator(seq)
            probs = torch.sigmoid(logits).squeeze(1)
            scores.extend((1.0 - probs).cpu().numpy().tolist())
            labels.extend(label.numpy().tolist())
    y_true = np.asarray(labels, dtype=np.int64)
    y_score = np.asarray(scores, dtype=np.float32)
    # 当测试集只有单一类别（例如 Monday 全 BENIGN）时，AUC/AP 的定义会退化；
    # 这里显式处理，避免 sklearn 发出 warning 并让输出更可读。
    if np.unique(y_true).size < 2:
        auc = float("nan")
    else:
        auc = float(roc_auc_score(y_true, y_score))
    if float(y_true.sum()) <= 0.0:
        ap = 0.0
    else:
        ap = float(average_precision_score(y_true, y_score))
    return auc, ap, y_true, y_score


def report_classification(y_true: np.ndarray, y_score: np.ndarray) -> dict[str, float]:
    # 用 PR 曲线选择使 F1 最大的阈值（仅用于“怎么看结果”的参考；部署建议按误报率选阈值）。
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    thresholds = np.asarray(thresholds, dtype=np.float32)
    if thresholds.size == 0:
        return {"best_threshold": float("nan"), "best_f1": 0.0, "best_precision": 0.0, "best_recall": 0.0}

    # precision/recall 的长度比 thresholds 多 1
    precision = np.asarray(precision[:-1], dtype=np.float32)
    recall = np.asarray(recall[:-1], dtype=np.float32)
    denom = precision + recall
    # np.where 会同时计算两边表达式，直接写除法会触发除零 warning；用 mask 避免。
    f1 = np.zeros_like(denom, dtype=np.float32)
    mask = denom > 0
    f1[mask] = (2 * precision[mask] * recall[mask]) / denom[mask]
    best_idx = int(np.argmax(f1))
    best_th = float(thresholds[best_idx])

    y_pred = (y_score >= best_th).astype(np.int64)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    eps = 1e-12
    best_precision = float(tp / (tp + fp + eps))
    best_recall = float(tp / (tp + fn + eps))
    best_f1 = float(2 * best_precision * best_recall / (best_precision + best_recall + eps))
    fpr = float(fp / (fp + tn + eps))
    return {
        "best_threshold": best_th,
        "best_precision": best_precision,
        "best_recall": best_recall,
        "best_f1": best_f1,
        "tn": float(tn),
        "fp": float(fp),
        "fn": float(fn),
        "tp": float(tp),
        "fpr": fpr,
    }


def _sample_rows(x: np.ndarray, max_n: int, seed: int) -> np.ndarray:
    x = np.asarray(x)
    max_n = int(max_n)
    if max_n <= 0 or len(x) <= max_n:
        return x
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(x), size=max_n, replace=False)
    return x[idx]


def score_sequences(
    discriminator: nn.Module, seqs: np.ndarray, device: torch.device, batch_size: int
) -> np.ndarray:
    discriminator.eval()
    seqs = np.asarray(seqs, dtype=np.float32)
    if len(seqs) == 0:
        return np.asarray([], dtype=np.float32)
    ds = SequenceDataset(seqs)
    loader = DataLoader(ds, batch_size=int(batch_size), shuffle=False)
    scores: list[float] = []
    with torch.no_grad():
        it = loader
        show = bool(os.environ.get("TCN_GAN_PROGRESS", "").strip()) and tqdm is not None
        if show:
            it = tqdm(it, desc="score_sequences", leave=False, file=sys.stderr)
        for batch in it:
            x = batch
            # SequenceDataset returns (seq, label); DataLoader may collate it as list/tuple.
            if isinstance(batch, (list, tuple)) and len(batch) == 2:
                x = batch[0]
            x = x.to(device)
            logits = discriminator(x)
            probs = torch.sigmoid(logits).squeeze(1)
            scores.extend((1.0 - probs).cpu().numpy().tolist())
    return np.asarray(scores, dtype=np.float32)


def _minmax_norm(x: np.ndarray, lo: float, hi: float) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    denom = max(float(hi - lo), 1e-12)
    out = (x - float(lo)) / denom
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def embed_sequences(
    discriminator: TCNDiscriminator, seqs: np.ndarray, device: torch.device, batch_size: int
) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns:
      - emb: [N,C] pooled discriminator features
      - unk: [N] baseline unknownness score = 1 - D(x)
    """
    discriminator.eval()
    seqs = np.asarray(seqs, dtype=np.float32)
    if len(seqs) == 0:
        return np.zeros((0, 0), dtype=np.float32), np.asarray([], dtype=np.float32)
    ds = SequenceDataset(seqs)
    loader = DataLoader(ds, batch_size=int(batch_size), shuffle=False)
    embs: list[np.ndarray] = []
    unks: list[np.ndarray] = []
    with torch.no_grad():
        it = loader
        show = bool(os.environ.get("TCN_GAN_PROGRESS", "").strip()) and tqdm is not None
        if show:
            it = tqdm(it, desc="embed_sequences", leave=False, file=sys.stderr)
        for batch in it:
            x = batch
            if isinstance(batch, (list, tuple)) and len(batch) == 2:
                x = batch[0]
            x = x.to(device)
            feats = discriminator.forward_features(x)  # [B,C,T]
            emb = feats.mean(dim=2)  # [B,C]
            logits = discriminator.classifier(feats).mean(dim=2)  # [B,1]
            probs = torch.sigmoid(logits).squeeze(1)
            embs.append(emb.cpu().numpy().astype(np.float32))
            unks.append((1.0 - probs).cpu().numpy().astype(np.float32))
    return np.concatenate(embs, axis=0), np.concatenate(unks, axis=0)


def feature_deviation_scores(
    emb: np.ndarray,
    emb_ref: np.ndarray,
    method: str = "l2",
    eps: float = 1e-6,
) -> np.ndarray:
    emb = np.asarray(emb, dtype=np.float32)
    emb_ref = np.asarray(emb_ref, dtype=np.float32)
    if len(emb) == 0:
        return np.asarray([], dtype=np.float32)
    mu = emb_ref.mean(axis=0, keepdims=True)
    diff = emb - mu
    method = str(method).lower().strip()
    if method in ("mahal", "mahalanobis"):
        cov = np.cov(emb_ref.T).astype(np.float32)
        cov = cov + (eps * np.eye(cov.shape[0], dtype=np.float32))
        inv = np.linalg.pinv(cov).astype(np.float32)
        m = (diff @ inv) * diff
        return np.sqrt(np.maximum(m.sum(axis=1), 0.0)).astype(np.float32)
    return np.sqrt(np.maximum((diff * diff).sum(axis=1), 0.0)).astype(np.float32)


def compute_anomaly_scores(
    discriminator: TCNDiscriminator,
    x: np.ndarray,
    device: torch.device,
    batch_size: int,
    score_mode: str,
    score_alpha: float,
    x_ref_benign: np.ndarray | None = None,
) -> np.ndarray:
    """
    score_mode:
      - prob: 1 - D(x) (baseline)
      - feat_l2 / feat_mahal: feature deviation from benign reference
      - fused: α*norm(prob) + (1-α)*norm(feat_dev)
    """
    score_mode = str(score_mode).lower().strip()
    if score_mode == "prob" or x_ref_benign is None:
        return score_sequences(discriminator, x, device, batch_size)

    emb_ref, unk_ref = embed_sequences(discriminator, x_ref_benign, device, batch_size)
    emb, unk = embed_sequences(discriminator, x, device, batch_size)
    if emb_ref.size == 0 or emb.size == 0:
        return score_sequences(discriminator, x, device, batch_size)

    dev_method = "mahal" if score_mode == "feat_mahal" else "l2"
    dev_ref = feature_deviation_scores(emb_ref, emb_ref, method=dev_method)
    dev = feature_deviation_scores(emb, emb_ref, method=dev_method)

    dev_n = _minmax_norm(dev, float(np.min(dev_ref)), float(np.max(dev_ref)))
    unk_n = _minmax_norm(unk, float(np.min(unk_ref)), float(np.max(unk_ref)))
    if score_mode in ("feat_l2", "feat_mahal"):
        return dev_n.astype(np.float32)
    a = float(score_alpha)
    return (a * unk_n + (1.0 - a) * dev_n).astype(np.float32)


def _prepare_ref_stats(
    discriminator: TCNDiscriminator,
    x_ref_benign: np.ndarray,
    device: torch.device,
    batch_size: int,
    score_mode: str,
    eps: float = 1e-6,
) -> dict[str, object] | None:
    """
    Prepare benign reference statistics for feature deviation normalization.
    Returned stats are consumed by _anomaly_score_torch for differentiable XAI.
    """
    score_mode = str(score_mode).lower().strip()
    if score_mode == "prob":
        return None
    emb_ref, unk_ref = embed_sequences(discriminator, x_ref_benign, device, batch_size)
    if emb_ref.size == 0:
        return None
    dev_method = "mahal" if score_mode == "feat_mahal" else "l2"
    mu = emb_ref.mean(axis=0).astype(np.float32)
    inv = None
    if dev_method in ("mahal", "mahalanobis"):
        cov = np.cov(emb_ref.T).astype(np.float32)
        cov = cov + (float(eps) * np.eye(cov.shape[0], dtype=np.float32))
        inv = np.linalg.pinv(cov).astype(np.float32)
    dev_ref = feature_deviation_scores(emb_ref, emb_ref, method=dev_method, eps=eps)
    return {
        "mu": mu,
        "inv": inv,
        "dev_min": float(np.min(dev_ref)) if dev_ref.size else 0.0,
        "dev_max": float(np.max(dev_ref)) if dev_ref.size else 1.0,
        "unk_min": float(np.min(unk_ref)) if unk_ref.size else 0.0,
        "unk_max": float(np.max(unk_ref)) if unk_ref.size else 1.0,
        "dev_method": dev_method,
    }


def _anomaly_score_torch(
    discriminator: TCNDiscriminator,
    x: torch.Tensor,
    *,
    score_mode: str,
    score_alpha: float,
    ref_stats: dict[str, object] | None,
) -> torch.Tensor:
    """
    Differentiable anomaly score for XAI.
    x: [B,T,F]
    returns: [B]
    """
    score_mode = str(score_mode).lower().strip()
    feats = discriminator.forward_features(x)  # [B,C,T]
    emb = feats.mean(dim=2)  # [B,C]
    # NOTE: discriminator(x) already applies pooling; but we need unk per window only.
    logits = discriminator(x).squeeze(1)  # [B]
    unk = 1.0 - torch.sigmoid(logits)  # [B]
    if score_mode == "prob" or ref_stats is None:
        return unk

    mu = torch.as_tensor(ref_stats["mu"], device=x.device, dtype=emb.dtype).reshape(1, -1)  # type: ignore[index]
    diff = emb - mu
    inv = ref_stats.get("inv")
    if inv is not None:
        inv_t = torch.as_tensor(inv, device=x.device, dtype=emb.dtype)  # [C,C]
        m = (diff @ inv_t) * diff
        dev = torch.sqrt(torch.clamp(m.sum(dim=1), min=0.0))
    else:
        dev = torch.sqrt(torch.clamp((diff * diff).sum(dim=1), min=0.0))

    dev_min = float(ref_stats.get("dev_min", 0.0))
    dev_max = float(ref_stats.get("dev_max", 1.0))
    unk_min = float(ref_stats.get("unk_min", 0.0))
    unk_max = float(ref_stats.get("unk_max", 1.0))

    dev_n = torch.clamp((dev - dev_min) / max(dev_max - dev_min, 1e-12), 0.0, 1.0)
    unk_n = torch.clamp((unk - unk_min) / max(unk_max - unk_min, 1e-12), 0.0, 1.0)
    if score_mode in ("feat_l2", "feat_mahal"):
        return dev_n
    a = float(score_alpha)
    return a * unk_n + (1.0 - a) * dev_n


def _xai_gradxinput_report(
    discriminator: TCNDiscriminator,
    test_seqs: np.ndarray,
    test_labels: np.ndarray,
    *,
    device: torch.device,
    batch_size: int,
    score_mode: str,
    score_alpha: float,
    ref_stats: dict[str, object] | None,
    feature_names: list[str] | None,
    n_samples: int,
    seed: int,
    out_dir: str,
) -> dict[str, object]:
    """
    Generate a simple XAI report via |grad * input| attributions.
    Outputs plots (if matplotlib exists) and a JSON report to out_dir.
    """
    x = np.asarray(test_seqs, dtype=np.float32)
    y = np.asarray(test_labels, dtype=np.int64).reshape(-1)
    if x.size == 0 or y.size == 0:
        raise ValueError("XAI: test_seqs/test_labels 为空")

    t = int(x.shape[1])
    fdim = int(x.shape[2])
    if not feature_names or len(feature_names) != fdim:
        feature_names = [f"f{i}" for i in range(fdim)]

    idx_b = np.where(y == 0)[0]
    idx_a = np.where(y == 1)[0]
    rng = np.random.default_rng(int(seed))
    n_each = max(int(n_samples) // 2, 1)
    idx_b_s = rng.choice(idx_b, size=min(idx_b.size, n_each), replace=False) if idx_b.size else np.asarray([], dtype=int)
    idx_a_s = rng.choice(idx_a, size=min(idx_a.size, n_each), replace=False) if idx_a.size else np.asarray([], dtype=int)

    def _accumulate(idxs: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, int]:
        if idxs.size == 0:
            return np.zeros((t,), dtype=np.float32), np.zeros((fdim,), dtype=np.float32), None, 0
        ds = SequenceDataset(x[idxs])
        loader = DataLoader(ds, batch_size=int(batch_size), shuffle=False)
        time_sum = np.zeros((t,), dtype=np.float64)
        feat_sum = np.zeros((fdim,), dtype=np.float64)
        attn_sum = np.zeros((t,), dtype=np.float64)
        count = 0
        discriminator.eval()
        for batch in loader:
            xb = batch
            if isinstance(batch, (list, tuple)) and len(batch) == 2:
                xb = batch[0]
            xb = xb.to(device).float()
            # Optional attention weights (no grad needed)
            with torch.no_grad():
                w = discriminator.forward_attn_weights(xb)
                if w is not None:
                    attn_sum += w.sum(dim=0).detach().cpu().numpy()
            xb.requires_grad_(True)
            discriminator.zero_grad(set_to_none=True)
            scores = _anomaly_score_torch(
                discriminator, xb, score_mode=score_mode, score_alpha=score_alpha, ref_stats=ref_stats
            )
            scores.mean().backward()
            g = xb.grad
            if g is None:
                continue
            attr = (g * xb).abs()  # [B,T,F]
            time_sum += attr.sum(dim=2).sum(dim=0).detach().cpu().numpy()
            feat_sum += attr.sum(dim=1).sum(dim=0).detach().cpu().numpy()
            count += int(xb.shape[0])
        if count <= 0:
            return np.zeros((t,), dtype=np.float32), np.zeros((fdim,), dtype=np.float32), None, 0
        attn_mean = (attn_sum / count).astype(np.float32)
        if discriminator.attn is None:
            attn_mean = None
        return (time_sum / count).astype(np.float32), (feat_sum / count).astype(np.float32), attn_mean, count

    benign_time, benign_feat, benign_attn, n_b = _accumulate(idx_b_s)
    anom_time, anom_feat, anom_attn, n_a = _accumulate(idx_a_s)

    os.makedirs(out_dir, exist_ok=True)
    report: dict[str, object] = {
        "xai_method": "gradxinput",
        "score_mode": str(score_mode),
        "score_alpha": float(score_alpha),
        "n_samples": int(n_samples),
        "n_benign": int(n_b),
        "n_anomaly": int(n_a),
        "time_importance": {"benign": benign_time.tolist(), "anomaly": anom_time.tolist()},
        "feature_importance": {
            "names": list(feature_names),
            "benign": benign_feat.tolist(),
            "anomaly": anom_feat.tolist(),
        },
    }
    if benign_attn is not None and anom_attn is not None:
        report["attn_weights"] = {
            "benign": benign_attn.tolist(),
            "anomaly": anom_attn.tolist(),
        }

    topk = min(20, fdim)
    order = np.argsort(-anom_feat)[:topk]
    report["top_features_by_anomaly_importance"] = [
        {"feature": feature_names[int(i)], "importance": float(anom_feat[int(i)])} for i in order
    ]

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # noqa: PLC0415

        fig = plt.figure(figsize=(10, 3.2))
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(benign_time, label=f"BENIGN (n={n_b})")
        ax.plot(anom_time, label=f"ANOM (n={n_a})")
        ax.set_title("TCN-GAN XAI: mean |grad*input| time importance")
        ax.set_xlabel("t (within window)")
        ax.set_ylabel("importance")
        ax.grid(True, alpha=0.25)
        ax.legend()
        p_time = os.path.join(out_dir, "xai_time_importance.png")
        fig.tight_layout()
        fig.savefig(p_time, dpi=160)
        plt.close(fig)
        report["plot_time_importance"] = p_time

        if benign_attn is not None and anom_attn is not None:
            fig = plt.figure(figsize=(10, 3.2))
            ax = fig.add_subplot(1, 1, 1)
            ax.plot(benign_attn, label=f"BENIGN attn (n={n_b})")
            ax.plot(anom_attn, label=f"ANOM attn (n={n_a})")
            ax.set_title("TCN-GAN Attention weights (mean over samples)")
            ax.set_xlabel("t (within window)")
            ax.set_ylabel("weight")
            ax.grid(True, alpha=0.25)
            ax.legend()
            p_attn = os.path.join(out_dir, "attn_weights.png")
            fig.tight_layout()
            fig.savefig(p_attn, dpi=160)
            plt.close(fig)
            report["plot_attn_weights"] = p_attn

        fig = plt.figure(figsize=(10, 4.5))
        ax = fig.add_subplot(1, 1, 1)
        top = np.argsort(-anom_feat)[:topk]
        names = [feature_names[int(i)] for i in top][::-1]
        vals = [float(anom_feat[int(i)]) for i in top][::-1]
        ax.barh(names, vals)
        ax.set_title("TCN-GAN XAI: top features by anomaly importance (|grad*input|)")
        ax.set_xlabel("importance")
        ax.grid(True, axis="x", alpha=0.25)
        p_feat = os.path.join(out_dir, "xai_feature_importance.png")
        fig.tight_layout()
        fig.savefig(p_feat, dpi=160)
        plt.close(fig)
        report["plot_feature_importance"] = p_feat
    except Exception as e:
        report["plot_error"] = str(e)

    p_json = os.path.join(out_dir, "xai_report.json")
    with open(p_json, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    report["report_json"] = p_json
    return report


def metrics_at_threshold(y_true: np.ndarray, y_score: np.ndarray, th: float) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=np.int64)
    y_score = np.asarray(y_score, dtype=np.float32)
    y_pred = (y_score >= float(th)).astype(np.int64)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    eps = 1e-12
    prec = float(tp / (tp + fp + eps))
    rec = float(tp / (tp + fn + eps))
    f1 = float(2 * prec * rec / (prec + rec + eps))
    fpr = float(fp / (fp + tn + eps))
    return {
        "threshold": float(th),
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "fpr": fpr,
        "tn": float(tn),
        "fp": float(fp),
        "fn": float(fn),
        "tp": float(tp),
    }


def benign_fpr_at_threshold(y_score_benign: np.ndarray, th: float) -> float:
    y_score_benign = np.asarray(y_score_benign, dtype=np.float32).reshape(-1)
    if y_score_benign.size == 0:
        return float("nan")
    return float((y_score_benign >= float(th)).mean())


def threshold_from_benign_fpr(benign_scores: np.ndarray, target_fpr: float) -> float:
    benign_scores = np.asarray(benign_scores, dtype=np.float32).reshape(-1)
    target_fpr = float(target_fpr)
    if not (0.0 < target_fpr < 1.0):
        raise ValueError("--target-fpr 必须在 (0,1) 范围内，例如 0.05")
    if benign_scores.size == 0:
        raise ValueError("没有可用于标定阈值的 BENIGN 分数")
    benign_scores = benign_scores[np.isfinite(benign_scores)]
    if benign_scores.size == 0:
        raise ValueError("BENIGN 分数全部为非有限值，无法标定阈值")
    # Want P(score >= th) ~= target_fpr  => th at (1-target_fpr) quantile.
    return float(np.quantile(benign_scores, 1.0 - target_fpr))


def main() -> None:
    parser = argparse.ArgumentParser(description="TCN-GAN：CICIDS2017 异常检测")
    parser.add_argument(
        "--dataset",
        default="cicids2017",
        choices=["cicids2017", "swat", "ton_iot", "generic"],
        help="数据集类型。cicids2017 保持原论文逻辑；swat/ton_iot/generic 使用通用窗口加载器。",
    )
    parser.add_argument("--data-dir", default="attack/dataset/CICIDS2017", help="CICIDS2017 CSV 目录路径")
    parser.add_argument("--window-size", type=int, default=32, help="滑动窗口长度（序列长度）")
    parser.add_argument("--stride", type=int, default=4, help="滑动窗口步长")
    parser.add_argument(
        "--anomaly-ratio",
        type=float,
        default=0.15,
        help="窗口内异常占比阈值（>= 该值则该窗口标为异常）",
    )
    parser.add_argument("--test-fraction", type=float, default=0.2, help="从正常窗口中划分测试集的比例")
    parser.add_argument("--epochs", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--test-batch-size", type=int, default=512)
    parser.add_argument("--latent-dim", type=int, default=64)
    parser.add_argument("--hidden-channels", nargs="+", type=int, default=[128, 128])
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument(
        "--disc-pooling",
        choices=["mean", "attn"],
        default="mean",
        help="判别器时间维池化方式：mean=平均池化；attn=轻量时间注意力加权池化（更适合预警）",
    )
    parser.add_argument(
        "--gan-loss",
        choices=["vanilla", "wgan-gp"],
        default="vanilla",
        help="GAN 训练损失：vanilla=BCE；wgan-gp=Wasserstein + gradient penalty（更稳定）",
    )
    parser.add_argument("--gp-lambda", type=float, default=10.0, help="WGAN-GP 的 gradient penalty 系数（默认 10.0）")
    parser.add_argument("--n-critic", type=int, default=5, help="WGAN-GP 的 critic 更新次数（默认 5；vanilla 下会忽略）")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "mps", "cuda"],
        default="auto",
        help="训练/评估设备。默认 auto；当前 macOS smoke test 可显式用 cpu 避免 MPS 后端问题。",
    )
    parser.add_argument(
        "--list-files",
        action="store_true",
        help="列出 data-dir 下的 CSV 文件名并退出（用于按天/按文件划分训练测试）",
    )
    parser.add_argument(
        "--train-files",
        nargs="*",
        default=[],
        help="训练使用的 CSV 文件名（只写文件名，不写路径）。例如：Monday-WorkingHours.pcap_ISCX.csv",
    )
    parser.add_argument(
        "--test-files",
        nargs="*",
        default=[],
        help="测试使用的 CSV 文件名（只写文件名，不写路径）。留空表示自动用除 train-files 外的所有文件",
    )
    parser.add_argument(
        "--train-benign-only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="训练时只使用 BENIGN 序列（推荐，符合无监督异常检测设定）",
    )
    parser.add_argument(
        "--save-best",
        default="",
        help="保存最佳模型权重的路径（留空表示不保存），例如：attack/tcn_gan_best.pt",
    )
    parser.add_argument(
        "--report-path",
        default="",
        help="保存测试集打分明细到 CSV（留空表示不保存），例如：attack/tcn_gan_scores.csv",
    )
    parser.add_argument(
        "--target-fpr",
        type=float,
        default=None,
        help="部署/论文友好：用训练集 BENIGN 序列的分数标定阈值，使 BENIGN 误报率≈该值（例如 0.05）。标定后会额外输出该阈值下的 Precision/Recall/F1/FPR",
    )
    parser.add_argument(
        "--score-mode",
        choices=["prob", "feat_l2", "feat_mahal", "fused"],
        default="prob",
        help="异常分数：prob=1-D(x)；feat_* 为判别器特征偏离；fused=α*prob+(1-α)*norm(feat_dev)",
    )
    parser.add_argument("--score-alpha", type=float, default=0.5, help="score-mode=fused 的 α（默认 0.5）")
    parser.add_argument(
        "--feat-calib-max-seqs",
        type=int,
        default=20000,
        help="特征偏离度参考分布使用的训练 BENIGN 序列上限（默认 20000，加速）",
    )
    parser.add_argument(
        "--calib-max-seqs",
        type=int,
        default=50000,
        help="标定阈值时最多用多少条训练序列（0=不限制；默认 50000，加速用）",
    )
    parser.add_argument(
        "--load",
        default="",
        help="加载已保存的 checkpoint（例如 --save-best 生成的 .pt），用于评估或继续训练",
    )
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="只做评估不训练（通常与 --load 一起使用）",
    )
    parser.add_argument(
        "--out-json",
        default="",
        help="可选：把本次评估/训练的关键结果保存为 JSON（便于自动汇总画表/写论文），例如 attack/results/final_experiments/manual_eval/tcn_gan_main.json",
    )
    parser.add_argument(
        "--xai-report",
        action="store_true",
        help="可选：对测试窗口做一次 XAI 诊断（grad*input），输出关键时间步/特征重要性图与 JSON 报告（主要用于调参与论文解释）",
    )
    parser.add_argument("--xai-samples", type=int, default=512, help="XAI 抽样窗口数（BENIGN/ANOM 各一半，默认 512）")
    parser.add_argument("--xai-batch-size", type=int, default=128, help="XAI 计算 batch size（默认 128）")
    parser.add_argument(
        "--xai-out-dir",
        default="",
        help="XAI 报告输出目录（默认自动生成到 attack/results/xai_tcn/<run_tag>/）",
    )
    parser.add_argument(
        "--progress",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="显示进度条（训练/评估/窗口构建）。如果你用 tee 重定向，进度条会走 stderr。",
    )
    parser.add_argument(
        "--scaler",
        choices=["minmax", "standard"],
        default="minmax",
        help="非 CICIDS 通用加载器使用的特征缩放方式；TCN-GAN 推荐 minmax。",
    )
    args = parser.parse_args()
    t_start = time.perf_counter()

    set_seed(args.seed)
    if args.device == "cpu":
        device = torch.device("cpu")
    elif args.device == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("--device cuda 但当前 torch.cuda.is_available() 为 False")
        device = torch.device("cuda")
    elif args.device == "mps":
        if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
            raise ValueError("--device mps 但当前 torch.backends.mps.is_available() 为 False")
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    # 两种模式：
    # 1) 默认模式：把所有文件合并后滑窗，再按“正常序列抽样 + 全部异常序列”构建测试集。
    # 2) 按文件划分模式：指定 --train-files/--test-files，避免跨文件滑窗与更贴近真实的“按天训练/按天测试”。
    train_names: list[str] = []
    test_names: list[str] = []
    feature_names: list[str] | None = None

    dataset_name = normalize_dataset_name(args.dataset)
    if dataset_name != "cicids2017":
        if args.list_files:
            for name in list_dataset_files(args.data_dir, dataset_name):
                print(name)
            return
        ds = load_windowed_dataset(
            dataset=dataset_name,
            data_dir=args.data_dir,
            train_files=args.train_files,
            test_files=args.test_files,
            window_size=args.window_size,
            stride=args.stride,
            anomaly_ratio=args.anomaly_ratio,
            train_benign_only=bool(args.train_benign_only),
            scaler=args.scaler,
            clip_minmax=True,
            progress=bool(args.progress),
        )
        train_names = list(ds.train_files)
        test_names = list(ds.test_files)
        feature_names = list(ds.feature_names)
        train_seqs = ds.x_train
        test_seqs = ds.x_test
        test_labels = ds.y_test
        print(
            f"Loaded {dataset_name}: train_windows={len(train_seqs)}, "
            f"test_windows={len(test_seqs)}, test_anom_ratio={float(test_labels.mean()):.4f}, "
            f"features={len(feature_names)}",
            flush=True,
        )
    elif args.list_files or args.train_files or args.test_files:
        files = load_cicids_files(args.data_dir)
        if args.list_files:
            for name in sorted(files.keys()):
                print(name)
            return

        all_names = set(files.keys())
        train_names = list(args.train_files) if args.train_files else sorted(all_names)
        missing_train = [n for n in train_names if n not in all_names]
        if missing_train:
            raise ValueError(f"train-files 中存在不存在的文件名：{missing_train}")
        test_names = list(args.test_files) if args.test_files else sorted(all_names.difference(train_names))
        missing_test = [n for n in test_names if n not in all_names]
        if missing_test:
            raise ValueError(f"test-files 中存在不存在的文件名：{missing_test}")
        if not test_names:
            raise ValueError("test-files 为空：请显式指定 --test-files，或让它自动选择 train-files 以外的文件")

        train_numeric_frames: list[pd.DataFrame] = []
        train_labels_list: list[np.ndarray] = []
        test_numeric_frames: list[pd.DataFrame] = []
        test_labels_list: list[np.ndarray] = []
        for name in train_names:
            numeric, y = preprocess_raw_frame(files[name])
            train_numeric_frames.append(numeric)
            train_labels_list.append(y)
        for name in test_names:
            numeric, y = preprocess_raw_frame(files[name])
            test_numeric_frames.append(numeric)
            test_labels_list.append(y)

        if train_numeric_frames and feature_names is None:
            feature_names = [str(c) for c in list(train_numeric_frames[0].columns)]

        scaler = fit_scaler_from_train(train_numeric_frames, train_labels_list, benign_only=True)

        train_seq_parts: list[np.ndarray] = []
        for idx, (numeric, y) in enumerate(zip(train_numeric_frames, train_labels_list, strict=True), start=1):
            x = scaler.transform(numeric.astype(np.float32)).astype(np.float32)
            x = np.clip(x, 0.0, 1.0)  # 与生成器 sigmoid 输出范围一致，避免判别器“范围投机”。
            t0 = time.perf_counter()
            seqs, seq_y = build_sequences(
                x,
                y,
                args.window_size,
                args.stride,
                args.anomaly_ratio,
                progress=bool(args.progress),
                desc=f"train windows {idx}/{len(train_numeric_frames)}",
            )
            if args.progress:
                print(f"train 窗口完成 {idx}/{len(train_numeric_frames)}：windows={len(seqs)} ({time.perf_counter()-t0:.2f}s)", flush=True)
            if args.train_benign_only:
                keep = seq_y == 0
                seqs = seqs[keep]
            if len(seqs):
                train_seq_parts.append(seqs)
        if not train_seq_parts:
            raise ValueError("训练序列为空：请检查 train-files 是否包含足够 BENIGN，或调整 window-size/stride/anomaly-ratio")
        train_seqs = np.concatenate(train_seq_parts, axis=0)

        test_seq_parts: list[np.ndarray] = []
        test_label_parts: list[np.ndarray] = []
        for idx, (numeric, y) in enumerate(zip(test_numeric_frames, test_labels_list, strict=True), start=1):
            x = scaler.transform(numeric.astype(np.float32)).astype(np.float32)
            x = np.clip(x, 0.0, 1.0)
            t0 = time.perf_counter()
            seqs, seq_y = build_sequences(
                x,
                y,
                args.window_size,
                args.stride,
                args.anomaly_ratio,
                progress=bool(args.progress),
                desc=f"test windows {idx}/{len(test_numeric_frames)}",
            )
            if args.progress:
                print(f"test 窗口完成 {idx}/{len(test_numeric_frames)}：windows={len(seqs)} ({time.perf_counter()-t0:.2f}s)", flush=True)
            test_seq_parts.append(seqs)
            test_label_parts.append(seq_y.astype(np.uint8))
        test_seqs = np.concatenate(test_seq_parts, axis=0)
        test_labels = np.concatenate(test_label_parts, axis=0)
    else:
        df = load_cicids(args.data_dir)
        features, labels = preprocess_frame(df)
        feature_names = [f"f{i}" for i in range(int(features.shape[1]))]
        t0 = time.perf_counter()
        sequences, seq_labels = build_sequences(
            features,
            labels,
            args.window_size,
            args.stride,
            args.anomaly_ratio,
            progress=bool(args.progress),
            desc="all windows",
        )
        if args.progress:
            print(f"窗口完成：windows={len(sequences)} ({time.perf_counter()-t0:.2f}s)", flush=True)
        train_seqs, test_seqs, test_labels = split_sequences(
            sequences, seq_labels, args.test_fraction, args.seed
        )

    train_loader = DataLoader(
        SequenceDataset(train_seqs), batch_size=args.batch_size, shuffle=True, drop_last=False
    )
    test_loader = DataLoader(
        SequenceDataset(test_seqs, test_labels), batch_size=args.test_batch_size, shuffle=False
    )

    # 如果指定了 --load，优先用 checkpoint 中保存的模型超参来构建网络，避免用户漏传参数导致结构不一致。
    ckpt = None
    ckpt_args = None
    if args.load:
        ckpt = torch.load(args.load, map_location="cpu")
        if isinstance(ckpt, dict):
            ckpt_args = ckpt.get("args")

    window_size = args.window_size
    latent_dim = args.latent_dim
    hidden_channels = args.hidden_channels
    dropout = args.dropout
    disc_pooling = str(args.disc_pooling)
    if isinstance(ckpt_args, dict):
        if "window_size" in ckpt_args and int(ckpt_args["window_size"]) != args.window_size:
            raise ValueError(
                f"checkpoint 的 window_size={ckpt_args['window_size']}，但当前 --window-size={args.window_size}；请改为一致后再加载"
            )
        # 这些参数影响网络形状；优先采用 checkpoint 的值
        latent_dim = int(ckpt_args.get("latent_dim", latent_dim))
        hidden_channels = list(ckpt_args.get("hidden_channels", hidden_channels))
        dropout = float(ckpt_args.get("dropout", dropout))
        disc_pooling = str(ckpt_args.get("disc_pooling", disc_pooling))

    feat_dim = train_seqs.shape[2]
    generator = TCNGenerator(window_size, feat_dim, latent_dim, hidden_channels, dropout).to(device)
    discriminator = TCNDiscriminator(window_size, feat_dim, hidden_channels, dropout, pooling=disc_pooling).to(device)
    g_optimizer = torch.optim.Adam(generator.parameters(), lr=args.lr, betas=(0.5, 0.999))
    d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=args.lr, betas=(0.5, 0.999))
    criterion = nn.BCEWithLogitsLoss()

    if args.load:
        ckpt_weights = torch.load(args.load, map_location=device)
        if isinstance(ckpt_weights, dict) and "generator" in ckpt_weights and "discriminator" in ckpt_weights:
            generator.load_state_dict(ckpt_weights["generator"])
            discriminator.load_state_dict(ckpt_weights["discriminator"])
        elif isinstance(ckpt_weights, dict):
            # 容忍用户保存的是 state_dict
            try:
                discriminator.load_state_dict(ckpt_weights)
            except Exception as e:
                raise ValueError(f"无法从 --load 加载权重：{args.load}") from e
        else:
            raise ValueError(f"--load 文件格式不支持：{args.load}")

    if args.eval_only:
        # Evaluate using chosen score definition (default=prob).
        t_eval0 = time.perf_counter()
        y_true = np.asarray(test_labels, dtype=np.int64)
        x_ref = _sample_rows(train_seqs, args.feat_calib_max_seqs, args.seed + 11) if args.score_mode != "prob" else None
        y_score = compute_anomaly_scores(
            discriminator,
            test_seqs,
            device,
            args.test_batch_size,
            score_mode=args.score_mode,
            score_alpha=args.score_alpha,
            x_ref_benign=x_ref,
        )
        if np.unique(y_true).size < 2:
            auc = float("nan")
        else:
            auc = float(roc_auc_score(y_true, y_score))
        ap = float(average_precision_score(y_true, y_score)) if float(y_true.sum()) > 0.0 else 0.0
        t_eval1 = time.perf_counter()
        print(f"仅评估：AUC {auc:.4f}，AP {ap:.4f}")
        print(f"- Eval 用时：{(t_eval1 - t_eval0):.2f}s | test_windows={len(test_seqs)} | {len(test_seqs)/max(t_eval1-t_eval0,1e-9):.1f} windows/s")
        if str(args.score_mode) != "prob":
            if args.score_mode == "fused":
                print(f"- 评分方式：score_mode=fused(alpha={float(args.score_alpha):.2f})")
            else:
                print(f"- 评分方式：score_mode={args.score_mode}")
        cls = report_classification(y_true, y_score)
        if np.isfinite(cls.get("best_threshold", float("nan"))):
            print(f"- 参考阈值（F1 最大）：{cls['best_threshold']:.6f}")
            print(
                "- 参考指标：Precision {:.4f}，Recall {:.4f}，F1 {:.4f}，FPR {:.4f}".format(
                    cls.get("best_precision", 0.0),
                    cls.get("best_recall", 0.0),
                    cls.get("best_f1", 0.0),
                    cls.get("fpr", 0.0),
                )
            )
        if args.target_fpr is not None:
            calib_seqs = _sample_rows(train_seqs, args.calib_max_seqs, args.seed)
            x_ref2 = (
                _sample_rows(train_seqs, args.feat_calib_max_seqs, args.seed + 11) if args.score_mode != "prob" else None
            )
            benign_scores = compute_anomaly_scores(
                discriminator,
                calib_seqs,
                device,
                args.test_batch_size,
                score_mode=args.score_mode,
                score_alpha=args.score_alpha,
                x_ref_benign=x_ref2,
            )
            th = threshold_from_benign_fpr(benign_scores, args.target_fpr)
            m = metrics_at_threshold(y_true, y_score, th)
            # 训练 BENIGN（用于标定的那批）上的“实际误报率”应接近 target_fpr
            train_benign_fpr = benign_fpr_at_threshold(benign_scores, th)
            # 测试集 BENIGN 子集上的“实际误报率”（按天切分会受分布漂移影响，可能偏离 target_fpr）
            test_benign_scores = np.asarray(y_score, dtype=np.float32)[np.asarray(y_true, dtype=np.int64) == 0]
            test_benign_fpr = benign_fpr_at_threshold(test_benign_scores, th)
            print(
                "- 标定阈值（训练 BENIGN FPR≈{:.3f}）：{:.6f}；Precision {:.4f}，Recall {:.4f}，F1 {:.4f}，FPR {:.4f}（train_BENIGN_FPR {:.4f} | test_BENIGN_FPR {:.4f}）".format(
                    float(args.target_fpr),
                    th,
                    m["precision"],
                    m["recall"],
                    m["f1"],
                    m["fpr"],
                    train_benign_fpr,
                    test_benign_fpr,
                )
            )
        if args.report_path:
            out = pd.DataFrame({"y_true": y_true.astype(int), "y_score": y_score.astype(float)})
            out.to_csv(args.report_path, index=False)
            print(f"已保存测试集打分明细：{args.report_path}")

        xai_report: dict[str, object] | None = None
        if bool(args.xai_report):
            try:
                t_xai0 = time.perf_counter()
                tag = ""
                if str(args.out_json).strip():
                    tag = os.path.splitext(os.path.basename(str(args.out_json)))[0]
                elif str(args.report_path).strip():
                    tag = os.path.splitext(os.path.basename(str(args.report_path)))[0]
                else:
                    tag = "tcn_gan"
                run_tag = f"{tag}_w{int(args.window_size)}_s{int(args.stride)}_{str(args.score_mode)}"
                if str(args.score_mode) == "fused":
                    run_tag += f"_a{float(args.score_alpha):.2f}"
                out_dir = (
                    str(args.xai_out_dir).strip()
                    if str(args.xai_out_dir).strip()
                    else os.path.join("attack", "results", "xai_tcn", run_tag)
                )
                x_ref3 = _sample_rows(
                    train_seqs,
                    args.feat_calib_max_seqs if args.score_mode != "prob" else args.calib_max_seqs,
                    args.seed + 11,
                )
                ref_stats = (
                    _prepare_ref_stats(
                        discriminator,
                        x_ref3,
                        device,
                        batch_size=int(args.xai_batch_size),
                        score_mode=str(args.score_mode),
                    )
                    if args.score_mode != "prob"
                    else None
                )
                xai_report = _xai_gradxinput_report(
                    discriminator,
                    test_seqs,
                    test_labels,
                    device=device,
                    batch_size=int(args.xai_batch_size),
                    score_mode=str(args.score_mode),
                    score_alpha=float(args.score_alpha),
                    ref_stats=ref_stats,
                    feature_names=feature_names,
                    n_samples=int(args.xai_samples),
                    seed=int(args.seed) + 999,
                    out_dir=out_dir,
                )
                t_xai1 = time.perf_counter()
                print(f"已生成 XAI 报告：{xai_report.get('report_json', out_dir)}")
                print(f"- XAI 用时：{(t_xai1 - t_xai0):.2f}s（samples={int(args.xai_samples)} batch={int(args.xai_batch_size)}）")
            except Exception as e:
                print(f"提示：生成 XAI 报告失败：{e}")
        if str(args.out_json).strip():
            try:
                payload: dict[str, object] = {}
                payload["task"] = "window_anomaly_detection"
                payload["data_dir"] = str(args.data_dir)
                payload["train_files"] = list(train_names)
                payload["test_files"] = list(test_names)
                payload["window_size"] = int(args.window_size)
                payload["stride"] = int(args.stride)
                payload["anomaly_ratio"] = float(args.anomaly_ratio)
                payload["train_benign_only"] = bool(args.train_benign_only)
                payload["load"] = str(args.load)
                payload["eval_only"] = True
                payload["disc_pooling"] = str(discriminator.pooling)
                payload["gan_loss"] = str(args.gan_loss)
                payload["gp_lambda"] = float(args.gp_lambda)
                payload["n_critic"] = int(args.n_critic)
                payload["timing"] = {
                    "eval_seconds": float(t_eval1 - t_eval0),
                    "total_seconds": float(time.perf_counter() - t_start),
                    "n_train_windows": int(len(train_seqs)),
                    "n_test_windows": int(len(test_seqs)),
                    "eval_windows_per_sec": float(len(test_seqs) / max(t_eval1 - t_eval0, 1e-9)),
                }
                payload["metrics"] = {
                    "auc": float(auc) if np.isfinite(auc) else None,
                    "ap": float(ap),
                    "best_f1_threshold": float(cls.get("best_threshold", float("nan"))),
                    "best_precision": float(cls.get("best_precision", 0.0)),
                    "best_recall": float(cls.get("best_recall", 0.0)),
                    "best_f1": float(cls.get("best_f1", 0.0)),
                    "best_fpr": float(cls.get("fpr", 0.0)),
                }
                payload["score_mode"] = str(args.score_mode)
                payload["score_alpha"] = float(args.score_alpha)
                if args.target_fpr is not None:
                    # Note: the target is on the calibration benign subset (train benign); test benign FPR may drift.
                    payload["calibrated"] = {
                        "target_fpr": float(args.target_fpr),
                        "threshold": float(th),
                        "precision": float(m["precision"]),
                        "recall": float(m["recall"]),
                        "f1": float(m["f1"]),
                        "fpr": float(m["fpr"]),
                        "train_benign_fpr": float(train_benign_fpr),
                        "test_benign_fpr": float(test_benign_fpr),
                        "n_train_benign_calib": int(len(benign_scores)),
                        "n_test_benign": int(len(test_benign_scores)),
                    }
                payload["scores_csv"] = str(args.report_path) if args.report_path else ""
                if xai_report is not None:
                    payload["xai"] = {
                        "enabled": True,
                        "method": str(xai_report.get("xai_method", "gradxinput")),
                        "report_json": str(xai_report.get("report_json", "")),
                        "plot_time_importance": str(xai_report.get("plot_time_importance", "")),
                        "plot_feature_importance": str(xai_report.get("plot_feature_importance", "")),
                        "top_features_by_anomaly_importance": xai_report.get("top_features_by_anomaly_importance", []),
                    }
                else:
                    payload["xai"] = {"enabled": False}
                os.makedirs(os.path.dirname(str(args.out_json)) or ".", exist_ok=True)
                with open(str(args.out_json), "w", encoding="utf-8") as f:
                    json.dump(payload, f, ensure_ascii=False, indent=2)
                print(f"已保存结果 JSON：{args.out_json}")
            except Exception as e:
                print(f"提示：保存 --out-json 失败：{e}")
        return

    best_auc = 0.0
    best_ap = 0.0
    last_y_true: np.ndarray | None = None
    last_y_score: np.ndarray | None = None
    for epoch in range(1, args.epochs + 1):
        t_train0 = time.perf_counter()
        d_loss, g_loss = train_one_epoch(
            generator,
            discriminator,
            train_loader,
            device,
            g_optimizer,
            d_optimizer,
            criterion,
            args.latent_dim,
            gan_loss=str(args.gan_loss),
            gp_lambda=float(args.gp_lambda),
            n_critic=int(args.n_critic),
            progress=bool(args.progress),
            desc=f"train ep{epoch}/{args.epochs}",
        )
        t_train1 = time.perf_counter()
        t_ev0 = time.perf_counter()
        auc, ap, y_true, y_score = evaluate_model(
            discriminator,
            test_loader,
            device,
            progress=bool(args.progress),
            desc=f"eval ep{epoch}/{args.epochs}",
        )
        t_ev1 = time.perf_counter()
        last_y_true, last_y_score = y_true, y_score
        if auc > best_auc:
            best_auc = auc
            best_ap = ap
            if args.save_best:
                torch.save(
                    {
                        "generator": generator.state_dict(),
                        "discriminator": discriminator.state_dict(),
                        "args": vars(args),
                        "best_auc": best_auc,
                        "best_ap": best_ap,
                    },
                    args.save_best,
                )
        print(
            "轮次 {}/{}：D 损失 {:.4f}，G 损失 {:.4f}，测试 AUC {:.4f}，AP {:.4f}，最佳 AUC {:.4f} | train {:.2f}s eval {:.2f}s".format(
                epoch,
                args.epochs,
                d_loss,
                g_loss,
                auc,
                ap,
                best_auc,
                (t_train1 - t_train0),
                (t_ev1 - t_ev0),
            )
        )
    if args.save_best:
        print(f"已保存最佳模型到：{args.save_best}（最佳 AUC {best_auc:.4f}）")

    if last_y_true is not None and last_y_score is not None:
        print("")
        print("结果解读（测试集）")
        anomaly_rate = float(last_y_true.mean()) if last_y_true.size else 0.0
        print(f"- 异常比例（随机基线 AP）：{anomaly_rate:.4f}")
        print(f"- 最后一次评估：AUC {auc:.4f}，AP {ap:.4f}")
        print(f"- 训练过程中最佳：AUC {best_auc:.4f}，AP {best_ap:.4f}")
        cls = report_classification(last_y_true, last_y_score)
        if np.isfinite(cls.get("best_threshold", float("nan"))):
            print(f"- 参考阈值（F1 最大）：{cls['best_threshold']:.6f}")
            print(
                "- 参考指标：Precision {:.4f}，Recall {:.4f}，F1 {:.4f}，FPR {:.4f}".format(
                    cls.get("best_precision", 0.0),
                    cls.get("best_recall", 0.0),
                    cls.get("best_f1", 0.0),
                    cls.get("fpr", 0.0),
                )
            )
            print(
                "- 混淆矩阵：TN {:.0f}，FP {:.0f}，FN {:.0f}，TP {:.0f}".format(
                    cls.get("tn", 0.0),
                    cls.get("fp", 0.0),
                    cls.get("fn", 0.0),
                    cls.get("tp", 0.0),
                )
            )
        if args.target_fpr is not None:
            calib_seqs = _sample_rows(train_seqs, args.calib_max_seqs, args.seed)
            benign_scores = score_sequences(discriminator, calib_seqs, device, args.test_batch_size)
            th = threshold_from_benign_fpr(benign_scores, args.target_fpr)
            m = metrics_at_threshold(last_y_true, last_y_score, th)
            print(
                "- 标定阈值（训练 BENIGN FPR≈{:.3f}）：{:.6f}；Precision {:.4f}，Recall {:.4f}，F1 {:.4f}，FPR {:.4f}".format(
                    float(args.target_fpr),
                    th,
                    m["precision"],
                    m["recall"],
                    m["f1"],
                    m["fpr"],
                )
            )
        if args.report_path:
            out = pd.DataFrame(
                {"y_true": last_y_true.astype(int), "y_score": last_y_score.astype(float)}
            )
            out.to_csv(args.report_path, index=False)
            print(f"- 已保存测试集打分明细：{args.report_path}")

    # ---- Optional: write a JSON summary even after training ----
    # The training loop above evaluates the basic score (prob = 1 - D(x)) each epoch.
    # For paper-facing results, users often want the final JSON in the *chosen* score_mode
    # (e.g., fused) and with target_fpr calibration and optional XAI, which are implemented
    # in the eval-only branch. Here we run one eval-only style pass (using best checkpoint
    # when available) and save the JSON so the caller can "train once, get JSON once".
    if str(args.out_json).strip():
        try:
            load_path = str(args.save_best).strip() if str(args.save_best).strip() else ""
            if load_path and os.path.exists(load_path):
                ckpt_eval = torch.load(load_path, map_location=device)
                if isinstance(ckpt_eval, dict) and "generator" in ckpt_eval and "discriminator" in ckpt_eval:
                    generator.load_state_dict(ckpt_eval["generator"])
                    discriminator.load_state_dict(ckpt_eval["discriminator"])
                elif isinstance(ckpt_eval, dict):
                    discriminator.load_state_dict(ckpt_eval)

            # Evaluate using chosen score definition (default=prob).
            t_eval0 = time.perf_counter()
            y_true = np.asarray(test_labels, dtype=np.int64)
            x_ref = (
                _sample_rows(train_seqs, args.feat_calib_max_seqs, args.seed + 11)
                if str(args.score_mode).lower().strip() != "prob"
                else None
            )
            y_score = compute_anomaly_scores(
                discriminator,
                test_seqs,
                device,
                args.test_batch_size,
                score_mode=args.score_mode,
                score_alpha=args.score_alpha,
                x_ref_benign=x_ref,
            )
            if np.unique(y_true).size < 2:
                auc = float("nan")
            else:
                auc = float(roc_auc_score(y_true, y_score))
            ap = float(average_precision_score(y_true, y_score)) if float(y_true.sum()) > 0.0 else 0.0
            cls = report_classification(y_true, y_score)
            t_eval1 = time.perf_counter()

            th = float("nan")
            m: dict[str, float] = {"precision": 0.0, "recall": 0.0, "f1": 0.0, "fpr": 0.0}
            train_benign_fpr = float("nan")
            test_benign_fpr = float("nan")
            if args.target_fpr is not None:
                calib_seqs = _sample_rows(train_seqs, args.calib_max_seqs, args.seed)
                x_ref2 = (
                    _sample_rows(train_seqs, args.feat_calib_max_seqs, args.seed + 11)
                    if str(args.score_mode).lower().strip() != "prob"
                    else None
                )
                benign_scores = compute_anomaly_scores(
                    discriminator,
                    calib_seqs,
                    device,
                    args.test_batch_size,
                    score_mode=args.score_mode,
                    score_alpha=args.score_alpha,
                    x_ref_benign=x_ref2,
                )
                th = threshold_from_benign_fpr(benign_scores, args.target_fpr)
                m = metrics_at_threshold(y_true, y_score, th)
                train_benign_fpr = benign_fpr_at_threshold(benign_scores, th)
                test_benign_scores = np.asarray(y_score, dtype=np.float32)[np.asarray(y_true, dtype=np.int64) == 0]
                test_benign_fpr = benign_fpr_at_threshold(test_benign_scores, th)

            # Optional: XAI report
            xai_report: dict[str, object] | None = None
            if bool(args.xai_report):
                try:
                    tag = os.path.splitext(os.path.basename(str(args.out_json)))[0]
                    run_tag = f"{tag}_w{int(args.window_size)}_s{int(args.stride)}_{str(args.score_mode)}"
                    out_dir = (
                        str(args.xai_out_dir).strip()
                        if str(args.xai_out_dir).strip()
                        else os.path.join("attack", "results", "xai_tcn", run_tag)
                    )
                    os.makedirs(out_dir, exist_ok=True)
                    t_xai0 = time.perf_counter()
                    xai_report = run_xai_gradxinput(
                        discriminator=discriminator,
                        sequences=test_seqs,
                        labels=test_labels,
                        device=device,
                        batch_size=int(args.xai_batch_size),
                        score_mode=str(args.score_mode),
                        score_alpha=float(args.score_alpha),
                        ref_stats=ref_stats if "ref_stats" in locals() else None,
                        feature_names=feature_names,
                        n_samples=int(args.xai_samples),
                        seed=int(args.seed) + 999,
                        out_dir=out_dir,
                    )
                    t_xai1 = time.perf_counter()
                    print(f"已生成 XAI 报告：{xai_report.get('report_json', out_dir)}")
                    print(
                        f"- XAI 用时：{(t_xai1 - t_xai0):.2f}s（samples={int(args.xai_samples)} batch={int(args.xai_batch_size)}）"
                    )
                except Exception as e:
                    print(f"提示：生成 XAI 报告失败：{e}")

            payload: dict[str, object] = {}
            payload["task"] = "window_anomaly_detection"
            payload["data_dir"] = str(args.data_dir)
            payload["train_files"] = list(train_names)
            payload["test_files"] = list(test_names)
            payload["window_size"] = int(args.window_size)
            payload["stride"] = int(args.stride)
            payload["anomaly_ratio"] = float(args.anomaly_ratio)
            payload["train_benign_only"] = bool(args.train_benign_only)
            payload["load"] = str(load_path)
            payload["eval_only"] = False
            payload["disc_pooling"] = str(discriminator.pooling)
            payload["gan_loss"] = str(args.gan_loss)
            payload["gp_lambda"] = float(args.gp_lambda)
            payload["n_critic"] = int(args.n_critic)
            payload["timing"] = {
                "eval_seconds": float(t_eval1 - t_eval0),
                "total_seconds": float(time.perf_counter() - t_start),
                "n_train_windows": int(len(train_seqs)),
                "n_test_windows": int(len(test_seqs)),
                "eval_windows_per_sec": float(len(test_seqs) / max(t_eval1 - t_eval0, 1e-9)),
            }
            payload["metrics"] = {
                "auc": float(auc) if np.isfinite(auc) else None,
                "ap": float(ap),
                "best_f1_threshold": float(cls.get("best_threshold", float("nan"))),
                "best_precision": float(cls.get("best_precision", 0.0)),
                "best_recall": float(cls.get("best_recall", 0.0)),
                "best_f1": float(cls.get("best_f1", 0.0)),
                "best_fpr": float(cls.get("fpr", 0.0)),
            }
            payload["score_mode"] = str(args.score_mode)
            payload["score_alpha"] = float(args.score_alpha)
            if args.target_fpr is not None:
                payload["calibrated"] = {
                    "target_fpr": float(args.target_fpr),
                    "threshold": float(th),
                    "precision": float(m["precision"]),
                    "recall": float(m["recall"]),
                    "f1": float(m["f1"]),
                    "fpr": float(m["fpr"]),
                    "train_benign_fpr": float(train_benign_fpr),
                    "test_benign_fpr": float(test_benign_fpr),
                }
            if xai_report is not None:
                payload["xai"] = {
                    "enabled": True,
                    "method": str(xai_report.get("xai_method", "gradxinput")),
                    "report_json": str(xai_report.get("report_json", "")),
                    "plot_time_importance": str(xai_report.get("plot_time_importance", "")),
                    "plot_feature_importance": str(xai_report.get("plot_feature_importance", "")),
                    "top_features_by_anomaly_importance": xai_report.get("top_features_by_anomaly_importance", []),
                }
            else:
                payload["xai"] = {"enabled": False}
            os.makedirs(os.path.dirname(str(args.out_json)) or ".", exist_ok=True)
            with open(str(args.out_json), "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
            print(f"已保存结果 JSON：{args.out_json}")
        except Exception as e:
            print(f"提示：训练后保存 --out-json 失败：{e}")


if __name__ == "__main__":
    main()
