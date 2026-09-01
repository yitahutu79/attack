#!/usr/bin/env python3
"""
窗口级baseline对比实验
包括：IsolationForest, OneClassSVM, RF, SVM, MLP, LSTM-AE, LSTM-AD
与TCN-GAN进行同口径对比
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from datetime import datetime
from dataclasses import dataclass, asdict
from pathlib import Path

# Make `attack/` importable
_REPO_ROOT = Path(__file__).resolve().parents[1]
_REPO_ROOT_S = str(_REPO_ROOT)
if _REPO_ROOT_S not in sys.path:
    sys.path.insert(0, _REPO_ROOT_S)

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.svm import OneClassSVM, SVC
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve

from dataset_loaders import (
    list_dataset_files,
    load_windowed_chrono_unsupervised_split,
    load_windowed_dataset,
    load_windowed_mixed_split,
    load_windowed_unsupervised_split,
    normalize_dataset_name,
)


@dataclass
class BaselineResult:
    """Baseline实验结果"""
    method: str
    window_size: int
    stride: int
    auc: float
    ap: float
    target_fpr: float
    calib_f1: float
    calib_recall: float
    calib_precision: float
    calib_threshold: float
    train_benign_fpr: float
    test_benign_fpr: float
    eval_seconds: float
    n_train_windows: int
    n_test_windows: int


def parse_target_fprs(target_fpr: float, target_fpr_grid: str) -> list[float]:
    """解析一个或多个目标误报率，保留顺序并去重。"""
    values: list[float] = [float(target_fpr)]
    for token in str(target_fpr_grid).split(","):
        token = token.strip()
        if not token:
            continue
        values.append(float(token))

    out: list[float] = []
    seen: set[float] = set()
    for value in values:
        if not (0.0 < value < 1.0):
            raise ValueError("target_fpr 必须在 (0,1) 范围内，例如 0.05")
        key = round(float(value), 10)
        if key in seen:
            continue
        seen.add(key)
        out.append(float(value))
    return out


def skipped_results(
    method: str,
    window_size: int,
    stride: int,
    target_fpr: float | list[float],
    reason: str,
    n_train_windows: int,
    n_test_windows: int,
) -> list[BaselineResult]:
    """Create placeholder rows when a baseline is not applicable to the split."""
    print(f"跳过 {method}: {reason}")
    values = target_fpr if isinstance(target_fpr, list) else [float(target_fpr)]
    return [
        BaselineResult(
            method=f"{method} [skipped: {reason}]",
            window_size=window_size,
            stride=stride,
            auc=float("nan"),
            ap=float("nan"),
            target_fpr=float(v),
            calib_f1=float("nan"),
            calib_recall=float("nan"),
            calib_precision=float("nan"),
            calib_threshold=float("nan"),
            train_benign_fpr=float("nan"),
            test_benign_fpr=float("nan"),
            eval_seconds=0.0,
            n_train_windows=int(n_train_windows),
            n_test_windows=int(n_test_windows),
        )
        for v in values
    ]


# ============================================================================
# 数据加载与窗口化（复用TCN-GAN的逻辑）
# ============================================================================

def load_cicids_windowed(
    data_dir: str,
    train_files: list[str],
    test_files: list[str],
    window_size: int,
    stride: int,
    anomaly_ratio: float = 0.15,
    train_benign_only: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    加载CICIDS2017数据并进行窗口化

    Returns:
        X_train: (n_train, window_size, n_features)
        y_train: (n_train,)
        X_test: (n_test, window_size, n_features)
        y_test: (n_test,)
    """
    print(f"加载数据: {data_dir}")
    print(f"训练文件: {train_files}")
    print(f"测试文件: {test_files}")

    # 加载训练数据
    train_dfs = []
    for fname in train_files:
        fpath = os.path.join(data_dir, fname)
        if not os.path.exists(fpath):
            print(f"警告: 文件不存在 {fpath}")
            continue
        df = pd.read_csv(fpath, skipinitialspace=True, low_memory=False)
        train_dfs.append(df)
        print(f"  加载 {fname}: {len(df)} 行")

    train_df = pd.concat(train_dfs, ignore_index=True)
    print(f"训练集总行数: {len(train_df)}")

    # 加载测试数据
    test_dfs = []
    for fname in test_files:
        fpath = os.path.join(data_dir, fname)
        if not os.path.exists(fpath):
            print(f"警告: 文件不存在 {fpath}")
            continue
        df = pd.read_csv(fpath, skipinitialspace=True, low_memory=False)
        test_dfs.append(df)
        print(f"  加载 {fname}: {len(df)} 行")

    test_df = pd.concat(test_dfs, ignore_index=True)
    print(f"测试集总行数: {len(test_df)}")

    # 处理标签
    def process_labels(df):
        """将标签转换为二分类：0=正常，1=攻击"""
        if 'Label' in df.columns:
            labels = df['Label'].values
        elif 'label' in df.columns:
            labels = df['label'].values
        else:
            raise ValueError("找不到标签列")

        # 转换为小写字符串
        labels = np.array([str(l).lower().strip() for l in labels])

        # BENIGN = 0, 其他 = 1
        y = (labels != 'benign').astype(int)
        return y

    y_train_raw = process_labels(train_df)
    y_test_raw = process_labels(test_df)

    print(f"训练集攻击比例: {y_train_raw.mean():.4f}")
    print(f"测试集攻击比例: {y_test_raw.mean():.4f}")

    # 提取特征
    feature_cols = [c for c in train_df.columns
                    if c not in ['Label', 'label', 'Flow ID', 'Source IP', 'Destination IP',
                                'Source Port', 'Destination Port', 'Timestamp']]

    # 只保留数值特征
    numeric_cols = []
    for c in feature_cols:
        try:
            train_df[c].astype(float)
            numeric_cols.append(c)
        except:
            pass

    print(f"特征数量: {len(numeric_cols)}")

    # 提取特征矩阵
    X_train_raw = train_df[numeric_cols].values.astype(np.float32)
    X_test_raw = test_df[numeric_cols].values.astype(np.float32)

    # 处理无穷大和NaN
    X_train_raw = np.nan_to_num(X_train_raw, nan=0.0, posinf=0.0, neginf=0.0)
    X_test_raw = np.nan_to_num(X_test_raw, nan=0.0, posinf=0.0, neginf=0.0)

    # 标准化
    scaler = StandardScaler()
    X_train_raw = scaler.fit_transform(X_train_raw)
    X_test_raw = scaler.transform(X_test_raw)

    # 窗口化
    def create_windows(X, y, window_size, stride, anomaly_ratio, benign_only=False):
        """创建窗口样本"""
        n_samples = X.shape[0]
        n_windows = (n_samples - window_size) // stride + 1

        windows = []
        labels = []

        for i in range(n_windows):
            start = i * stride
            end = start + window_size

            if end > n_samples:
                break

            # 窗口特征
            window = X[start:end]
            windows.append(window)

            # 窗口标签：如果窗口内攻击比例 >= anomaly_ratio，则标记为异常
            window_labels = y[start:end]
            attack_ratio = window_labels.mean()
            label = 1 if attack_ratio >= anomaly_ratio else 0
            labels.append(label)

        windows = np.array(windows)
        labels = np.array(labels)

        if benign_only:
            # 只保留正常窗口
            benign_idx = labels == 0
            windows = windows[benign_idx]
            labels = labels[benign_idx]

        return windows, labels

    print(f"\n创建窗口 (window_size={window_size}, stride={stride})...")
    X_train, y_train = create_windows(
        X_train_raw, y_train_raw, window_size, stride, anomaly_ratio, train_benign_only
    )
    X_test, y_test = create_windows(
        X_test_raw, y_test_raw, window_size, stride, anomaly_ratio, False
    )

    print(f"训练窗口数: {len(X_train)} (benign_only={train_benign_only})")
    print(f"测试窗口数: {len(X_test)}")
    print(f"测试集异常比例: {y_test.mean():.4f}")

    return X_train, y_train, X_test, y_test


# ============================================================================
# 传统方法：IsolationForest, OneClassSVM
# ============================================================================

def run_iforest_baseline(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    window_size: int,
    stride: int,
    contamination: float = 0.1,
    target_fpr: float = 0.05,
    X_calib: np.ndarray | None = None,
) -> list[BaselineResult]:
    """
    IsolationForest baseline
    """
    print("\n" + "="*60)
    print("运行 IsolationForest baseline")
    print("="*60)

    start_time = time.time()

    # 展平窗口
    n_train, w, f = X_train.shape
    n_test = X_test.shape[0]

    X_train_flat = X_train.reshape(n_train, w * f)
    X_test_flat = X_test.reshape(n_test, w * f)

    print(f"训练集形状: {X_train_flat.shape}")
    print(f"测试集形状: {X_test_flat.shape}")

    # 训练模型
    print("训练 IsolationForest...")
    model = IsolationForest(
        contamination=contamination,
        random_state=42,
        n_jobs=-1,
        n_estimators=100,
    )
    model.fit(X_train_flat)

    # 预测（异常分数：越小越异常）
    print("预测...")
    scores_train = -model.decision_function(X_train_flat)  # 转换为越大越异常
    scores_test = -model.decision_function(X_test_flat)
    scores_calib = None
    y_calib = None
    if X_calib is not None and len(X_calib):
        X_calib_flat = X_calib.reshape(X_calib.shape[0], w * f)
        scores_calib = -model.decision_function(X_calib_flat)
        y_calib = np.zeros(len(X_calib_flat))

    # 计算指标
    result = compute_metrics(
        scores_train, np.zeros(n_train),  # 训练集全是正常
        scores_test, y_test,
        window_size, stride,
        method="IsolationForest",
        target_fpr=target_fpr,
        start_time=start_time,
        n_train_windows=n_train,
        n_test_windows=n_test,
        scores_calib=scores_calib,
        y_calib=y_calib,
    )

    return result


def run_ocsvm_baseline(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    window_size: int,
    stride: int,
    nu: float = 0.1,
    target_fpr: float = 0.05,
    max_samples: int = 10000,
    use_pca: bool = True,
    pca_components: int = 100,
    X_calib: np.ndarray | None = None,
) -> list[BaselineResult]:
    """
    OneClassSVM baseline (优化版: 采样+降维)
    """
    print("\n" + "="*60)
    print("运行 OneClassSVM baseline (优化版)")
    print("="*60)

    start_time = time.time()

    # 展平窗口
    n_train, w, f = X_train.shape
    n_test = X_test.shape[0]

    X_train_flat = X_train.reshape(n_train, w * f)
    X_test_flat = X_test.reshape(n_test, w * f)

    print(f"原始训练集形状: {X_train_flat.shape}")
    print(f"原始测试集形状: {X_test_flat.shape}")

    # 1. 采样训练集
    if n_train > max_samples:
        print(f"采样训练集: {n_train} -> {max_samples}")
        sample_idx = np.random.choice(n_train, max_samples, replace=False)
        X_train_sampled = X_train_flat[sample_idx]
    else:
        X_train_sampled = X_train_flat

    # 2. PCA降维
    if use_pca and X_train_sampled.shape[1] > pca_components:
        print(f"PCA降维: {X_train_sampled.shape[1]} -> {pca_components}")
        from sklearn.decomposition import PCA
        pca = PCA(n_components=pca_components, random_state=42)
        X_train_reduced = pca.fit_transform(X_train_sampled)
        X_test_reduced = pca.transform(X_test_flat)
        X_calib_reduced = pca.transform(X_calib.reshape(X_calib.shape[0], w * f)) if X_calib is not None and len(X_calib) else None
        print(f"PCA解释方差比: {pca.explained_variance_ratio_.sum():.4f}")
    else:
        X_train_reduced = X_train_sampled
        X_test_reduced = X_test_flat
        X_calib_reduced = X_calib.reshape(X_calib.shape[0], w * f) if X_calib is not None and len(X_calib) else None

    print(f"最终训练集形状: {X_train_reduced.shape}")
    print(f"最终测试集形状: {X_test_reduced.shape}")

    # 训练模型
    print("训练 OneClassSVM...")
    model = OneClassSVM(
        nu=nu,
        kernel='rbf',
        gamma='scale',
    )
    model.fit(X_train_reduced)

    # 预测（异常分数：越小越异常）
    print("预测...")
    scores_train = -model.decision_function(X_train_reduced)
    scores_test = -model.decision_function(X_test_reduced)
    scores_calib = -model.decision_function(X_calib_reduced) if X_calib_reduced is not None else None
    y_calib = np.zeros(len(scores_calib)) if scores_calib is not None else None

    # 计算指标
    result = compute_metrics(
        scores_train, np.zeros(n_train),
        scores_test, y_test,
        window_size, stride,
        method="OneClassSVM",
        target_fpr=target_fpr,
        start_time=start_time,
        n_train_windows=n_train,
        n_test_windows=n_test,
        scores_calib=scores_calib,
        y_calib=y_calib,
    )

    return result


# ============================================================================
# 深度学习方法：LSTM-AE, LSTM-AD
# ============================================================================

class LSTMAutoencoder(nn.Module):
    """LSTM自编码器"""

    def __init__(self, input_dim, hidden_dim=64, num_layers=2):
        super().__init__()

        self.encoder = nn.LSTM(
            input_dim, hidden_dim, num_layers,
            batch_first=True, dropout=0.2
        )

        self.decoder = nn.LSTM(
            hidden_dim, hidden_dim, num_layers,
            batch_first=True, dropout=0.2
        )

        self.output_layer = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        # Encode
        encoded, (h_n, c_n) = self.encoder(x)

        # Decode
        decoded, _ = self.decoder(encoded, (h_n, c_n))

        # Output
        output = self.output_layer(decoded)

        return output

    def get_reconstruction_error(self, x, batch_size=1000):
        """计算重构误差(分批处理避免内存溢出)"""
        errors = []
        n_samples = x.shape[0]

        with torch.no_grad():
            for i in range(0, n_samples, batch_size):
                batch = x[i:i+batch_size]
                x_recon = self.forward(batch)
                # MSE重构误差
                error = torch.mean((batch - x_recon) ** 2, dim=(1, 2))
                errors.append(error.cpu().numpy())

        return np.concatenate(errors)


class LSTMAnomalyDetector(nn.Module):
    """LSTM预测式异常检测"""

    def __init__(self, input_dim, hidden_dim=64, num_layers=2):
        super().__init__()

        self.lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers,
            batch_first=True, dropout=0.2
        )

        self.output_layer = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        # 用前n-1步预测第n步
        x_input = x[:, :-1, :]
        lstm_out, (h_n, c_n) = self.lstm(x_input)
        prediction = self.output_layer(h_n[-1])
        return prediction

    def get_prediction_error(self, x, batch_size=1000):
        """计算预测误差(分批处理避免内存溢出)"""
        errors = []
        n_samples = x.shape[0]

        with torch.no_grad():
            for i in range(0, n_samples, batch_size):
                batch = x[i:i+batch_size]
                x_true = batch[:, -1, :]
                x_pred = self.forward(batch)
                error = torch.mean((x_true - x_pred) ** 2, dim=1)
                errors.append(error.cpu().numpy())

        return np.concatenate(errors)


def train_lstm_model(
    model: nn.Module,
    X_train: np.ndarray,
    model_type: str = 'ae',
    epochs: int = 50,
    batch_size: int = 128,
    lr: float = 1e-3,
    device: str = 'cpu',
):
    """训练LSTM模型（带进度条）"""
    if str(device).lower() == 'cpu':
        # macOS + PyTorch CPU can stall badly with default thread counts on small LSTM baselines.
        try:
            torch.set_num_threads(1)
        except Exception:
            pass
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            # Safe to ignore once parallel work has already started in the process.
            pass

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    X_train = np.asarray(X_train, dtype=np.float32)
    n_samples = X_train.shape[0]

    model.train()
    
    # 使用tqdm显示训练进度
    epoch_pbar = tqdm(range(epochs), desc=f"Training {model_type.upper()}")
    
    for epoch in epoch_pbar:
        # Shuffle on numpy side to avoid large device-side permutations that may hang on macOS.
        perm = np.random.permutation(n_samples)

        total_loss = 0.0
        n_batches = 0

        for i in range(0, n_samples, batch_size):
            batch_np = X_train[perm[i:i+batch_size]]
            batch = torch.from_numpy(batch_np).to(device)

            optimizer.zero_grad()

            if model_type == 'ae':
                # 自编码器：重构损失
                output = model(batch)
                loss = torch.mean((batch - output) ** 2)
            else:
                # 预测式：预测损失
                x_true = batch[:, -1, :]
                x_pred = model(batch)
                loss = torch.mean((x_true - x_pred) ** 2)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / n_batches
        epoch_pbar.set_postfix({'loss': f'{avg_loss:.6f}'})
        if epoch == 0 or (epoch + 1) % 5 == 0 or epoch + 1 == epochs:
            print(f"LSTM-{model_type.upper()} epoch {epoch + 1}/{epochs}: loss={avg_loss:.6f}", flush=True)

    model.eval()
    return model


def run_lstm_ae_baseline(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    window_size: int,
    stride: int,
    hidden_dim: int = 64,
    num_layers: int = 2,
    epochs: int = 50,
    batch_size: int = 128,
    lr: float = 1e-3,
    target_fpr: float = 0.05,
    device: str = 'cpu',
    X_calib: np.ndarray | None = None,
) -> list[BaselineResult]:
    """
    LSTM-AE baseline
    """
    print("\n" + "="*60)
    print("运行 LSTM-AE baseline")
    print("="*60)

    start_time = time.time()

    n_train, w, f = X_train.shape
    n_test = X_test.shape[0]

    print(f"训练集形状: {X_train.shape}")
    print(f"测试集形状: {X_test.shape}")
    print(f"模型参数: hidden_dim={hidden_dim}, num_layers={num_layers}")

    # 创建模型
    model = LSTMAutoencoder(f, hidden_dim, num_layers)

    # 训练
    print("训练 LSTM-AE...")
    model = train_lstm_model(
        model, X_train, model_type='ae',
        epochs=epochs, batch_size=batch_size, lr=lr, device=device
    )

    # 预测
    print("预测...")
    X_train_tensor = torch.FloatTensor(X_train).to(device)
    X_test_tensor = torch.FloatTensor(X_test).to(device)

    scores_train = model.get_reconstruction_error(X_train_tensor)
    scores_test = model.get_reconstruction_error(X_test_tensor)
    scores_calib = None
    y_calib = None
    if X_calib is not None and len(X_calib):
        X_calib_tensor = torch.FloatTensor(X_calib).to(device)
        scores_calib = model.get_reconstruction_error(X_calib_tensor)
        y_calib = np.zeros(len(scores_calib))

    # 计算指标
    result = compute_metrics(
        scores_train, np.zeros(n_train),
        scores_test, y_test,
        window_size, stride,
        method="LSTM-AE",
        target_fpr=target_fpr,
        start_time=start_time,
        n_train_windows=n_train,
        n_test_windows=n_test,
        scores_calib=scores_calib,
        y_calib=y_calib,
    )

    return result


def run_lstm_ad_baseline(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    window_size: int,
    stride: int,
    hidden_dim: int = 64,
    num_layers: int = 2,
    epochs: int = 50,
    batch_size: int = 128,
    lr: float = 1e-3,
    target_fpr: float = 0.05,
    device: str = 'cpu',
    X_calib: np.ndarray | None = None,
) -> list[BaselineResult]:
    """
    LSTM-AD baseline
    """
    print("\n" + "="*60)
    print("运行 LSTM-AD baseline")
    print("="*60)

    start_time = time.time()

    n_train, w, f = X_train.shape
    n_test = X_test.shape[0]

    print(f"训练集形状: {X_train.shape}")
    print(f"测试集形状: {X_test.shape}")
    print(f"模型参数: hidden_dim={hidden_dim}, num_layers={num_layers}")

    # 创建模型
    model = LSTMAnomalyDetector(f, hidden_dim, num_layers)

    # 训练
    print("训练 LSTM-AD...")
    model = train_lstm_model(
        model, X_train, model_type='ad',
        epochs=epochs, batch_size=batch_size, lr=lr, device=device
    )

    # 预测
    print("预测...")
    X_train_tensor = torch.FloatTensor(X_train).to(device)
    X_test_tensor = torch.FloatTensor(X_test).to(device)

    scores_train = model.get_prediction_error(X_train_tensor)
    scores_test = model.get_prediction_error(X_test_tensor)
    scores_calib = None
    y_calib = None
    if X_calib is not None and len(X_calib):
        X_calib_tensor = torch.FloatTensor(X_calib).to(device)
        scores_calib = model.get_prediction_error(X_calib_tensor)
        y_calib = np.zeros(len(scores_calib))

    # 计算指标
    result = compute_metrics(
        scores_train, np.zeros(n_train),
        scores_test, y_test,
        window_size, stride,
        method="LSTM-AD",
        target_fpr=target_fpr,
        start_time=start_time,
        n_train_windows=n_train,
        n_test_windows=n_test,
        scores_calib=scores_calib,
        y_calib=y_calib,
    )

    return result


# ============================================================================
# 深度异常检测：DeepSVDD
# ============================================================================

class DeepSVDDNet(nn.Module):
    """A lightweight DeepSVDD encoder on flattened windows."""

    def __init__(self, input_dim: int, rep_dim: int = 32):
        super().__init__()
        hidden1 = min(512, max(64, input_dim // 4))
        hidden2 = min(128, max(32, hidden1 // 2))
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden1),
            nn.ReLU(),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Linear(hidden2, rep_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def train_deepsvdd_model(
    model: DeepSVDDNet,
    X_train_flat: np.ndarray,
    epochs: int = 30,
    batch_size: int = 256,
    lr: float = 1e-3,
    device: str = 'cpu',
) -> tuple[DeepSVDDNet, torch.Tensor]:
    """Train DeepSVDD and return the fitted center."""
    if str(device).lower() == 'cpu':
        try:
            torch.set_num_threads(1)
        except Exception:
            pass
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            pass

    model = model.to(device)
    X_train_flat = np.asarray(X_train_flat, dtype=np.float32)
    n_samples = X_train_flat.shape[0]

    with torch.no_grad():
        chunks = []
        for i in range(0, n_samples, batch_size):
            batch = torch.from_numpy(X_train_flat[i:i+batch_size]).to(device)
            chunks.append(model(batch))
        center = torch.cat(chunks, dim=0).mean(dim=0)
        center[(center.abs() < 1e-6)] = 1e-6

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    epoch_pbar = tqdm(range(epochs), desc="Training DeepSVDD")
    model.train()
    for epoch in epoch_pbar:
        perm = np.random.permutation(n_samples)
        total_loss = 0.0
        n_batches = 0
        for i in range(0, n_samples, batch_size):
            batch_np = X_train_flat[perm[i:i+batch_size]]
            batch = torch.from_numpy(batch_np).to(device)
            optimizer.zero_grad()
            z = model(batch)
            loss = torch.mean(torch.sum((z - center) ** 2, dim=1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / max(n_batches, 1)
        epoch_pbar.set_postfix({'loss': f'{avg_loss:.6f}'})
        if epoch == 0 or (epoch + 1) % 5 == 0 or epoch + 1 == epochs:
            print(f"DeepSVDD epoch {epoch + 1}/{epochs}: loss={avg_loss:.6f}", flush=True)

    model.eval()
    return model, center.detach()


def compute_deepsvdd_scores(
    model: DeepSVDDNet,
    center: torch.Tensor,
    X_flat: np.ndarray,
    batch_size: int = 512,
    device: str = 'cpu',
) -> np.ndarray:
    scores: list[np.ndarray] = []
    X_flat = np.asarray(X_flat, dtype=np.float32)
    with torch.no_grad():
        for i in range(0, len(X_flat), batch_size):
            batch = torch.from_numpy(X_flat[i:i+batch_size]).to(device)
            z = model(batch)
            dist = torch.sum((z - center) ** 2, dim=1)
            scores.append(dist.cpu().numpy())
    return np.concatenate(scores, axis=0)


def run_deepsvdd_baseline(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    window_size: int,
    stride: int,
    rep_dim: int = 32,
    epochs: int = 30,
    batch_size: int = 256,
    lr: float = 1e-3,
    target_fpr: float = 0.05,
    device: str = 'cpu',
    X_calib: np.ndarray | None = None,
) -> list[BaselineResult]:
    """DeepSVDD baseline on flattened windows."""
    print("\n" + "="*60)
    print("运行 DeepSVDD baseline")
    print("="*60)

    start_time = time.time()
    n_train, w, f = X_train.shape
    n_test = X_test.shape[0]

    X_train_flat = X_train.reshape(n_train, w * f)
    X_test_flat = X_test.reshape(n_test, w * f)
    print(f"训练集形状: {X_train_flat.shape}")
    print(f"测试集形状: {X_test_flat.shape}")
    print(f"模型参数: rep_dim={rep_dim}")

    model = DeepSVDDNet(X_train_flat.shape[1], rep_dim=rep_dim)
    print("训练 DeepSVDD...")
    model, center = train_deepsvdd_model(
        model,
        X_train_flat,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        device=device,
    )

    print("预测...")
    scores_train = compute_deepsvdd_scores(model, center, X_train_flat, batch_size=batch_size, device=device)
    scores_test = compute_deepsvdd_scores(model, center, X_test_flat, batch_size=batch_size, device=device)
    scores_calib = None
    y_calib = None
    if X_calib is not None and len(X_calib):
        X_calib_flat = X_calib.reshape(X_calib.shape[0], w * f)
        scores_calib = compute_deepsvdd_scores(model, center, X_calib_flat, batch_size=batch_size, device=device)
        y_calib = np.zeros(len(scores_calib))

    result = compute_metrics(
        scores_train, np.zeros(n_train),
        scores_test, y_test,
        window_size, stride,
        method="DeepSVDD",
        target_fpr=target_fpr,
        start_time=start_time,
        n_train_windows=n_train,
        n_test_windows=n_test,
        scores_calib=scores_calib,
        y_calib=y_calib,
    )
    return result


# ============================================================================
# 时序异常检测：TranAD (minimal reproducible baseline)
# ============================================================================

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class MinimalTranAD(nn.Module):
    """A compact TranAD-style reconstruction model for window-level AD."""

    def __init__(self, input_dim: int, d_model: int = 64, nhead: int = 4, num_layers: int = 2):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos = PositionalEncoding(d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 2,
            dropout=0.1,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.decoder = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, input_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.input_proj(x)
        z = self.pos(z)
        z = self.encoder(z)
        return self.decoder(z)

    def get_reconstruction_error(self, x: torch.Tensor, batch_size: int = 512) -> np.ndarray:
        errors = []
        n_samples = x.shape[0]
        with torch.no_grad():
            for i in range(0, n_samples, batch_size):
                batch = x[i:i+batch_size]
                recon = self.forward(batch)
                err = torch.mean((batch - recon) ** 2, dim=(1, 2))
                errors.append(err.cpu().numpy())
        return np.concatenate(errors)


def train_tranad_model(
    model: MinimalTranAD,
    X_train: np.ndarray,
    epochs: int = 30,
    batch_size: int = 128,
    lr: float = 1e-3,
    device: str = 'cpu',
) -> MinimalTranAD:
    if str(device).lower() == 'cpu':
        try:
            torch.set_num_threads(1)
        except Exception:
            pass
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            pass

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    X_train = np.asarray(X_train, dtype=np.float32)
    n_samples = X_train.shape[0]

    model.train()
    epoch_pbar = tqdm(range(epochs), desc="Training TranAD")
    for epoch in epoch_pbar:
        perm = np.random.permutation(n_samples)
        total_loss = 0.0
        n_batches = 0
        for i in range(0, n_samples, batch_size):
            batch_np = X_train[perm[i:i+batch_size]]
            batch = torch.from_numpy(batch_np).to(device)
            optimizer.zero_grad()
            recon = model(batch)
            loss = torch.mean((batch - recon) ** 2)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1
        avg_loss = total_loss / max(n_batches, 1)
        epoch_pbar.set_postfix({'loss': f'{avg_loss:.6f}'})
        if epoch == 0 or (epoch + 1) % 5 == 0 or epoch + 1 == epochs:
            print(f"TranAD epoch {epoch + 1}/{epochs}: loss={avg_loss:.6f}", flush=True)

    model.eval()
    return model


def run_tranad_baseline(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    window_size: int,
    stride: int,
    d_model: int = 64,
    nhead: int = 4,
    num_layers: int = 2,
    epochs: int = 30,
    batch_size: int = 128,
    lr: float = 1e-3,
    target_fpr: float = 0.05,
    device: str = 'cpu',
    X_calib: np.ndarray | None = None,
) -> list[BaselineResult]:
    print("\n" + "="*60)
    print("运行 TranAD baseline")
    print("="*60)

    start_time = time.time()
    n_train, _, f = X_train.shape
    n_test = X_test.shape[0]

    print(f"训练集形状: {X_train.shape}")
    print(f"测试集形状: {X_test.shape}")
    print(f"模型参数: d_model={d_model}, nhead={nhead}, num_layers={num_layers}")

    model = MinimalTranAD(f, d_model=d_model, nhead=nhead, num_layers=num_layers)
    print("训练 TranAD...")
    model = train_tranad_model(
        model,
        X_train,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        device=device,
    )

    print("预测...")
    X_train_tensor = torch.FloatTensor(X_train).to(device)
    X_test_tensor = torch.FloatTensor(X_test).to(device)
    scores_train = model.get_reconstruction_error(X_train_tensor)
    scores_test = model.get_reconstruction_error(X_test_tensor)
    scores_calib = None
    y_calib = None
    if X_calib is not None and len(X_calib):
        X_calib_tensor = torch.FloatTensor(X_calib).to(device)
        scores_calib = model.get_reconstruction_error(X_calib_tensor)
        y_calib = np.zeros(len(scores_calib))

    result = compute_metrics(
        scores_train, np.zeros(n_train),
        scores_test, y_test,
        window_size, stride,
        method="TranAD",
        target_fpr=target_fpr,
        start_time=start_time,
        n_train_windows=n_train,
        n_test_windows=n_test,
        scores_calib=scores_calib,
        y_calib=y_calib,
    )
    return result


# ============================================================================
# GAN-based SOTA: GANomaly
# ============================================================================

class GANomalyNet(nn.Module):
    """
    GANomaly architecture: Encoder -> Decoder -> Encoder
    """
    def __init__(self, input_dim: int, latent_dim: int = 64):
        super().__init__()
        # Encoder 1
        self.encoder1 = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, latent_dim)
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, input_dim),
            nn.Sigmoid()
        )
        # Encoder 2
        self.encoder2 = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LeakyReLU(0.2),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, latent_dim)
        )

    def forward(self, x):
        z = self.encoder1(x)
        x_prime = self.decoder(z)
        z_prime = self.encoder2(x_prime)
        return x_prime, z, z_prime


def train_ganomaly_model(
    model: GANomalyNet,
    X_train_flat: np.ndarray,
    epochs: int = 30,
    batch_size: int = 128,
    lr: float = 1e-3,
    device: str = 'cpu',
):
    # On macOS/conda environments, large dense torch ops can segfault when
    # multiple BLAS/OpenMP threads are spawned. Keep this baseline conservative.
    try:
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
    except RuntimeError:
        # set_num_interop_threads can only be called before parallel work starts.
        pass

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    X_train_flat = np.asarray(X_train_flat, dtype=np.float32)
    n_samples = X_train_flat.shape[0]

    model.train()
    epoch_pbar = tqdm(range(epochs), desc="Training GANomaly", file=sys.stderr)
    for epoch in epoch_pbar:
        perm = np.random.permutation(n_samples)
        total_loss = 0.0
        n_batches = 0
        for i in range(0, n_samples, batch_size):
            batch_np = X_train_flat[perm[i:i+batch_size]]
            batch = torch.from_numpy(batch_np).to(device)

            optimizer.zero_grad()
            x_prime, z, z_prime = model(batch)

            # Losses
            l_con = torch.mean(torch.abs(batch - x_prime)) # Context loss
            l_lat = torch.mean((z - z_prime)**2)           # Latent loss
            # Simplified GANomaly: we skip the discriminator part for a faster baseline
            # as the encoder-decoder-encoder reconstruction is the core of its anomaly scoring.
            loss = 50.0 * l_con + 1.0 * l_lat

            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / max(n_batches, 1)
        epoch_pbar.set_postfix({'loss': f'{avg_loss:.6f}'})

    model.eval()
    return model


def run_ganomaly_baseline(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    window_size: int,
    stride: int,
    latent_dim: int = 64,
    epochs: int = 30,
    batch_size: int = 128,
    lr: float = 1e-3,
    target_fpr: float = 0.05,
    device: str = 'cpu',
    X_calib: np.ndarray | None = None,
) -> list[BaselineResult]:
    print("\n" + "="*60)
    print("运行 GANomaly baseline")
    print("="*60)

    start_time = time.time()
    n_train, w, f = X_train.shape
    n_test = X_test.shape[0]

    X_train_flat = X_train.reshape(n_train, w * f)
    X_test_flat = X_test.reshape(n_test, w * f)

    model = GANomalyNet(w * f, latent_dim=latent_dim)
    print("训练 GANomaly...")
    model = train_ganomaly_model(
        model, X_train_flat, epochs=epochs, batch_size=batch_size, lr=lr, device=device
    )

    print("预测...")
    def get_scores(X_flat, score_batch_size: int = 4096):
        X_flat = np.asarray(X_flat, dtype=np.float32)
        chunks: list[np.ndarray] = []
        with torch.no_grad():
            for start in range(0, X_flat.shape[0], score_batch_size):
                batch = torch.from_numpy(X_flat[start:start + score_batch_size]).to(device)
                _, z, z_prime = model(batch)
                # Anomaly score is the latent reconstruction error.
                score = torch.mean((z - z_prime)**2, dim=1)
                chunks.append(score.cpu().numpy())
        return np.concatenate(chunks, axis=0)

    scores_train = get_scores(X_train_flat)
    scores_test = get_scores(X_test_flat)
    scores_calib = get_scores(X_calib.reshape(X_calib.shape[0], w * f)) if X_calib is not None and len(X_calib) else None
    y_calib = np.zeros(len(scores_calib)) if scores_calib is not None else None

    result = compute_metrics(
        scores_train, np.zeros(n_train),
        scores_test, y_test,
        window_size, stride,
        method="GANomaly",
        target_fpr=target_fpr,
        start_time=start_time,
        n_train_windows=n_train,
        n_test_windows=n_test,
        scores_calib=scores_calib,
        y_calib=y_calib,
    )
    return result


# ============================================================================
# 监督学习方法：RF, SVM, MLP (窗口级)
# ============================================================================

def run_rf_baseline(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    window_size: int,
    stride: int,
    target_fpr: float = 0.05,
    n_estimators: int = 100,
    max_depth: int = 20,
) -> list[BaselineResult]:
    """
    Random Forest baseline (窗口级监督学习)
    """
    print("\n" + "="*60)
    print("运行 Random Forest baseline (窗口级)")
    print("="*60)

    start_time = time.time()

    # 展平窗口
    n_train, w, f = X_train.shape
    n_test = X_test.shape[0]

    X_train_flat = X_train.reshape(n_train, w * f)
    X_test_flat = X_test.reshape(n_test, w * f)

    print(f"训练集形状: {X_train_flat.shape}")
    print(f"测试集形状: {X_test_flat.shape}")

    # 训练模型
    print("训练 Random Forest...")
    with tqdm(total=100, desc="RF Training") as pbar:
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=42,
            n_jobs=-1,
            verbose=0,
        )
        model.fit(X_train_flat, y_train)
        pbar.update(100)

    # 预测概率
    print("预测...")
    probs_test = model.predict_proba(X_test_flat)[:, 1]  # 异常类的概率
    probs_train = model.predict_proba(X_train_flat)[:, 1]

    # 计算指标
    result = compute_metrics(
        probs_train, y_train,
        probs_test, y_test,
        window_size, stride,
        method="RF (Window)",
        target_fpr=target_fpr,
        start_time=start_time,
        n_train_windows=n_train,
        n_test_windows=n_test,
    )

    return result


def run_svm_baseline(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    window_size: int,
    stride: int,
    target_fpr: float = 0.05,
) -> list[BaselineResult]:
    """
    SVM baseline (窗口级监督学习)
    """
    print("\n" + "="*60)
    print("运行 SVM baseline (窗口级)")
    print("="*60)

    start_time = time.time()

    # 展平窗口
    n_train, w, f = X_train.shape
    n_test = X_test.shape[0]

    X_train_flat = X_train.reshape(n_train, w * f)
    X_test_flat = X_test.reshape(n_test, w * f)

    print(f"训练集形状: {X_train_flat.shape}")
    print(f"测试集形状: {X_test_flat.shape}")

    # 训练模型
    print("训练 SVM...")
    with tqdm(total=100, desc="SVM Training") as pbar:
        model = SVC(
            kernel='rbf',
            probability=True,
            random_state=42,
        )
        model.fit(X_train_flat, y_train)
        pbar.update(100)

    # 预测概率
    print("预测...")
    probs_test = model.predict_proba(X_test_flat)[:, 1]
    probs_train = model.predict_proba(X_train_flat)[:, 1]

    # 计算指标
    result = compute_metrics(
        probs_train, y_train,
        probs_test, y_test,
        window_size, stride,
        method="SVM (Window)",
        target_fpr=target_fpr,
        start_time=start_time,
        n_train_windows=n_train,
        n_test_windows=n_test,
    )

    return result


def run_mlp_baseline(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    window_size: int,
    stride: int,
    target_fpr: float = 0.05,
    max_iter: int = 100,
) -> list[BaselineResult]:
    """
    MLP baseline (窗口级监督学习)
    """
    print("\n" + "="*60)
    print("运行 MLP baseline (窗口级)")
    print("="*60)

    start_time = time.time()

    # 展平窗口
    n_train, w, f = X_train.shape
    n_test = X_test.shape[0]

    X_train_flat = X_train.reshape(n_train, w * f)
    X_test_flat = X_test.reshape(n_test, w * f)

    print(f"训练集形状: {X_train_flat.shape}")
    print(f"测试集形状: {X_test_flat.shape}")

    # 训练模型
    print("训练 MLP...")
    model = MLPClassifier(
        hidden_layer_sizes=(256, 128, 64),
        activation='relu',
        solver='adam',
        batch_size=256,
        learning_rate_init=1e-3,
        max_iter=max_iter,
        random_state=42,
        verbose=True,
    )
    model.fit(X_train_flat, y_train)

    # 预测概率
    print("预测...")
    probs_test = model.predict_proba(X_test_flat)[:, 1]
    probs_train = model.predict_proba(X_train_flat)[:, 1]

    # 计算指标
    result = compute_metrics(
        probs_train, y_train,
        probs_test, y_test,
        window_size, stride,
        method="MLP (Window)",
        target_fpr=target_fpr,
        start_time=start_time,
        n_train_windows=n_train,
        n_test_windows=n_test,
    )

    return result


# ============================================================================
# 指标计算
# ============================================================================

def compute_metrics(
    scores_train: np.ndarray,
    y_train: np.ndarray,
    scores_test: np.ndarray,
    y_test: np.ndarray,
    window_size: int,
    stride: int,
    method: str,
    target_fpr: float | list[float],
    start_time: float,
    n_train_windows: int,
    n_test_windows: int,
    scores_calib: np.ndarray | None = None,
    y_calib: np.ndarray | None = None,
) -> list[BaselineResult]:
    """计算评价指标"""

    # AUC/AP：测试集只有单一类别时 AUC 退化，保留为 NaN，避免 smoke test 直接崩掉。
    if np.unique(y_test).size < 2:
        auc = float("nan")
    else:
        auc = roc_auc_score(y_test, scores_test)
    ap = average_precision_score(y_test, scores_test) if np.sum(y_test) > 0 else 0.0

    print(f"AUC: {auc:.4f}, AP: {ap:.4f}")

    # 可部署阈值标定
    # 只使用训练集 BENIGN 分数控制 FPR。监督 baseline 的训练集包含异常样本，
    # 如果把所有训练分数都用于阈值分位数，会把阈值推得过高，导致 RF 等模型全不报警。
    scores_train = np.asarray(scores_train, dtype=float)
    y_train = np.asarray(y_train, dtype=int)
    if scores_calib is not None:
        calib_scores_arr = np.asarray(scores_calib, dtype=float)
        calib_y_arr = np.zeros(len(calib_scores_arr), dtype=int) if y_calib is None else np.asarray(y_calib, dtype=int)
        if len(calib_y_arr) == len(calib_scores_arr):
            train_benign_scores = calib_scores_arr[calib_y_arr == 0]
        else:
            train_benign_scores = calib_scores_arr
    else:
        if len(y_train) == len(scores_train):
            train_benign_scores = scores_train[y_train == 0]
        else:
            train_benign_scores = scores_train
    train_benign_scores = train_benign_scores[np.isfinite(train_benign_scores)]
    if train_benign_scores.size == 0:
        raise ValueError(f"{method}: 训练集没有可用 BENIGN 分数，无法标定阈值")

    eval_seconds = time.time() - start_time
    target_fprs = target_fpr if isinstance(target_fpr, list) else [float(target_fpr)]
    results: list[BaselineResult] = []

    for fpr_target in target_fprs:
        fpr_target = float(fpr_target)
        if not (0.0 < fpr_target < 1.0):
            raise ValueError("target_fpr 必须在 (0,1) 范围内，例如 0.05")

        threshold = np.percentile(train_benign_scores, 100 * (1 - fpr_target))
        train_benign_fpr = float(np.mean(train_benign_scores >= threshold))

        print(f"标定阈值 (target_fpr={fpr_target}): {threshold:.6f}")

        # 测试集上的表现
        y_pred = (scores_test >= threshold).astype(int)

        # 计算precision, recall, f1
        tp = np.sum((y_pred == 1) & (y_test == 1))
        fp = np.sum((y_pred == 1) & (y_test == 0))
        fn = np.sum((y_pred == 0) & (y_test == 1))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        # 测试集正常样本上的FPR
        test_benign_idx = y_test == 0
        test_benign_scores = scores_test[test_benign_idx]
        test_benign_fpr = (
            float(np.mean(test_benign_scores >= threshold))
            if test_benign_scores.size
            else float("nan")
        )

        print(f"Calib F1: {f1:.4f}, Recall: {recall:.4f}, Precision: {precision:.4f}")
        print(f"Train BENIGN FPR: {train_benign_fpr:.4f}")
        print(f"Test BENIGN FPR: {test_benign_fpr:.4f}")

        results.append(
            BaselineResult(
                method=method,
                window_size=window_size,
                stride=stride,
                auc=float(auc),
                ap=float(ap),
                target_fpr=float(fpr_target),
                calib_f1=float(f1),
                calib_recall=float(recall),
                calib_precision=float(precision),
                calib_threshold=float(threshold),
                train_benign_fpr=float(train_benign_fpr),
                test_benign_fpr=float(test_benign_fpr),
                eval_seconds=eval_seconds,
                n_train_windows=n_train_windows,
                n_test_windows=n_test_windows,
            )
        )

    return results


# ============================================================================
# 主函数
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="窗口级baseline对比实验")

    parser.add_argument('--dataset', type=str, default='cicids2017',
                        choices=['cicids2017', 'swat', 'ton_iot', 'unsw_nb15', 'generic'],
                        help='数据集类型；非 cicids2017 时使用通用窗口加载器')
    parser.add_argument('--data-dir', type=str, required=True,
                        help='数据目录')
    parser.add_argument('--train-files', nargs='*', default=[],
                        help='训练文件列表')
    parser.add_argument('--test-files', nargs='*', default=[],
                        help='测试文件列表')
    parser.add_argument('--list-files', action='store_true',
                        help='列出 data-dir 下当前 dataset 可用的 CSV 文件名并退出')
    parser.add_argument('--supervised-mixed-split', action='store_true',
                        help='非 CICIDS 数据集使用混合监督切分：每个文件前一部分训练、后一部分测试，使 RF/SVM/MLP 能看到两类')
    parser.add_argument('--mixed-files', nargs='*', default=[],
                        help='supervised-mixed-split 使用的文件；SWaT 留空则默认用 5sec benign+attack')
    parser.add_argument('--mixed-train-fraction', type=float, default=0.5,
                        help='supervised-mixed-split 中每个文件用于训练的前缀比例')
    parser.add_argument('--unsupervised-formal-split', action='store_true',
                        help='异常检测正式切分：benign train + benign calib + benign test + attack test')
    parser.add_argument('--unsup-benign-file', type=str, default='',
                        help='unsupervised-formal-split 使用的 benign 文件')
    parser.add_argument('--unsup-attack-file', type=str, default='',
                        help='unsupervised-formal-split 使用的 attack 文件')
    parser.add_argument('--unsup-train-fraction', type=float, default=0.6,
                        help='unsupervised-formal-split 中 benign 前缀用于训练的比例')
    parser.add_argument('--unsup-calib-fraction', type=float, default=0.2,
                        help='unsupervised-formal-split 中 benign 中段用于阈值标定的比例')
    parser.add_argument('--chrono-unsupervised-split', action='store_true',
                        help='单文件时间切分的异常检测协议：train/calib/test 按时间顺序划分')
    parser.add_argument('--chrono-file', type=str, default='',
                        help='chrono-unsupervised-split 使用的单个混合文件')
    parser.add_argument('--chrono-train-fraction', type=float, default=0.6,
                        help='chrono-unsupervised-split 中前缀训练比例')
    parser.add_argument('--chrono-calib-fraction', type=float, default=0.1,
                        help='chrono-unsupervised-split 中中段标定比例')
    parser.add_argument('--window-size', type=int, default=128,
                        help='窗口大小')
    parser.add_argument('--stride', type=int, default=16,
                        help='步长')
    parser.add_argument('--anomaly-ratio', type=float, default=0.15,
                        help='窗口异常比例阈值')
    parser.add_argument('--target-fpr', type=float, default=0.05,
                        help='目标FPR')
    parser.add_argument('--target-fpr-grid', type=str, default='',
                        help='额外扫描多个目标FPR，例如 0.01,0.05,0.10,0.20。模型只训练一次，然后换阈值评估')
    parser.add_argument('--methods', nargs='+',
                        default=['iforest', 'ocsvm', 'rf', 'mlp', 'lstm_ae', 'lstm_ad'],
                        help='要运行的方法: iforest, ocsvm, deepsvdd, tranad, ganomaly, rf, svm, mlp, lstm_ae, lstm_ad')
    parser.add_argument('--output-dir', type=str, default='attack/results/baseline_window_manual',
                        help='输出目录')
    parser.add_argument('--device', type=str, default='cpu',
                        help='设备 (cpu/cuda)')
    parser.add_argument('--epochs', type=int, default=50,
                        help='LSTM训练轮数')
    parser.add_argument('--batch-size', type=int, default=128,
                        help='批次大小')
    parser.add_argument('--train-max', type=int, default=0,
                        help='训练窗口采样上限；0 表示不采样。用于快速预实验')
    parser.add_argument('--scaler', type=str, choices=['minmax', 'standard'], default='standard',
                        help='非 cicids2017 通用加载器使用的特征缩放方式；传统 baseline 默认 standard')
    parser.add_argument('--rf-n-estimators', type=int, default=100,
                        help='RandomForest 树数量')
    parser.add_argument('--rf-max-depth', type=int, default=20,
                        help='RandomForest 最大深度')
    parser.add_argument('--mlp-max-iter', type=int, default=100,
                        help='MLP 最大迭代次数')
    parser.add_argument('--deepsvdd-rep-dim', type=int, default=32,
                        help='DeepSVDD 表征维度')
    parser.add_argument('--tranad-d-model', type=int, default=64,
                        help='TranAD Transformer hidden size')
    parser.add_argument('--tranad-nhead', type=int, default=4,
                        help='TranAD attention heads')
    parser.add_argument('--tranad-num-layers', type=int, default=2,
                        help='TranAD encoder layers')
    parser.add_argument('--out-log', help='Path to save execution log')

    args = parser.parse_args()
    if args.out_log:
        log_path = Path(args.out_log)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        class Logger(object):
            def __init__(self, filename):
                self.terminal = sys.stdout
                self.log = open(filename, "a", encoding='utf-8')
            def write(self, message):
                self.terminal.write(message)
                self.log.write(message)
                self.log.flush()
            def flush(self):
                self.terminal.flush()
                self.log.flush()
        sys.stdout = Logger(args.out_log)
        print(f"Logging to {args.out_log}")
        print(f"Run started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    target_fprs = parse_target_fprs(args.target_fpr, args.target_fpr_grid)
    dataset_name = normalize_dataset_name(args.dataset)

    if args.list_files:
        for name in list_dataset_files(args.data_dir, dataset_name):
            print(name)
        return

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)

    # 判断是否需要监督学习数据
    need_supervised = any(m in args.methods for m in ['rf', 'svm', 'mlp'])

    # 加载数据
    print("\n" + "="*60)
    print("加载数据")
    print("="*60)
    
    X_calib = None
    y_calib = None
    
    if dataset_name == 'cicids2017':
        if not args.train_files or not args.test_files:
            raise ValueError('cicids2017 baseline 需要显式提供 --train-files 和 --test-files')
        X_train, y_train, X_test, y_test = load_cicids_windowed(
            args.data_dir,
            args.train_files,
            args.test_files,
            args.window_size,
            args.stride,
            args.anomaly_ratio,
            train_benign_only=not need_supervised,  # 监督学习需要所有标签
        )
    elif args.unsupervised_formal_split:
        ds = load_windowed_unsupervised_split(
            dataset=dataset_name,
            data_dir=args.data_dir,
            benign_file=args.unsup_benign_file or None,
            attack_file=args.unsup_attack_file or None,
            window_size=args.window_size,
            stride=args.stride,
            anomaly_ratio=args.anomaly_ratio,
            benign_train_fraction=args.unsup_train_fraction,
            benign_calib_fraction=args.unsup_calib_fraction,
            scaler=args.scaler,
            clip_minmax=(args.scaler == 'minmax'),
            progress=True,
        )
        X_train, y_train, X_test, y_test = ds.x_train, ds.y_train, ds.x_test, ds.y_test
        X_calib = ds.x_calib if ds.x_calib is not None else None
        y_calib = ds.y_calib if ds.y_calib is not None else None
        print(
            f"Loaded {dataset_name} unsupervised formal split: train_windows={len(X_train)}, "
            f"calib_windows={0 if X_calib is None else len(X_calib)}, "
            f"test_windows={len(X_test)}, test_anom_ratio={float(y_test.mean()):.4f}, "
            f"features={len(ds.feature_names)}"
        )
    elif args.chrono_unsupervised_split:
        ds = load_windowed_chrono_unsupervised_split(
            dataset=dataset_name,
            data_dir=args.data_dir,
            mixed_file=args.chrono_file or None,
            window_size=args.window_size,
            stride=args.stride,
            anomaly_ratio=args.anomaly_ratio,
            train_fraction=args.chrono_train_fraction,
            calib_fraction=args.chrono_calib_fraction,
            scaler=args.scaler,
            clip_minmax=(args.scaler == 'minmax'),
            progress=True,
        )
        X_train, y_train, X_test, y_test = ds.x_train, ds.y_train, ds.x_test, ds.y_test
        X_calib = ds.x_calib if ds.x_calib is not None else None
        y_calib = ds.y_calib if ds.y_calib is not None else None
        print(
            f"Loaded {dataset_name} chrono unsupervised split: train_windows={len(X_train)}, "
            f"calib_windows={0 if X_calib is None else len(X_calib)}, "
            f"test_windows={len(X_test)}, test_anom_ratio={float(y_test.mean()):.4f}, "
            f"features={len(ds.feature_names)}"
        )
    elif args.supervised_mixed_split:
        ds = load_windowed_mixed_split(
            dataset=dataset_name,
            data_dir=args.data_dir,
            files=args.mixed_files if args.mixed_files else None,
            window_size=args.window_size,
            stride=args.stride,
            anomaly_ratio=args.anomaly_ratio,
            train_fraction=args.mixed_train_fraction,
            scaler=args.scaler,
            clip_minmax=False,
            progress=True,
        )
        X_train, y_train, X_test, y_test = ds.x_train, ds.y_train, ds.x_test, ds.y_test
        X_calib = None
        y_calib = None
        print(
            f"Loaded {dataset_name} supervised mixed split: train_windows={len(X_train)}, "
            f"train_anom_ratio={float(y_train.mean()):.4f}, test_windows={len(X_test)}, "
            f"test_anom_ratio={float(y_test.mean()):.4f}, features={len(ds.feature_names)}"
        )
    else:
        ds = load_windowed_dataset(
            dataset=dataset_name,
            data_dir=args.data_dir,
            train_files=args.train_files,
            test_files=args.test_files,
            window_size=args.window_size,
            stride=args.stride,
            anomaly_ratio=args.anomaly_ratio,
            train_benign_only=not need_supervised,
            scaler=args.scaler,
            clip_minmax=False,
            progress=True,
        )
        X_train, y_train, X_test, y_test = ds.x_train, ds.y_train, ds.x_test, ds.y_test
        X_calib = None
        y_calib = None
        print(
            f"Loaded {dataset_name}: train_windows={len(X_train)}, "
            f"test_windows={len(X_test)}, test_anom_ratio={float(y_test.mean()):.4f}, "
            f"features={len(ds.feature_names)}"
        )

    benign_train_idx = y_train == 0
    X_train_benign = X_train[benign_train_idx] if np.any(benign_train_idx) else X_train
    X_calib_benign = None
    if X_calib is not None:
        calib_idx = y_calib == 0 if y_calib is not None else np.ones(len(X_calib), dtype=bool)
        X_calib_benign = X_calib[calib_idx] if np.any(calib_idx) else X_calib

    if args.train_max and len(X_train) > args.train_max:
        print(f"\n训练窗口采样: {len(X_train)} -> {args.train_max}")
        rng = np.random.default_rng(42)
        idx = rng.choice(len(X_train), size=int(args.train_max), replace=False)
        X_train = X_train[idx]
        y_train = y_train[idx]
        benign_train_idx = y_train == 0
        X_train_benign = X_train[benign_train_idx] if np.any(benign_train_idx) else X_train

    # 运行baseline
    results = []
    total_methods = len(args.methods)
    
    print("\n" + "="*60)
    print(f"开始运行 {total_methods} 个baseline方法")
    print("="*60)
    method_idx = 0

    if 'iforest' in args.methods:
        method_idx += 1
        print(f"\n[{method_idx}/{total_methods}] IsolationForest")
        result = run_iforest_baseline(
            X_train_benign, X_test, y_test,
            args.window_size, args.stride,
            contamination=0.1,
            target_fpr=target_fprs,
            X_calib=X_calib_benign,
        )
        results.extend(result)

    if 'ocsvm' in args.methods:
        method_idx += 1
        print(f"\n[{method_idx}/{total_methods}] OneClassSVM")
        result = run_ocsvm_baseline(
            X_train_benign, X_test, y_test,
            args.window_size, args.stride,
            nu=0.1,
            target_fpr=target_fprs,
            X_calib=X_calib_benign,
        )
        results.extend(result)

    if 'deepsvdd' in args.methods:
        method_idx += 1
        print(f"\n[{method_idx}/{total_methods}] DeepSVDD")
        result = run_deepsvdd_baseline(
            X_train_benign, X_test, y_test,
            args.window_size, args.stride,
            rep_dim=args.deepsvdd_rep_dim,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=1e-3,
            target_fpr=target_fprs,
            device=args.device,
            X_calib=X_calib_benign,
        )
        results.extend(result)

    if 'tranad' in args.methods:
        method_idx += 1
        print(f"\n[{method_idx}/{total_methods}] TranAD")
        result = run_tranad_baseline(
            X_train_benign, X_test, y_test,
            args.window_size, args.stride,
            d_model=args.tranad_d_model,
            nhead=args.tranad_nhead,
            num_layers=args.tranad_num_layers,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=1e-3,
            target_fpr=target_fprs,
            device=args.device,
            X_calib=X_calib_benign,
        )
        results.extend(result)

    if 'ganomaly' in args.methods:
        method_idx += 1
        print(f"\n[{method_idx}/{total_methods}] GANomaly")
        result = run_ganomaly_baseline(
            X_train_benign, X_test, y_test,
            args.window_size, args.stride,
            latent_dim=64,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=1e-3,
            target_fpr=target_fprs,
            device=args.device,
            X_calib=X_calib_benign,
        )
        results.extend(result)

    if 'rf' in args.methods:
        method_idx += 1
        print(f"\n[{method_idx}/{total_methods}] Random Forest")
        if np.unique(y_train).size < 2:
            result = skipped_results(
                "RF (Window)",
                args.window_size,
                args.stride,
                target_fprs,
                "training labels contain only one class",
                len(X_train),
                len(X_test),
            )
        else:
            result = run_rf_baseline(
                X_train, y_train, X_test, y_test,
                args.window_size, args.stride,
                target_fpr=target_fprs,
                n_estimators=args.rf_n_estimators,
                max_depth=args.rf_max_depth,
            )
        results.extend(result)

    if 'svm' in args.methods:
        method_idx += 1
        print(f"\n[{method_idx}/{total_methods}] SVM")
        if np.unique(y_train).size < 2:
            result = skipped_results(
                "SVM (Window)",
                args.window_size,
                args.stride,
                target_fprs,
                "training labels contain only one class",
                len(X_train),
                len(X_test),
            )
        else:
            result = run_svm_baseline(
                X_train, y_train, X_test, y_test,
                args.window_size, args.stride,
                target_fpr=target_fprs,
            )
        results.extend(result)

    if 'mlp' in args.methods:
        method_idx += 1
        print(f"\n[{method_idx}/{total_methods}] MLP")
        if np.unique(y_train).size < 2:
            result = skipped_results(
                "MLP (Window)",
                args.window_size,
                args.stride,
                target_fprs,
                "training labels contain only one class",
                len(X_train),
                len(X_test),
            )
        else:
            result = run_mlp_baseline(
                X_train, y_train, X_test, y_test,
                args.window_size, args.stride,
                target_fpr=target_fprs,
                max_iter=args.mlp_max_iter,
            )
        results.extend(result)

    if 'lstm_ae' in args.methods:
        method_idx += 1
        print(f"\n[{method_idx}/{total_methods}] LSTM-AE")
        result = run_lstm_ae_baseline(
            X_train_benign, X_test, y_test,
            args.window_size, args.stride,
            hidden_dim=64, num_layers=2,
            epochs=args.epochs, batch_size=args.batch_size,
            lr=1e-3, target_fpr=target_fprs,
            device=args.device,
            X_calib=X_calib_benign,
        )
        results.extend(result)

    if 'lstm_ad' in args.methods:
        method_idx += 1
        print(f"\n[{method_idx}/{total_methods}] LSTM-AD")
        result = run_lstm_ad_baseline(
            X_train_benign, X_test, y_test,
            args.window_size, args.stride,
            hidden_dim=64, num_layers=2,
            epochs=args.epochs, batch_size=args.batch_size,
            lr=1e-3, target_fpr=target_fprs,
            device=args.device,
            X_calib=X_calib_benign,
        )
        results.extend(result)

    # 保存结果
    print("\n" + "="*60)
    print("保存结果")
    print("="*60)

    # 保存JSON
    results_dict = [asdict(r) for r in results]
    json_path = os.path.join(args.output_dir, 'baseline_results.json')
    with open(json_path, 'w') as f:
        json.dump(results_dict, f, indent=2)
    print(f"结果已保存到: {json_path}")

    # 保存CSV
    df = pd.DataFrame(results_dict)
    csv_path = os.path.join(args.output_dir, 'baseline_results.csv')
    df.to_csv(csv_path, index=False)
    print(f"结果已保存到: {csv_path}")

    # 打印汇总表格
    print("\n" + "="*60)
    print("Baseline对比结果汇总")
    print("="*60)
    print(df[['method', 'auc', 'ap', 'calib_f1', 'calib_recall', 'test_benign_fpr', 'eval_seconds']])


if __name__ == '__main__':
    main()
