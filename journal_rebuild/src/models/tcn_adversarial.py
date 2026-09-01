from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


try:
    from torch.nn.utils.parametrizations import weight_norm as _weight_norm  # type: ignore[attr-defined]
except Exception:  # pragma: no cover
    from torch.nn.utils import weight_norm as _weight_norm  # type: ignore[no-redef]


class WindowDataset(Dataset):
    def __init__(self, features: np.ndarray, starts: np.ndarray, window_size: int, labels: np.ndarray) -> None:
        self.features = np.asarray(features, dtype=np.float32)
        self.starts = np.asarray(starts, dtype=np.int64)
        self.window_size = int(window_size)
        self.labels = np.asarray(labels, dtype=np.uint8)

    def __len__(self) -> int:
        return len(self.starts)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        start = int(self.starts[idx])
        end = start + self.window_size
        window = self.features[start:end]
        label = int(self.labels[idx])
        return torch.from_numpy(window).float(), torch.tensor(label, dtype=torch.long)


class Chomp1d(nn.Module):
    def __init__(self, chomp_size: int) -> None:
        super().__init__()
        self.chomp_size = int(chomp_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.chomp_size <= 0:
            return x
        return x[:, :, :-self.chomp_size]


class TemporalBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int, dilation: int, dropout: float) -> None:
        super().__init__()
        padding = (kernel_size - 1) * dilation
        self.conv1 = _weight_norm(
            nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation)
        )
        self.chomp1 = Chomp1d(padding)
        self.conv2 = _weight_norm(
            nn.Conv1d(out_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation)
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
    def __init__(self, num_inputs: int, num_channels: Iterable[int], kernel_size: int = 3, dropout: float = 0.2) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        channels = list(num_channels)
        for idx, out_channels in enumerate(channels):
            dilation = 2 ** idx
            in_channels = num_inputs if idx == 0 else channels[idx - 1]
            layers.append(TemporalBlock(in_channels, out_channels, kernel_size, 1, dilation, dropout))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class TCNGenerator(nn.Module):
    def __init__(self, seq_len: int, feat_dim: int, latent_dim: int, hidden_channels: Iterable[int], dropout: float) -> None:
        super().__init__()
        channels = list(hidden_channels)
        self.seq_len = int(seq_len)
        self.initial = nn.Linear(int(latent_dim), int(channels[0]))
        self.tcn = TemporalConvNet(int(channels[0]), channels, dropout=float(dropout))
        self.to_feature = nn.Conv1d(int(channels[-1]), int(feat_dim), kernel_size=1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = self.initial(z).unsqueeze(-1)
        x = x.expand(-1, -1, self.seq_len)
        x = self.tcn(x)
        out = self.to_feature(x)
        return torch.sigmoid(out.permute(0, 2, 1))


class TCNDiscriminator(nn.Module):
    def __init__(self, seq_len: int, feat_dim: int, hidden_channels: Iterable[int], dropout: float, pooling: str = "attn") -> None:
        super().__init__()
        channels = list(hidden_channels)
        self.pooling = str(pooling).lower().strip()
        self.tcn = TemporalConvNet(int(feat_dim), channels, dropout=float(dropout))
        self.classifier = nn.Conv1d(int(channels[-1]), 1, kernel_size=1)
        self.attn = nn.Conv1d(int(channels[-1]), 1, kernel_size=1) if self.pooling in {"attn", "attention"} else None

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.tcn(x.permute(0, 2, 1))

    def forward_time_logits(self, features: torch.Tensor) -> torch.Tensor:
        return self.classifier(features).squeeze(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.forward_features(x)
        logits_t = self.forward_time_logits(features)
        if self.attn is not None:
            attn_weights = torch.softmax(self.attn(features).squeeze(1), dim=1)
            return (attn_weights * logits_t).sum(dim=1, keepdim=True)
        return logits_t.mean(dim=1, keepdim=True)


@dataclass
class TrainingArtifacts:
    generator: TCNGenerator
    discriminator: TCNDiscriminator
    training_log: list[dict[str, float | int | bool]]
    train_seconds: float
    failed_nan: bool
    g_optimizer_state: dict[str, Any]
    d_optimizer_state: dict[str, Any]


def build_models(model_config: dict[str, Any], feature_dim: int, window_size: int) -> tuple[TCNGenerator, TCNDiscriminator]:
    generator = TCNGenerator(
        seq_len=int(window_size),
        feat_dim=int(feature_dim),
        latent_dim=int(model_config["latent_dim"]),
        hidden_channels=model_config["hidden_channels"],
        dropout=float(model_config["dropout"]),
    )
    discriminator = TCNDiscriminator(
        seq_len=int(window_size),
        feat_dim=int(feature_dim),
        hidden_channels=model_config["hidden_channels"],
        dropout=float(model_config["dropout"]),
        pooling=str(model_config["pooling"]),
    )
    return generator, discriminator


def build_loader(features: np.ndarray, starts: np.ndarray, window_size: int, labels: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    dataset = WindowDataset(features=features, starts=starts, window_size=window_size, labels=labels)
    return DataLoader(dataset, batch_size=int(batch_size), shuffle=bool(shuffle), num_workers=0)


def train_one_epoch(
    generator: nn.Module,
    discriminator: nn.Module,
    train_loader: DataLoader,
    device: torch.device,
    g_optimizer: torch.optim.Optimizer,
    d_optimizer: torch.optim.Optimizer,
    latent_dim: int,
    loss_type: str,
    gp_lambda: float,
    n_critic: int,
) -> tuple[float, float]:
    generator.train()
    discriminator.train()
    bce = nn.BCEWithLogitsLoss()
    loss_type = str(loss_type).lower().strip()
    d_loss_total = 0.0
    g_loss_total = 0.0
    for real_windows, _ in train_loader:
        real_windows = real_windows.to(device)
        batch_size = real_windows.size(0)
        real_labels = torch.ones(batch_size, 1, device=device)
        fake_labels = torch.zeros(batch_size, 1, device=device)
        critic_steps = int(n_critic) if loss_type == "wgan_gp" else 1

        for _ in range(critic_steps):
            z = torch.randn(batch_size, int(latent_dim), device=device)
            fake_windows = generator(z)
            d_real = discriminator(real_windows)
            d_fake = discriminator(fake_windows.detach())
            if loss_type == "wgan_gp":
                d_loss = d_fake.mean() - d_real.mean()
                eps = torch.rand(batch_size, 1, 1, device=device)
                x_hat = eps * real_windows + (1.0 - eps) * fake_windows.detach()
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
                d_loss = bce(d_real, real_labels) + bce(d_fake, fake_labels)
            d_optimizer.zero_grad()
            d_loss.backward()
            d_optimizer.step()

        z = torch.randn(batch_size, int(latent_dim), device=device)
        fake_windows = generator(z)
        d_fake = discriminator(fake_windows)
        if loss_type == "wgan_gp":
            g_loss = -d_fake.mean()
        else:
            g_loss = bce(d_fake, real_labels)
        g_optimizer.zero_grad()
        g_loss.backward()
        g_optimizer.step()

        d_loss_total += float(d_loss.item())
        g_loss_total += float(g_loss.item())
    num_batches = max(len(train_loader), 1)
    return d_loss_total / num_batches, g_loss_total / num_batches


def train_model(
    model_config: dict[str, Any],
    *,
    feature_dim: int,
    window_size: int,
    train_features: np.ndarray,
    train_starts: np.ndarray,
    train_labels: np.ndarray,
    device: torch.device,
    epochs: int,
) -> TrainingArtifacts:
    generator, discriminator = build_models(model_config, feature_dim=feature_dim, window_size=window_size)
    generator = generator.to(device)
    discriminator = discriminator.to(device)
    optimizer_cfg = model_config["optimizer"]
    g_optimizer = torch.optim.Adam(
        generator.parameters(),
        lr=float(optimizer_cfg["lr"]),
        betas=tuple(float(x) for x in optimizer_cfg["betas"]),
    )
    d_optimizer = torch.optim.Adam(
        discriminator.parameters(),
        lr=float(optimizer_cfg["lr"]),
        betas=tuple(float(x) for x in optimizer_cfg["betas"]),
    )
    train_loader = build_loader(
        features=train_features,
        starts=train_starts,
        window_size=window_size,
        labels=train_labels,
        batch_size=int(model_config["batch_size"]),
        shuffle=True,
    )
    training_log: list[dict[str, float | int | bool]] = []
    failed_nan = False
    start_time = torch.cuda.Event(enable_timing=True) if device.type == "cuda" else None
    end_time = torch.cuda.Event(enable_timing=True) if device.type == "cuda" else None
    cpu_start = torch.tensor(0.0)
    if start_time is not None and end_time is not None:
        start_time.record()
    else:
        import time

        cpu_start = torch.tensor(time.perf_counter(), dtype=torch.float64)
    for epoch in range(1, int(epochs) + 1):
        d_loss, g_loss = train_one_epoch(
            generator=generator,
            discriminator=discriminator,
            train_loader=train_loader,
            device=device,
            g_optimizer=g_optimizer,
            d_optimizer=d_optimizer,
            latent_dim=int(model_config["latent_dim"]),
            loss_type=str(model_config["loss_type"]),
            gp_lambda=float(model_config["gp_lambda"]),
            n_critic=int(model_config["n_critic"]),
        )
        failed = (not np.isfinite(d_loss)) or (not np.isfinite(g_loss))
        failed_nan = failed_nan or failed
        training_log.append(
            {
                "epoch": int(epoch),
                "d_loss": float(d_loss),
                "g_loss": float(g_loss),
                "failed_nan": bool(failed),
            }
        )
        if failed:
            break
    if start_time is not None and end_time is not None:
        end_time.record()
        torch.cuda.synchronize()
        train_seconds = float(start_time.elapsed_time(end_time) / 1000.0)
    else:
        import time

        train_seconds = float(time.perf_counter() - float(cpu_start))
    return TrainingArtifacts(
        generator=generator,
        discriminator=discriminator,
        training_log=training_log,
        train_seconds=train_seconds,
        failed_nan=failed_nan,
        g_optimizer_state=g_optimizer.state_dict(),
        d_optimizer_state=d_optimizer.state_dict(),
    )


def count_parameters(module: nn.Module) -> int:
    return int(sum(p.numel() for p in module.parameters() if p.requires_grad))


def compare_model_configs_except_loss(left: dict[str, Any], right: dict[str, Any]) -> tuple[bool, list[str]]:
    ignore_keys = {"model_name", "loss_type", "gp_lambda", "n_critic"}
    left_cmp = {k: copy.deepcopy(v) for k, v in left.items() if k not in ignore_keys}
    right_cmp = {k: copy.deepcopy(v) for k, v in right.items() if k not in ignore_keys}
    same = left_cmp == right_cmp
    diffs: list[str] = []
    if not same:
        keys = sorted(set(left_cmp) | set(right_cmp))
        for key in keys:
            if left_cmp.get(key) != right_cmp.get(key):
                diffs.append(key)
    return same, diffs


def save_checkpoint(
    path: Path,
    *,
    model_name: str,
    loss_type: str,
    seed: int,
    epoch: int,
    model_config: dict[str, Any],
    data_config: dict[str, Any],
    feature_names: list[str],
    scaler_hash: str,
    manifest_hash: str,
    source_file_hashes: dict[str, str],
    git_commit: str,
    timestamp: str,
    generator: nn.Module,
    discriminator: nn.Module,
    g_optimizer: Any = None,
    d_optimizer: Any = None,
) -> None:
    payload = {
        "model_name": str(model_name),
        "loss_type": str(loss_type),
        "seed": int(seed),
        "epoch": int(epoch),
        "model_config": model_config,
        "data_config": data_config,
        "feature_names": feature_names,
        "scaler_hash": str(scaler_hash),
        "manifest_hash": str(manifest_hash),
        "source_file_hashes": source_file_hashes,
        "git_commit": str(git_commit),
        "timestamp": str(timestamp),
        "optimizer_state": {
            "generator": g_optimizer.state_dict() if hasattr(g_optimizer, "state_dict") else g_optimizer,
            "discriminator": d_optimizer.state_dict() if hasattr(d_optimizer, "state_dict") else d_optimizer,
        },
        "model_state": {
            "generator": generator.state_dict(),
            "discriminator": discriminator.state_dict(),
        },
    }
    torch.save(payload, path)


def load_checkpoint_validated(
    path: Path,
    *,
    expected_model_name: str,
    expected_loss_type: str,
    expected_manifest_hash: str,
    expected_scaler_hash: str,
) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu")
    if str(payload.get("model_name")) != str(expected_model_name):
        raise ValueError("Checkpoint model_name mismatch")
    if str(payload.get("loss_type")) != str(expected_loss_type):
        raise ValueError("Checkpoint loss_type mismatch")
    if str(payload.get("manifest_hash")) != str(expected_manifest_hash):
        raise ValueError("Checkpoint manifest_hash mismatch")
    if str(payload.get("scaler_hash")) != str(expected_scaler_hash):
        raise ValueError("Checkpoint scaler_hash mismatch")
    return payload


def load_models_from_checkpoint(
    path: Path,
    *,
    expected_model_name: str,
    expected_loss_type: str,
    expected_manifest_hash: str,
    expected_scaler_hash: str,
    feature_dim: int,
    window_size: int,
    device: torch.device,
) -> tuple[dict[str, Any], TCNGenerator, TCNDiscriminator]:
    payload = load_checkpoint_validated(
        path,
        expected_model_name=expected_model_name,
        expected_loss_type=expected_loss_type,
        expected_manifest_hash=expected_manifest_hash,
        expected_scaler_hash=expected_scaler_hash,
    )
    model_config = dict(payload["model_config"])
    generator, discriminator = build_models(model_config, feature_dim=feature_dim, window_size=window_size)
    generator.load_state_dict(payload["model_state"]["generator"])
    discriminator.load_state_dict(payload["model_state"]["discriminator"])
    generator = generator.to(device)
    discriminator = discriminator.to(device)
    generator.eval()
    discriminator.eval()
    return payload, generator, discriminator
