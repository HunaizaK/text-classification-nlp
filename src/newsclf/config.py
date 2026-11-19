"""Typed configuration loaded from YAML, with an optional `full:` override block."""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel


class DataConfig(BaseModel):
    raw_dir: Path
    train_url: str
    test_url: str
    train_sha256: str
    test_sha256: str
    train_size: int
    val_size: int
    test_size: int


class BaselineConfig(BaseModel):
    c_grid: list[float]
    learning_curve_sizes: list[int]


class TransformerConfig(BaseModel):
    checkpoint: str
    max_length: int
    epochs: float
    batch_size: int
    learning_rate: float
    weight_decay: float
    warmup_ratio: float


class DistillConfig(BaseModel):
    unlabeled_size: int
    min_confidence: float


class CompressConfig(BaseModel):
    latency_samples: int


class Config(BaseModel):
    seed: int
    data: DataConfig
    baseline: BaselineConfig
    transformer: TransformerConfig
    distill: DistillConfig
    compress: CompressConfig
    artifacts_dir: Path
    reports_dir: Path


def _merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for key, value in override.items():
        out[key] = _merge(out[key], value) if isinstance(value, dict) and isinstance(out.get(key), dict) else value
    return out


def load_config(path: str | Path = "configs/config.yaml", full: bool = False) -> Config:
    raw = yaml.safe_load(Path(path).read_text())
    overrides = raw.pop("full", {}) or {}
    if full:
        raw = _merge(raw, overrides)
    return Config(**raw)
