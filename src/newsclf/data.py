"""Download, verify, clean and split AG News."""

from __future__ import annotations

import hashlib
import html
import logging
import re
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from newsclf import LABELS
from newsclf.config import Config

log = logging.getLogger(__name__)

# AG News replaced "&" with a space: "&#36;" -> " #36;" and "&quot;" -> " quot;".
_ENTITY = re.compile(r"\s?#(\d+);")
_NAMED_ENTITY = re.compile(r"\s?\b(quot|amp|lt|gt|apos|nbsp|hellip|[lr]squo|[lr]dquo|[mn]dash);")
_SPACES = re.compile(r"\s+")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(url: str, dest: Path, expected_sha256: str) -> Path:
    """Download once, then always verify the checksum so results are tied to exact bytes."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.exists():
        log.info("Downloading %s", url)
        urllib.request.urlretrieve(url, dest)
    actual = sha256(dest)
    if actual != expected_sha256:
        raise ValueError(f"Checksum mismatch for {dest}: {actual}")
    return dest


def clean_text(text: str) -> str:
    """Undo AG News encoding artefacts: '\' line breaks, ' #36;' entities, HTML escapes, tags."""
    text = text.replace("\\", " ")
    text = _ENTITY.sub(lambda m: html.unescape(f"&#{m.group(1)};"), text)
    text = html.unescape(text)  # genuine "&lt;" etc. first, so they are not half-matched below
    text = _NAMED_ENTITY.sub(lambda m: html.unescape(f"&{m.group(1)};"), text)
    text = re.sub(r"<[^>]+>", " ", text)
    return _SPACES.sub(" ", text).strip()


def load_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, header=None, names=["label", "title", "description"], dtype={"label": int, "title": str, "description": str})
    df = df.fillna("")
    df["label"] = df["label"] - 1  # 1..4 -> 0..3
    df["text"] = (df["title"].map(clean_text) + ". " + df["description"].map(clean_text)).str.strip()
    validate(df)
    return df[["text", "label"]]


def validate(df: pd.DataFrame) -> None:
    if not df["label"].between(0, len(LABELS) - 1).all():
        raise ValueError("Labels outside 0..3")
    if (df["text"].str.len() == 0).any():
        raise ValueError("Empty documents found")


@dataclass
class Splits:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame
    train_pool: pd.DataFrame  # everything in the training CSV not used for validation (learning curve)


def stratified_take(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    if n >= len(df):
        return df.sample(frac=1, random_state=seed).reset_index(drop=True)
    taken, _ = train_test_split(df, train_size=n, stratify=df["label"], random_state=seed)
    return taken.reset_index(drop=True)


def make_splits(train_df: pd.DataFrame, test_df: pd.DataFrame, cfg: Config) -> Splits:
    n_raw = len(train_df)
    train_df = train_df.drop_duplicates("text")
    pool, val = train_test_split(train_df, test_size=cfg.data.val_size, stratify=train_df["label"], random_state=cfg.seed)
    # Drop exact test duplicates from training data so the test score is not inflated by leakage.
    leaked = pool["text"].isin(set(test_df["text"]))
    log.info("removed %d duplicate training articles and %d that also appear in the test set", n_raw - len(train_df), int(leaked.sum()))
    pool = pool[~leaked].reset_index(drop=True)
    return Splits(
        train=stratified_take(pool, cfg.data.train_size, cfg.seed),
        val=val.reset_index(drop=True),
        test=stratified_take(test_df, cfg.data.test_size, cfg.seed),
        train_pool=pool,
    )


def load_splits(cfg: Config) -> Splits:
    raw = Path(cfg.data.raw_dir)
    train_path = download(cfg.data.train_url, raw / "train.csv", cfg.data.train_sha256)
    test_path = download(cfg.data.test_url, raw / "test.csv", cfg.data.test_sha256)
    splits = make_splits(load_csv(train_path), load_csv(test_path), cfg)
    log.info("train=%d val=%d test=%d pool=%d", len(splits.train), len(splits.val), len(splits.test), len(splits.train_pool))
    return splits
