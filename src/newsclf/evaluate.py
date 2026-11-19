"""Metrics, significance testing, plots and error analysis."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
)

from newsclf import LABELS


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "macro_f1": round(float(f1_score(y_true, y_pred, average="macro")), 4),
        "per_class_f1": {label: round(float(f), 4) for label, f in zip(LABELS, f1_score(y_true, y_pred, average=None), strict=True)},
    }


def mcnemar(y_true: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray) -> dict:
    """Exact McNemar test: is the accuracy difference between two models on the SAME test set real?

    Only discordant articles matter: b = A right & B wrong, c = A wrong & B right.
    Under H0 (equal error rates) b ~ Binomial(b + c, 0.5).
    """
    a_right, b_right = pred_a == y_true, pred_b == y_true
    b = int(np.sum(a_right & ~b_right))
    c = int(np.sum(~a_right & b_right))
    p = binomtest(b, b + c, 0.5).pvalue if b + c else 1.0
    return {"only_a_correct": b, "only_b_correct": c, "p_value": float(p)}


def bootstrap_ci(y_true: np.ndarray, y_pred: np.ndarray, n: int = 1000, seed: int = 0) -> tuple[float, float]:
    """95% bootstrap confidence interval for accuracy."""
    rng = np.random.default_rng(seed)
    correct = (y_true == y_pred).astype(float)
    samples = [correct[rng.integers(0, len(correct), len(correct))].mean() for _ in range(n)]
    return round(float(np.percentile(samples, 2.5)), 4), round(float(np.percentile(samples, 97.5)), 4)


def plot_confusion(y_true: np.ndarray, y_pred: np.ndarray, title: str, path: Path) -> None:
    cm = confusion_matrix(y_true, y_pred, labels=range(len(LABELS)))
    fig, ax = plt.subplots(figsize=(5, 4.3))
    ax.imshow(cm, cmap="Blues")
    for i in range(len(LABELS)):
        for j in range(len(LABELS)):
            ax.text(j, i, cm[i, j], ha="center", va="center", color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set(xticks=range(len(LABELS)), yticks=range(len(LABELS)), xticklabels=LABELS, yticklabels=LABELS, xlabel="Predicted", ylabel="True", title=title)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_learning_curve(curve: list[dict], transformer_point: tuple[int, float], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot([r["train_size"] for r in curve], [r["accuracy"] for r in curve], "o-", label="TF-IDF + linear SVM")
    ax.plot(*transformer_point, "r*", markersize=14, label="DistilBERT (fine-tuned)")
    ax.set(xscale="log", xlabel="Labelled training articles (log scale)", ylabel="Test accuracy", title="Accuracy vs labelled data")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def confusion_pairs(y_true: np.ndarray, y_pred: np.ndarray, top: int = 4) -> list[dict]:
    cm = confusion_matrix(y_true, y_pred, labels=range(len(LABELS)))
    np.fill_diagonal(cm, 0)
    flat = np.argsort(cm, axis=None)[::-1][:top]
    return [{"true": LABELS[i], "predicted": LABELS[j], "count": int(cm[i, j])} for i, j in zip(*np.unravel_index(flat, cm.shape), strict=True)]


def error_table(test: pd.DataFrame, probs_a: np.ndarray, probs_b: np.ndarray) -> pd.DataFrame:
    """One row per test article with both models' predictions, confidence and length."""
    df = test.copy()
    df["pred_baseline"], df["pred_bert"] = probs_a.argmax(1), probs_b.argmax(1)
    df["conf_baseline"], df["conf_bert"] = probs_a.max(1), probs_b.max(1)
    df["n_words"] = df["text"].str.split().str.len()
    return df


def error_breakdown(df: pd.DataFrame) -> dict:
    a_wrong, b_wrong = df["pred_baseline"] != df["label"], df["pred_bert"] != df["label"]
    both = a_wrong & b_wrong
    # Both models agree with each other but not with the gold label: a strong hint of label noise.
    agree_against_gold = both & (df["pred_baseline"] == df["pred_bert"])
    buckets = pd.cut(df["n_words"], bins=[0, 25, 35, 45, 1000], labels=["<=25", "26-35", "36-45", ">45"])
    by_length = (
        df.assign(err_baseline=a_wrong, err_bert=b_wrong, bucket=buckets)
        .groupby("bucket", observed=True)
        .agg(n=("label", "size"), baseline_error=("err_baseline", "mean"), bert_error=("err_bert", "mean"))
        .round(4)
    )
    confident_wrong = b_wrong & (df["conf_bert"] >= 0.9)
    return {
        "n_test": len(df),
        "baseline_errors": int(a_wrong.sum()),
        "bert_errors": int(b_wrong.sum()),
        "both_wrong": int(both.sum()),
        "both_wrong_same_label": int(agree_against_gold.sum()),
        "bert_confident_errors": int(confident_wrong.sum()),
        "error_rate_by_length": by_length.reset_index().astype({"bucket": str}).to_dict(orient="records"),
    }


def sample_errors(df: pd.DataFrame, n: int = 12) -> pd.DataFrame:
    """Most confident DistilBERT mistakes: the most informative errors to read by hand."""
    wrong = df[df["pred_bert"] != df["label"]].sort_values("conf_bert", ascending=False).head(n)
    return pd.DataFrame(
        {
            "gold": wrong["label"].map(dict(enumerate(LABELS))),
            "bert": wrong["pred_bert"].map(dict(enumerate(LABELS))),
            "conf": wrong["conf_bert"].round(3),
            "baseline": wrong["pred_baseline"].map(dict(enumerate(LABELS))),
            "text": wrong["text"].str.slice(0, 160),
        }
    )
