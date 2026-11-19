"""End-to-end steps. Each step saves its outputs so later steps can be re-run cheaply."""

from __future__ import annotations

import json
import logging
import shutil
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from newsclf import LABELS, baseline, compress, distill, evaluate, transformer
from newsclf.config import Config
from newsclf.data import Splits, load_splits, stratified_take

log = logging.getLogger(__name__)


def _paths(cfg: Config) -> dict[str, Path]:
    art, rep = Path(cfg.artifacts_dir), Path(cfg.reports_dir)
    (rep / "figures").mkdir(parents=True, exist_ok=True)
    art.mkdir(parents=True, exist_ok=True)
    return {
        "baseline": art / "baseline.joblib",
        "student": art / "student.joblib",
        "bert": art / "distilbert",
        "probs": art / "test_probs.npz",
        "metrics": rep / "metrics.json",
        "figures": rep / "figures",
        "reports": rep,
    }


def _update_metrics(path: Path, key: str, value) -> None:
    data = json.loads(path.read_text()) if path.exists() else {}
    data[key] = value
    path.write_text(json.dumps(data, indent=2))


def _save_probs(path: Path, **arrays) -> None:
    existing = dict(np.load(path)) if path.exists() else {}
    existing.update(arrays)
    np.savez(path, **existing)


def step_baseline(cfg: Config, splits: Splits) -> None:
    p = _paths(cfg)
    model, info = baseline.train_baseline(splits, cfg)
    joblib.dump(model, p["baseline"])
    probs = model.predict_proba(splits.test["text"])
    _save_probs(p["probs"], baseline=probs, y=splits.test["label"].to_numpy())
    info["test"] = evaluate.metrics(splits.test["label"].to_numpy(), probs.argmax(1))
    info["top_features"] = baseline.top_features(model, LABELS)
    info["learning_curve"] = baseline.learning_curve(splits, cfg, info["best_c"])
    _update_metrics(p["metrics"], "baseline", info)
    log.info("baseline test %s", info["test"])


def step_bert(cfg: Config, splits: Splits) -> None:
    p = _paths(cfg)
    model, tokenizer, info = transformer.train_transformer(splits, cfg)
    shutil.rmtree(p["bert"] / "checkpoints", ignore_errors=True)  # best weights are already saved at the top level
    start = time.perf_counter()
    probs = transformer.predict_proba(model, tokenizer, splits.test["text"].tolist(), cfg.transformer.max_length)
    info["test_inference_seconds"] = round(time.perf_counter() - start, 1)
    _save_probs(p["probs"], bert=probs)
    info["test"] = evaluate.metrics(splits.test["label"].to_numpy(), probs.argmax(1))
    _update_metrics(p["metrics"], "distilbert", info)
    log.info("distilbert test %s", info["test"])


def load_bert(cfg: Config):
    path = _paths(cfg)["bert"]
    return AutoModelForSequenceClassification.from_pretrained(path).eval(), AutoTokenizer.from_pretrained(path)


def _baseline_latency(model, texts: list[str], n: int) -> dict:
    times = []
    for text in texts[: n + 3]:
        start = time.perf_counter()
        model.predict_proba([text])
        times.append((time.perf_counter() - start) * 1000)
    times = np.array(times[3:])
    return {"p50_ms": round(float(np.median(times)), 2), "p95_ms": round(float(np.percentile(times, 95)), 2)}


def step_compress(cfg: Config, splits: Splits) -> None:
    """fp32 vs int8 DistilBERT vs the linear baseline: accuracy, size, single-request latency."""
    p = _paths(cfg)
    torch.manual_seed(cfg.seed)
    model, tokenizer = load_bert(cfg)
    qmodel = compress.quantize(model)
    y = splits.test["label"].to_numpy()
    texts = splits.test["text"].tolist()
    probs = np.load(p["probs"])
    start = time.perf_counter()
    qprobs = transformer.predict_proba(qmodel, tokenizer, texts, cfg.transformer.max_length)
    q_seconds = time.perf_counter() - start
    start = time.perf_counter()
    transformer.predict_proba(model, tokenizer, texts, cfg.transformer.max_length)
    fp_seconds = time.perf_counter() - start
    _save_probs(p["probs"], bert_int8=qprobs)
    n = cfg.compress.latency_samples
    base = joblib.load(p["baseline"])
    rows = {
        "distilbert_fp32": {
            **evaluate.metrics(y, probs["bert"].argmax(1)),
            "size_mb": round(compress.size_mb(model), 1),
            **compress.latency_ms(model, tokenizer, texts, n, cfg.transformer.max_length),
            "batch64_articles_per_s": round(len(texts) / fp_seconds, 1),
        },
        "distilbert_int8": {
            **evaluate.metrics(y, qprobs.argmax(1)),
            "size_mb": round(compress.size_mb(qmodel), 1),
            **compress.latency_ms(qmodel, tokenizer, texts, n, cfg.transformer.max_length),
            "batch64_articles_per_s": round(len(texts) / q_seconds, 1),
            "agreement_with_fp32": round(float((qprobs.argmax(1) == probs["bert"].argmax(1)).mean()), 4),
        },
        "tfidf_svm": {
            **evaluate.metrics(y, probs["baseline"].argmax(1)),
            "size_mb": round(p["baseline"].stat().st_size / 1e6, 1),
            **_baseline_latency(base, texts, n),
        },
        "torch_threads": torch.get_num_threads(),
    }
    _update_metrics(p["metrics"], "compression", rows)
    log.info("compression %s", json.dumps(rows))


def step_distill(cfg: Config, splits: Splits) -> None:
    """Teacher (int8 DistilBERT) labels unlabelled articles; the linear student retrains on gold + teacher labels."""
    p = _paths(cfg)
    model, tokenizer = load_bert(cfg)
    teacher = compress.quantize(model)  # faster for bulk labelling; agreement with fp32 is measured in step_compress
    pool = splits.train_pool[~splits.train_pool["text"].isin(set(splits.train["text"]))]
    unlabeled = stratified_take(pool, cfg.distill.unlabeled_size, cfg.seed + 1)
    cache = Path(cfg.artifacts_dir) / f"teacher_probs_{len(unlabeled)}.npy"  # labelling is the slow part; reuse it
    start = time.perf_counter()
    if cache.exists():
        tprobs = np.load(cache)
    else:
        tprobs = transformer.predict_proba(teacher, tokenizer, unlabeled["text"].tolist(), cfg.transformer.max_length)
        np.save(cache, tprobs)
    label_seconds = time.perf_counter() - start
    pseudo = distill.pseudo_label(tprobs, unlabeled["text"], cfg.distill.min_confidence)
    # Gold labels of the "unlabelled" pool are used ONLY to measure teacher quality, never for training the student.
    gold = unlabeled["label"].to_numpy()
    keep = tprobs.max(1) >= cfg.distill.min_confidence

    c = json.loads(p["metrics"].read_text())["baseline"]["best_c"]
    y = splits.test["label"].to_numpy()
    student_data = pd.concat([splits.train, pseudo], ignore_index=True)
    student = baseline.build_calibrated(c, cfg.seed).fit(student_data["text"], student_data["label"])
    student_pred = student.predict(splits.test["text"])
    # Control: the same extra articles with their TRUE labels = the ceiling distillation could reach.
    oracle_data = pd.concat([splits.train, unlabeled], ignore_index=True)
    oracle_pred = baseline.build_pipeline(c, cfg.seed).fit(oracle_data["text"], oracle_data["label"]).predict(splits.test["text"])
    base_pred = np.load(p["probs"])["baseline"].argmax(1)

    info = {
        "unlabeled_articles": len(unlabeled),
        "teacher_label_seconds": round(label_seconds, 1) if label_seconds > 5 else "cached",
        "teacher_label_accuracy": round(float((tprobs.argmax(1) == gold).mean()), 4),
        "kept_after_confidence_filter": int(keep.sum()),
        "kept_label_accuracy": round(float((tprobs.argmax(1)[keep] == gold[keep]).mean()), 4),
        "student_gold_only": evaluate.metrics(y, base_pred),
        "student_gold_plus_teacher": evaluate.metrics(y, student_pred),
        "oracle_gold_plus_true_labels": evaluate.metrics(y, oracle_pred),
        "mcnemar_gold_only_vs_student": evaluate.mcnemar(y, base_pred, student_pred),
        "student_latency": _baseline_latency(student, splits.test["text"].tolist(), cfg.compress.latency_samples),
    }
    joblib.dump(student, p["student"])
    _update_metrics(p["metrics"], "distillation", info)
    log.info("distillation %s", json.dumps(info))


def step_report(cfg: Config, splits: Splits) -> None:
    p = _paths(cfg)
    probs = np.load(p["probs"])
    y = probs["y"]
    pa, pb = probs["baseline"].argmax(1), probs["bert"].argmax(1)
    evaluate.plot_confusion(y, pa, "TF-IDF + linear SVM", p["figures"] / "confusion_baseline.png")
    evaluate.plot_confusion(y, pb, "DistilBERT (fine-tuned)", p["figures"] / "confusion_distilbert.png")
    metrics = json.loads(p["metrics"].read_text())
    evaluate.plot_learning_curve(
        metrics["baseline"]["learning_curve"], (len(splits.train), metrics["distilbert"]["test"]["accuracy"]), p["figures"] / "learning_curve.png"
    )
    table = evaluate.error_table(splits.test, probs["baseline"], probs["bert"])
    analysis = {
        "accuracy_ci95": {"baseline": evaluate.bootstrap_ci(y, pa, seed=cfg.seed), "distilbert": evaluate.bootstrap_ci(y, pb, seed=cfg.seed)},
        "mcnemar_baseline_vs_bert": evaluate.mcnemar(y, pa, pb),
        "confusion_pairs": {"baseline": evaluate.confusion_pairs(y, pa), "distilbert": evaluate.confusion_pairs(y, pb)},
        **evaluate.error_breakdown(table),
    }
    _update_metrics(p["metrics"], "analysis", analysis)
    errors = evaluate.sample_errors(table)
    errors.to_csv(p["reports"] / "top_confident_errors.csv", index=False)
    log.info("analysis %s", json.dumps(analysis, indent=1))


STEPS = {"baseline": step_baseline, "bert": step_bert, "compress": step_compress, "distill": step_distill, "report": step_report}


def run(cfg: Config, steps: list[str]) -> None:
    splits = load_splits(cfg)
    for name in steps:
        start = time.perf_counter()
        STEPS[name](cfg, splits)
        log.info("step %s finished in %.0f s", name, time.perf_counter() - start)
