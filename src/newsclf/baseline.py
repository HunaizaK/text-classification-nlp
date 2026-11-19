"""Classical baseline: TF-IDF (word + character n-grams) + linear SVM."""

from __future__ import annotations

import logging
import time

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC

from newsclf.config import Config
from newsclf.data import Splits, stratified_take

log = logging.getLogger(__name__)


def build_pipeline(c: float = 1.0, seed: int = 0) -> Pipeline:
    features = FeatureUnion(
        [
            # Word uni+bigrams capture topical vocabulary ("oil prices", "world cup").
            ("word", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, strip_accents="unicode")),
            # Character n-grams inside word boundaries handle rare words, typos and morphology.
            ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2, sublinear_tf=True, max_features=200_000)),
        ]
    )
    return Pipeline([("tfidf", features), ("svm", LinearSVC(C=c, random_state=seed))])


def build_calibrated(c: float = 1.0, seed: int = 0) -> Pipeline:
    """LinearSVC outputs margins, not probabilities; sigmoid calibration (3-fold CV) gives usable confidences."""
    model = build_pipeline(c, seed)
    model.steps[-1] = ("svm", CalibratedClassifierCV(model.steps[-1][1], method="sigmoid", cv=3))
    return model


def select_c(train: pd.DataFrame, val: pd.DataFrame, grid: list[float], seed: int) -> tuple[float, dict[float, float]]:
    scores = {}
    for c in grid:
        model = build_pipeline(c, seed).fit(train["text"], train["label"])
        scores[c] = f1_score(val["label"], model.predict(val["text"]), average="macro")
        log.info("C=%-5g val macro-F1=%.4f", c, scores[c])
    return max(scores, key=scores.get), scores


def train_baseline(splits: Splits, cfg: Config) -> tuple[Pipeline, dict]:
    """Pick C on the validation set, then refit with probability calibration for confidence scores."""
    best_c, scores = select_c(splits.train, splits.val, cfg.baseline.c_grid, cfg.seed)
    start = time.perf_counter()
    model = build_calibrated(best_c, cfg.seed)
    model.fit(splits.train["text"], splits.train["label"])
    info = {"best_c": best_c, "val_macro_f1_by_c": scores, "train_seconds": round(time.perf_counter() - start, 2)}
    return model, info


def learning_curve(splits: Splits, cfg: Config, c: float) -> list[dict]:
    """Test accuracy of the baseline as the amount of training data grows (beyond the CPU subset)."""
    rows = []
    for n in cfg.baseline.learning_curve_sizes:
        n = min(n, len(splits.train_pool))
        subset = stratified_take(splits.train_pool, n, cfg.seed)
        start = time.perf_counter()
        model = build_pipeline(c, cfg.seed).fit(subset["text"], subset["label"])
        pred = model.predict(splits.test["text"])
        rows.append(
            {
                "train_size": n,
                "accuracy": round(accuracy_score(splits.test["label"], pred), 4),
                "macro_f1": round(f1_score(splits.test["label"], pred, average="macro"), 4),
                "fit_seconds": round(time.perf_counter() - start, 1),
            }
        )
        log.info("learning curve n=%d acc=%.4f", n, rows[-1]["accuracy"])
    return rows


def top_features(model: Pipeline, labels: list[str], k: int = 12) -> dict[str, list[str]]:
    """Highest-weight word n-grams per class (averaged over the calibration folds)."""
    svm = model.named_steps["svm"]
    coefs = np.mean([cc.estimator.coef_ for cc in svm.calibrated_classifiers_], axis=0)
    word = model.named_steps["tfidf"].transformer_list[0][1]
    n_word = len(word.vocabulary_)
    names = word.get_feature_names_out()
    return {label: [names[i] for i in np.argsort(coefs[j, :n_word])[::-1][:k]] for j, label in enumerate(labels)}
