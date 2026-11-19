"""Knowledge distillation into the linear model: the transformer labels unlabelled articles.

The fine-tuned DistilBERT (teacher) only saw the small labelled subset, but it generalises well.
We let it label a much larger pool of *unlabelled* articles and train the cheap TF-IDF + SVM
(student) on gold labels + teacher labels. The student keeps its ~1 ms CPU latency.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def pseudo_label(probs: np.ndarray, texts: pd.Series, min_confidence: float) -> pd.DataFrame:
    """Keep only teacher predictions above a confidence threshold (fewer wrong labels for the student)."""
    keep = probs.max(axis=1) >= min_confidence
    return pd.DataFrame({"text": texts.to_numpy()[keep], "label": probs.argmax(axis=1)[keep]})
