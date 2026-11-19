"""Synthetic data and a tiny randomly initialised DistilBERT so tests run offline in seconds."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from transformers import DistilBertConfig, DistilBertForSequenceClassification, DistilBertTokenizerFast

TOPIC_WORDS = [
    ["election", "minister", "troops", "talks"],
    ["match", "goal", "season", "coach"],
    ["shares", "profit", "market", "oil"],
    ["software", "space", "internet", "chip"],
]


@pytest.fixture(scope="session")
def toy_df() -> pd.DataFrame:
    """Each article mixes topic words with shared filler words, so the task is learnable but not trivial."""
    rng = np.random.default_rng(0)
    filler = ["the", "a", "new", "on", "says", "report"]
    rows = []
    for label, words in enumerate(TOPIC_WORDS):
        for _ in range(40):
            tokens = list(rng.choice(words, 3)) + list(rng.choice(filler, 4))
            rng.shuffle(tokens)
            rows.append({"text": " ".join(tokens), "label": label})
    return pd.DataFrame(rows).sample(frac=1, random_state=0).reset_index(drop=True)


@pytest.fixture(scope="session")
def tiny_bert():
    vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "the", "a", "new", "on", "says", "report"]
    vocab += [w for words in TOPIC_WORDS for w in words]
    tokenizer = DistilBertTokenizerFast(vocab={w: i for i, w in enumerate(vocab)})
    config = DistilBertConfig(vocab_size=len(vocab), dim=32, n_layers=1, n_heads=2, hidden_dim=64, max_position_embeddings=64, num_labels=4)
    model = DistilBertForSequenceClassification(config)
    return model, tokenizer
