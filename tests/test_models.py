import numpy as np
import torch

from newsclf.baseline import build_pipeline
from newsclf.compress import quantize, size_mb
from newsclf.config import TransformerConfig
from newsclf.distill import pseudo_label
from newsclf.evaluate import bootstrap_ci, confusion_pairs, mcnemar, metrics
from newsclf.transformer import TextDataset, fine_tune, predict_proba


def test_baseline_learns_toy_topics(toy_df):
    model = build_pipeline(c=1.0).fit(toy_df["text"][:120], toy_df["label"][:120])
    assert (model.predict(toy_df["text"][120:]) == toy_df["label"][120:]).mean() > 0.9


def test_text_dataset_truncates_and_attaches_labels(tiny_bert):
    _, tokenizer = tiny_bert
    ds = TextDataset(["the match goal " * 50, "oil"], [1, 2], tokenizer, max_length=16)
    assert len(ds) == 2
    assert len(ds[0]["input_ids"]) == 16
    assert ds[1]["labels"] == 2


def test_fine_tune_reduces_loss_and_predicts_probabilities(tiny_bert, toy_df, tmp_path):
    model, tokenizer = tiny_bert
    before = predict_proba(model, tokenizer, toy_df["text"].tolist())
    cfg = TransformerConfig(checkpoint="-", max_length=32, epochs=25, batch_size=16, learning_rate=2e-3, weight_decay=0.0, warmup_ratio=0.0)
    trainer, info = fine_tune(model, tokenizer, toy_df[:128], toy_df[128:], cfg, tmp_path, seed=0)
    probs = predict_proba(trainer.model, tokenizer, toy_df["text"].tolist())
    assert probs.shape == (len(toy_df), 4)
    np.testing.assert_allclose(probs.sum(1), 1, rtol=1e-5)
    acc_before = (before.argmax(1) == toy_df["label"]).mean()
    acc_after = (probs.argmax(1) == toy_df["label"]).mean()
    assert acc_after > max(acc_before, 0.8)
    assert any("eval_macro_f1" in row for row in info["history"])


def test_predict_proba_order_is_preserved(tiny_bert):
    """Inference sorts by length for speed; outputs must still align with the input order."""
    model, tokenizer = tiny_bert
    texts = ["oil", "the match goal season coach a new", "chip space", "election"]
    batched = predict_proba(model, tokenizer, texts, batch_size=2)
    single = np.vstack([predict_proba(model, tokenizer, [t]) for t in texts])
    np.testing.assert_allclose(batched, single, atol=1e-5)


def test_quantization_shrinks_linear_layers_and_keeps_outputs_close(tiny_bert):
    model, tokenizer = tiny_bert
    q = quantize(model)
    assert size_mb(q) < size_mb(model)
    assert any("quantized" in type(m).__module__ for m in q.modules())
    assert not any("quantized" in type(m).__module__ for m in model.modules())  # original untouched
    texts = ["the match goal", "oil market shares"]
    with torch.inference_mode():
        np.testing.assert_allclose(predict_proba(q, tokenizer, texts), predict_proba(model, tokenizer, texts), atol=0.05)


def test_pseudo_label_filters_low_confidence():
    import pandas as pd

    probs = np.array([[0.95, 0.05, 0, 0], [0.4, 0.6, 0, 0], [0, 0, 0.05, 0.95]])
    out = pseudo_label(probs, pd.Series(["a", "b", "c"]), min_confidence=0.9)
    assert out["text"].tolist() == ["a", "c"]
    assert out["label"].tolist() == [0, 3]


def test_mcnemar_counts_discordant_pairs():
    y = np.array([0, 0, 0, 0, 1, 1])
    a = np.array([0, 0, 0, 1, 1, 1])  # 5 right
    b = np.array([0, 1, 1, 0, 1, 0])  # 3 right
    res = mcnemar(y, a, b)
    assert (res["only_a_correct"], res["only_b_correct"]) == (3, 1)
    assert 0 < res["p_value"] <= 1


def test_metrics_ci_and_confusion_pairs():
    y = np.array([0, 1, 2, 3] * 25)
    pred = y.copy()
    pred[:10] = 2
    m = metrics(y, pred)
    assert m["accuracy"] == 0.92
    lo, hi = bootstrap_ci(y, pred)
    assert lo <= 0.92 <= hi
    assert confusion_pairs(y, pred)[0] == {"true": "World", "predicted": "Business", "count": 3}


def test_tiny_tokenizer_knows_its_vocabulary(tiny_bert):
    """Guards the fixture itself: an all-[UNK] tokenizer makes every model test meaningless."""
    _, tokenizer = tiny_bert
    assert tokenizer.unk_token_id not in tokenizer("oil market goal")["input_ids"]
