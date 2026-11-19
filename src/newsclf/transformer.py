"""Fine-tune DistilBERT for 4-way topic classification with the Hugging Face Trainer."""

from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Trainer,
    TrainingArguments,
    set_seed,
)

from newsclf import LABELS
from newsclf.config import Config, TransformerConfig
from newsclf.data import Splits

log = logging.getLogger(__name__)


class TextDataset(torch.utils.data.Dataset):
    """Tokenised once up front; padding happens per batch in the collator (dynamic padding)."""

    def __init__(self, texts: list[str], labels: list[int] | None, tokenizer: PreTrainedTokenizerBase, max_length: int):
        self.encodings = tokenizer(texts, truncation=True, max_length=max_length)
        self.labels = labels

    def __len__(self) -> int:
        return len(self.encodings["input_ids"])

    def __getitem__(self, i: int) -> dict:
        item = {k: v[i] for k, v in self.encodings.items()}
        if self.labels is not None:
            item["labels"] = int(self.labels[i])
        return item


def compute_metrics(eval_pred) -> dict:
    logits, labels = eval_pred
    pred = np.argmax(logits, axis=-1)
    return {"accuracy": accuracy_score(labels, pred), "macro_f1": f1_score(labels, pred, average="macro")}


def load_pretrained(checkpoint: str) -> tuple[PreTrainedTokenizerBase, PreTrainedModel]:
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModelForSequenceClassification.from_pretrained(
        checkpoint,
        num_labels=len(LABELS),
        id2label=dict(enumerate(LABELS)),
        label2id={label: i for i, label in enumerate(LABELS)},
    )
    return tokenizer, model


def fine_tune(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    train: pd.DataFrame,
    val: pd.DataFrame,
    tcfg: TransformerConfig,
    output_dir: Path,
    seed: int,
) -> tuple[Trainer, dict]:
    set_seed(seed)
    train_ds = TextDataset(train["text"].tolist(), train["label"].tolist(), tokenizer, tcfg.max_length)
    val_ds = TextDataset(val["text"].tolist(), val["label"].tolist(), tokenizer, tcfg.max_length)
    args = TrainingArguments(
        output_dir=str(output_dir / "checkpoints"),
        num_train_epochs=tcfg.epochs,
        per_device_train_batch_size=tcfg.batch_size,
        per_device_eval_batch_size=tcfg.batch_size * 2,
        learning_rate=tcfg.learning_rate,
        weight_decay=tcfg.weight_decay,
        warmup_steps=tcfg.warmup_ratio,  # transformers 5: a float in [0, 1) is a fraction of total steps
        lr_scheduler_type="linear",
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        logging_steps=25,
        report_to="none",
        use_cpu=True,
        seed=seed,
        data_seed=seed,
        dataloader_num_workers=0,
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics,
    )
    start = time.perf_counter()
    trainer.train()
    seconds = time.perf_counter() - start
    history = [row for row in trainer.state.log_history if "loss" in row or "eval_macro_f1" in row]
    return trainer, {"train_seconds": round(seconds, 1), "history": history}


@torch.inference_mode()
def predict_proba(model: torch.nn.Module, tokenizer: PreTrainedTokenizerBase, texts: list[str], max_length: int = 128, batch_size: int = 64) -> np.ndarray:
    """Batched inference. Texts are sorted by length so each batch pads to a similar length."""
    model.eval()
    order = np.argsort([len(t) for t in texts])
    probs = np.zeros((len(texts), model.config.num_labels), dtype=np.float32)
    for start in range(0, len(texts), batch_size):
        idx = order[start : start + batch_size]
        batch = tokenizer([texts[i] for i in idx], truncation=True, max_length=max_length, padding=True, return_tensors="pt")
        probs[idx] = torch.softmax(model(**batch).logits, dim=-1).numpy()
    return probs


def train_transformer(splits: Splits, cfg: Config) -> tuple[PreTrainedModel, PreTrainedTokenizerBase, dict]:
    out = Path(cfg.artifacts_dir) / "distilbert"
    tokenizer, model = load_pretrained(cfg.transformer.checkpoint)
    trainer, info = fine_tune(model, tokenizer, splits.train, splits.val, cfg.transformer, out, cfg.seed)
    trainer.save_model(str(out))
    tokenizer.save_pretrained(str(out))
    return trainer.model, tokenizer, info
