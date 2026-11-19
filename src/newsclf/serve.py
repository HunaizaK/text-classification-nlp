"""FastAPI service. Backend chosen with NEWSCLF_BACKEND:

- `linear` (default): distilled TF-IDF + SVM student, ~1 ms per article, no GPU or torch needed at runtime
- `transformer`: int8-quantized DistilBERT, most accurate, tens of ms per article on CPU

Run: uvicorn newsclf.serve:app --port 8000
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from newsclf import LABELS
from newsclf.data import clean_text

ARTIFACTS = Path(os.environ.get("NEWSCLF_ARTIFACTS", "artifacts"))
MAX_BATCH = 64


class PredictRequest(BaseModel):
    texts: list[str] = Field(min_length=1, max_length=MAX_BATCH)


class Prediction(BaseModel):
    label: str
    confidence: float
    probabilities: dict[str, float]


class PredictResponse(BaseModel):
    backend: str
    predictions: list[Prediction]


class Predictor:
    def __init__(self, backend: str, artifacts: Path):
        self.backend = backend
        if backend == "linear":
            path = artifacts / "student.joblib"
            if not path.exists():
                path = artifacts / "baseline.joblib"
            self.model = joblib.load(path)
            self._proba = lambda texts: self._linear_proba(texts)
        elif backend == "transformer":
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            from newsclf.compress import quantize
            from newsclf.transformer import predict_proba

            model = AutoModelForSequenceClassification.from_pretrained(artifacts / "distilbert")
            tokenizer = AutoTokenizer.from_pretrained(artifacts / "distilbert")
            qmodel = quantize(model)
            self._proba = lambda texts: predict_proba(qmodel, tokenizer, texts)
        else:
            raise ValueError(f"Unknown backend {backend!r}")

    def _linear_proba(self, texts: list[str]) -> np.ndarray:
        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(texts)
        # Uncalibrated LinearSVC: softmax over margins gives a ranking-consistent score, not a calibrated probability.
        margins = self.model.decision_function(texts)
        exp = np.exp(margins - margins.max(axis=1, keepdims=True))
        return exp / exp.sum(axis=1, keepdims=True)

    def predict(self, texts: list[str]) -> list[Prediction]:
        probs = self._proba([clean_text(t) for t in texts])
        return [
            Prediction(
                label=LABELS[int(row.argmax())],
                confidence=round(float(row.max()), 4),
                probabilities={label: round(float(p), 4) for label, p in zip(LABELS, row, strict=True)},
            )
            for row in probs
        ]


@lru_cache(maxsize=1)
def get_predictor() -> Predictor:
    return Predictor(os.environ.get("NEWSCLF_BACKEND", "linear"), ARTIFACTS)


app = FastAPI(title="News topic classifier", version="1.0.0")


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "backend": get_predictor().backend}


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest) -> PredictResponse:
    if any(not t.strip() for t in request.texts):
        raise HTTPException(status_code=422, detail="Empty text")
    predictor = get_predictor()
    return PredictResponse(backend=predictor.backend, predictions=predictor.predict(request.texts))
