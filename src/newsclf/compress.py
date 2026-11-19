"""Post-training dynamic int8 quantization of the fine-tuned model: size, latency, accuracy."""

from __future__ import annotations

import copy
import io
import time
import warnings

import numpy as np
import torch
from torch.ao.quantization import quantize_dynamic


def quantize(model: torch.nn.Module) -> torch.nn.Module:
    """Store every nn.Linear weight as int8; activations are quantized on the fly at inference.

    Linear layers hold ~85% of DistilBERT's weights and most of its FLOPs. Embeddings and
    LayerNorm stay fp32. No calibration data is needed, which is why this is called "dynamic".
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # torch marks the eager quantization API deprecated in favour of torchao
        return quantize_dynamic(copy.deepcopy(model).eval(), {torch.nn.Linear}, dtype=torch.qint8)


def size_mb(model: torch.nn.Module) -> float:
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    return buffer.getbuffer().nbytes / 1e6


@torch.inference_mode()
def latency_ms(model: torch.nn.Module, tokenizer, texts: list[str], n: int, max_length: int = 128) -> dict:
    """Single-article latency (what one API request sees): median and p95 over n articles."""
    model.eval()
    times = []
    for text in texts[: n + 3]:
        batch = tokenizer([text], truncation=True, max_length=max_length, return_tensors="pt")
        start = time.perf_counter()
        model(**batch)
        times.append((time.perf_counter() - start) * 1000)
    times = np.array(times[3:])  # first calls include one-off allocation costs
    return {"p50_ms": round(float(np.median(times)), 1), "p95_ms": round(float(np.percentile(times, 95)), 1)}
