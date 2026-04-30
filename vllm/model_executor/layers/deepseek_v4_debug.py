# SPDX-License-Identifier: Apache-2.0
"""Debug-only dump helpers for DeepSeek-V4 accuracy triage.

All functionality is disabled unless VLLM_DSV4_DUMP_ROOT is set.
"""

from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Any

import torch

_COUNTS: dict[str, int] = {}
_LAYER_RE = re.compile(r"(?:^|\.)layers\.(\d+)(?:\.|$)")


def _truthy(value: str | None) -> bool:
    return value not in (None, "", "0", "false", "False", "no", "No")


def enabled() -> bool:
    return _truthy(os.environ.get("VLLM_DSV4_DUMP_ROOT"))


def side() -> str:
    return os.environ.get("VLLM_DSV4_DUMP_SIDE", "fp8")


def _rank() -> str:
    return os.environ.get("RANK") or os.environ.get("LOCAL_RANK") or "0"


def layer_from_name(name: str) -> int | None:
    m = _LAYER_RE.search(name)
    return int(m.group(1)) if m else None


def _selected_layer(name: str, layer_idx: int | None) -> bool:
    spec = os.environ.get("VLLM_DSV4_DUMP_LAYERS", "")
    if not spec:
        return True
    if layer_idx is None:
        return True
    vals: set[int] = set()
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        if "-" in item:
            lo, hi = item.split("-", 1)
            vals.update(range(int(lo), int(hi) + 1))
        else:
            vals.add(int(item))
    return layer_idx in vals


def _safe_float(x: torch.Tensor, fn: str) -> float | None:
    try:
        if fn == "mean":
            return float(x.mean().item())
        if fn == "std":
            return float(x.std(unbiased=False).item())
        if fn == "amax":
            return float(x.abs().amax().item())
        if fn == "amin":
            return float(x.amin().item())
        if fn == "max":
            return float(x.amax().item())
    except Exception:
        return None
    return None


def _summary(tensor: torch.Tensor) -> dict[str, Any]:
    t = tensor.detach()
    out: dict[str, Any] = {
        "shape": list(t.shape),
        "dtype": str(t.dtype),
        "device": str(t.device),
        "numel": int(t.numel()),
    }
    if t.numel() == 0:
        out.update({"finite": True, "nan_count": 0, "inf_count": 0})
        return out
    try:
        tf = t.float()
    except Exception:
        tf = t.to(torch.float32)
    try:
        finite = torch.isfinite(tf)
        out["finite"] = bool(finite.all().item())
        out["nan_count"] = int(torch.isnan(tf).sum().item())
        out["inf_count"] = int(torch.isinf(tf).sum().item())
    except Exception:
        out["finite"] = None
        out["nan_count"] = None
        out["inf_count"] = None
    out["mean"] = _safe_float(tf, "mean")
    out["std"] = _safe_float(tf, "std")
    out["amax_abs"] = _safe_float(tf, "amax")
    out["min"] = _safe_float(tf, "amin")
    out["max"] = _safe_float(tf, "max")
    if tf.ndim >= 1 and tf.shape[-1] > 0 and tf.numel() <= 2_000_000:
        try:
            row = tf.reshape(-1, tf.shape[-1])[-1]
            k = min(8, row.numel())
            vals, idx = torch.topk(row, k=k)
            out["last_row_top_ids"] = [int(x) for x in idx.cpu().tolist()]
            out["last_row_top_vals"] = [float(x) for x in vals.cpu().tolist()]
        except Exception:
            pass
    return out


def dump_tensor(name: str, tensor: torch.Tensor | None, *, layer_idx: int | None = None,
                note: str | None = None, max_writes: int | None = None) -> None:
    if not enabled() or tensor is None:
        return
    if layer_idx is None:
        layer_idx = layer_from_name(name)
    if not _selected_layer(name, layer_idx):
        return
    key = f"{side()}:{_rank()}:{name}"
    if max_writes is None:
        max_writes = int(os.environ.get("VLLM_DSV4_DUMP_MAX_WRITES", "16"))
    count = _COUNTS.get(key, 0)
    if count >= max_writes:
        return
    _COUNTS[key] = count + 1
    root = Path(os.environ["VLLM_DSV4_DUMP_ROOT"]) / side()
    root.mkdir(parents=True, exist_ok=True)
    rec: dict[str, Any] = {
        "ts": time.time(),
        "side": side(),
        "rank": _rank(),
        "name": name,
        "layer_idx": layer_idx,
        "write_idx": count,
        "note": note,
    }
    rec.update(_summary(tensor))
    with (root / f"rank_{_rank()}_summary.jsonl").open("a") as f:
        f.write(json.dumps(rec, ensure_ascii=False, allow_nan=True) + "\n")
    if _truthy(os.environ.get("VLLM_DSV4_DUMP_FULL_TENSOR")):
        import numpy as np
        safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", name)
        arr = tensor.detach().float().cpu().numpy()
        np.savez_compressed(root / f"rank_{_rank()}_{count:04d}_{safe}.npz", tensor=arr)
