# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import json
import os

import torch

_REQUIRED_META_KEYS = frozenset(
    {"layer_index", "target", "method", "shard_id", "num_channels"}
)
_VALID_METHODS = ("max_abs", "dynamic_range")


def compute_risk_scores(
    tensor: torch.Tensor,
    method: str = "max_abs",
    channel_dim: int = -1,
) -> torch.Tensor:
    if method not in _VALID_METHODS:
        raise ValueError(
            f"Unknown method {method!r}, expected one of {_VALID_METHODS}"
        )

    # Normalise channel_dim to positive index
    ndim = tensor.ndim
    cdim = channel_dim % ndim

    # Dims to reduce over (everything except the channel dim)
    reduce_dims = [d for d in range(ndim) if d != cdim]

    abs_t = tensor.abs()

    if method == "max_abs":
        scores = abs_t.amax(dim=reduce_dims)
    else:  # dynamic_range
        eps = 1e-10
        max_abs = abs_t.amax(dim=reduce_dims)

        # min nonzero abs per channel: mask zeros with inf, then take min
        masked = abs_t.where(abs_t > 0, torch.tensor(float("inf"),
                             device=tensor.device, dtype=tensor.dtype))
        min_nonzero = masked.amin(dim=reduce_dims)
        # If all elements are zero the min stays inf; replace with eps
        min_nonzero = min_nonzero.where(min_nonzero.isfinite(),
                                        torch.tensor(eps,
                                        device=tensor.device,
                                        dtype=tensor.dtype))
        scores = max_abs / (min_nonzero + eps)

    return scores.to(dtype=torch.float32)


def save_risk_scores(
    scores: torch.Tensor,
    path: str,
    metadata: dict,
) -> None:
    missing = _REQUIRED_META_KEYS - set(metadata)
    if missing:
        raise ValueError(f"Missing required metadata keys: {missing}")

    scores_cpu = scores.detach().cpu().to(dtype=torch.float32)

    data = {**metadata, "scores": scores_cpu.tolist()}

    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)

    with open(path, "w") as f:
        json.dump(data, f)


def load_risk_scores(path: str) -> tuple[torch.Tensor, dict]:
    with open(path) as f:
        data = json.load(f)

    scores = torch.tensor(data.pop("scores"), dtype=torch.float32)
    return scores, data
