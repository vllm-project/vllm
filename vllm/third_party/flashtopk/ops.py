# Copyright (c) 2026 the FlashTopK authors. Licensed under the Apache License, Version 2.0.
"""torch.ops.vllm_flashtopk.select: the selection as a custom op, so torch.compile traces through it (the fake gives the
output shapes) and treats it as one opaque call."""

from __future__ import annotations

from typing import Optional, Tuple

import torch

from .topk import select as _select


@torch.library.custom_op("vllm_flashtopk::select", mutates_args=())
def select(q: torch.Tensor, kv: torch.Tensor, weights: torch.Tensor, ks: torch.Tensor, ke: torch.Tensor,
           kv_scales: Optional[torch.Tensor], topk: int, pruning: Optional[str] = None,
           chunk_size: Optional[int] = None, return_scores: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
    """flashtopk.select as a custom op: (scores, indices [num_q, topk] int32), the scores [num_q, topk] float32 with
    return_scores and empty, [num_q, 0], without."""
    if return_scores:
        return _select(q, kv, weights, ks, ke, kv_scales, topk, pruning, chunk_size=chunk_size, return_scores=True)
    indices = _select(q, kv, weights, ks, ke, kv_scales, topk, pruning, chunk_size=chunk_size)
    return q.new_empty(q.shape[0], 0, dtype=torch.float32), indices


@select.register_fake
def _(q, kv, weights, ks, ke, kv_scales, topk, pruning=None, chunk_size=None, return_scores=False):
    num_q = q.shape[0]
    return (q.new_empty(num_q, topk if return_scores else 0, dtype=torch.float32),
            q.new_empty(num_q, topk, dtype=torch.int32))
