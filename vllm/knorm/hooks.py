# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-HUST project
"""
Runtime hook functions for Knorm.

These are called *explicitly* from the vllm-hust execution loop
(not via monkey-patch). Each function is a single-purpose hook
that is invoked at the appropriate point in the model runner
or scheduler lifecycle.
"""

from __future__ import annotations

import torch

from vllm.knorm.attention_backend import clear_pending_norms, get_pending_norms


def collect_knorm_scores(runner, input_batch) -> None:
    """Collect key L2 norms after model forward and aggregate per-block.

    Called from ``GPUModelRunner.execute_model`` after the model forward
    completes but before ``sample_tokens``.

    Stores results in ``runner._knorm_scores`` as:
        {req_id: [(block_idx, min_norm), ...]}
    """
    if not hasattr(runner, "kv_cache_config") or runner.kv_cache_config is None:
        runner._knorm_scores = None
        return

    norms_list = get_pending_norms()
    if not norms_list:
        runner._knorm_scores = None
        return

    # Stack norms from all layers, average. Shape: [num_tokens]
    all_norms = torch.stack(norms_list).float().mean(dim=0)
    clear_pending_norms()

    token_norms = all_norms.cpu().numpy()
    num_tokens = len(token_norms)

    req_ids = input_batch.req_ids
    num_reqs = input_batch.num_reqs
    num_scheduled = input_batch.num_tokens_no_spec[:num_reqs]
    positions_np = runner.positions[:num_tokens].cpu().numpy()
    block_size = runner.kv_cache_config.kv_cache_groups[0].kv_cache_spec.block_size

    # Aggregate: per (request_id, block_idx) → min norm (lower = more important)
    scores: dict[str, dict[int, float]] = {}
    idx = 0
    for req_id, num_toks in zip(req_ids, num_scheduled, strict=False):
        nt = int(num_toks)
        req_scores: dict[int, float] = {}
        for _ in range(nt):
            if idx >= num_tokens:
                break
            pos = int(positions_np[idx])
            block_idx = pos // block_size
            norm_val = float(token_norms[idx])
            req_scores[block_idx] = min(req_scores.get(block_idx, norm_val), norm_val)
            idx += 1
        if req_scores:
            scores[req_id] = req_scores

    if scores:
        formatted: dict[str, list[tuple[int, float]]] = {}
        for rid, bdict in scores.items():
            formatted[rid] = list(bdict.items())
        runner._knorm_scores = formatted
    else:
        runner._knorm_scores = None


def attach_knorm_scores(runner, result) -> None:
    """Attach knorm block scores to the model runner output.

    Called from ``GPUModelRunner.sample_tokens`` before returning.

    The scores are later read by ``Scheduler.update_from_output`` and
    routed to ``KnormFullAttentionManager``.
    """
    knorm_scores = getattr(runner, "_knorm_scores", None)
    if not knorm_scores or result is None:
        return

    result.knorm_block_scores = knorm_scores  # type: ignore[attr-defined]
