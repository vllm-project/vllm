# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Startup warmup for the optional hybrid NVFP4 lm-head path."""

from __future__ import annotations

from time import perf_counter
from typing import Any

import torch

import vllm.envs as envs
from vllm.logger import init_logger
from vllm.model_executor.layers.hybrid_nvfp4_lm_head import (
    HybridNvfp4LmHead,
    autotune_hybrid_nvfp4_lm_head,
    autotune_row_buckets,
    get_hybrid_nvfp4_lm_head,
    warmup_hybrid_nvfp4_lm_head_kernels,
)
from vllm.triton_utils import HAS_TRITON

logger = init_logger(__name__)


@torch.inference_mode()
def _warmup_compact_topk_sampler(worker: Any) -> None:
    """JIT the compact top-k/rejection kernels before graph capture."""
    if not HAS_TRITON or worker.device.type != "cuda":
        return

    from vllm.v1.worker.gpu.sample.compact_topk import (
        pack_topk_pairs,
        sample_compact_topk_pairs,
        select_compact_topk_pairs,
    )
    from vllm.v1.worker.gpu.sample.gumbel import gumbel_sample_compact
    from vllm.v1.worker.gpu.spec_decode.rejection_sampler import (
        _compact_rejection_sample_kernel,
        _prepare_compact_rejection_candidates,
        _prepare_compact_rejection_indices,
    )

    device = worker.device
    # ``_get_local_topk`` refines a wider candidate set but returns only the
    # requested top-k values.  Thus the pack kernel sees ``top_k`` (20 by
    # default), not the configured coarse candidate width (128), and TP=2
    # merges 40 pairs rather than 256.  Warm the actual constexpr widths;
    # otherwise the first real request still incurs a Triton JIT pause.
    for top_k in (1, 8, 20, 32, 64):
        local_values = torch.zeros(
            (1, top_k), dtype=torch.float32, device=device
        )
        local_ids = torch.arange(
            top_k, dtype=torch.int64, device=device
        ).view(1, -1)
        pack_topk_pairs(local_values, local_ids, 0)
        gathered_pairs = torch.zeros(
            (1, 2 * top_k, 2), dtype=torch.float32, device=device
        )
        select_compact_topk_pairs(gathered_pairs, top_k, 0.95)

    candidate_logits = torch.zeros((1, 20), dtype=torch.float32, device=device)
    candidate_ids = torch.arange(20, dtype=torch.int64, device=device).view(1, -1)
    expanded = torch.zeros((1,), dtype=torch.int64, device=device)
    seeds = torch.ones((1,), dtype=torch.int64, device=device)
    positions = torch.zeros((1,), dtype=torch.int64, device=device)
    temperatures = torch.ones((1,), dtype=torch.float32, device=device)
    sample_compact_topk_pairs(
        torch.zeros((1, 40, 2), dtype=torch.float32, device=device),
        20,
        0.95,
        expanded,
        seeds,
        positions,
    )
    gumbel_sample_compact(
        candidate_logits,
        candidate_ids,
        expanded,
        temperatures,
        seeds,
        positions,
    )
    _prepare_compact_rejection_candidates(
        candidate_logits,
        candidate_ids,
        torch.zeros((1,), dtype=torch.int64, device=device),
    )
    # The compact MTP layout is regular, so compile the direct index builder
    # before graph capture as well.  Otherwise the first real rejection step
    # can trigger a Triton JIT pause even though the sampler kernels are warm.
    _prepare_compact_rejection_indices(
        torch.tensor([0, 3], dtype=torch.int32, device=device),
        num_draft_tokens=2,
        num_reqs=1,
        device=device,
    )

    spec_len = 2
    sampled = torch.full(
        (1, spec_len + 1), -1, dtype=torch.int64, device=device
    )
    _compact_rejection_sample_kernel[(1,)](
        sampled,
        torch.tensor([1], dtype=torch.int32, device=device),
        torch.zeros((1,), dtype=torch.int64, device=device),
        torch.ones((1,), dtype=torch.float32, device=device),
        torch.zeros((1,), dtype=torch.int64, device=device),
        torch.zeros((1,), dtype=torch.int64, device=device),
        expanded,
        positions,
        seeds,
        spec_len,
    )


def _collect_lm_heads(
    model: torch.nn.Module,
) -> dict[int, tuple[torch.nn.Module, HybridNvfp4LmHead]]:
    heads: dict[int, tuple[torch.nn.Module, HybridNvfp4LmHead]] = {}
    for module in model.modules():
        state = get_hybrid_nvfp4_lm_head(module)
        if state is not None:
            heads[id(state)] = (module, state)
    return heads


def _row_shapes(worker: Any) -> tuple[int, ...]:
    config = worker.vllm_config
    max_rows = int(config.scheduler_config.max_num_seqs)
    speculative_config = config.speculative_config
    if speculative_config is not None:
        max_rows *= speculative_config.num_speculative_tokens + 1
    max_rows = min(
        max(1, max_rows),
        max(1, envs.VLLM_HYBRID_NVFP4_LM_HEAD_MAX_AUTOTUNE_ROWS),
    )
    configured_max_rows = envs.VLLM_HYBRID_NVFP4_LM_HEAD_MAX_ROWS
    if configured_max_rows > 0:
        max_rows = min(max_rows, configured_max_rows)

    capture_sizes = config.compilation_config.cudagraph_capture_sizes or []
    shapes = sorted({int(size) for size in capture_sizes if 0 < int(size) <= max_rows})
    if not shapes:
        shapes = list(autotune_row_buckets(max_rows))
    if 1 not in shapes:
        shapes.insert(0, 1)
    return tuple(shapes)


def hybrid_nvfp4_lm_head_warmup(worker: Any) -> None:
    """Tune and JIT the hybrid lm-head before CUDA graph capture."""
    if not envs.VLLM_HYBRID_NVFP4_LM_HEAD:
        return

    heads = _collect_lm_heads(worker.get_model())
    speculator = getattr(getattr(worker, "model_runner", None), "speculator", None)
    draft_model = getattr(speculator, "model", None)
    if isinstance(draft_model, torch.nn.Module):
        heads.update(_collect_lm_heads(draft_model))
    if not heads:
        logger.debug("No prepared NVFP4 lm-head was found; skipping warmup.")
        return

    started = perf_counter()
    shapes = _row_shapes(worker)
    for layer, state in heads.values():
        # Autotune materializes a BF16 coarse-logit matrix.  Keep that
        # profile-only allocation below the same explicit budget used by the
        # sampler workspace; otherwise a large vocabulary can evict the KV
        # cache before graph capture even starts.
        max_warmup_bytes = envs.VLLM_HYBRID_NVFP4_LM_HEAD_MAX_WARMUP_BYTES
        if max_warmup_bytes > 0:
            max_rows_by_memory = max_warmup_bytes // (state.output_size * 2)
            if max_rows_by_memory < 1:
                logger.warning_once(
                    "Skipping hybrid NVFP4 lm-head warmup for output size %d: "
                    "the configured memory budget (%d bytes) is smaller than "
                    "one coarse-logit row.",
                    state.output_size,
                    max_warmup_bytes,
                )
                continue
            shapes_for_state = tuple(
                rows for rows in shapes if rows <= max_rows_by_memory
            )
            if not shapes_for_state:
                shapes_for_state = (1,)
        else:
            shapes_for_state = shapes
        _, tuned_shapes = autotune_hybrid_nvfp4_lm_head(
            state,
            layer.weight,
            shapes_for_state,
        )
        warmup_hybrid_nvfp4_lm_head_kernels(
            state,
            layer.weight,
            tp_size=getattr(layer, "tp_size", 1),
        )
        logger.debug("Hybrid NVFP4 lm-head tuned row shapes: %s", tuned_shapes)

    _warmup_compact_topk_sampler(worker)
    torch.accelerator.synchronize()
    logger.info(
        "Warmed %d hybrid NVFP4 lm-head state(s) across %d row shapes in %.2fs.",
        len(heads),
        len(shapes),
        perf_counter() - started,
    )


__all__ = ["hybrid_nvfp4_lm_head_warmup"]
