# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Startup warmup for the optional hybrid NVFP4 lm-head path."""

from __future__ import annotations

from time import perf_counter

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
logger = init_logger(__name__)
_DEFAULT_MAX_ROWS = 2048


def _collect_lm_heads(
    model: torch.nn.Module,
) -> dict[int, tuple[torch.nn.Module, HybridNvfp4LmHead]]:
    heads: dict[int, tuple[torch.nn.Module, HybridNvfp4LmHead]] = {}
    for module in model.modules():
        state = get_hybrid_nvfp4_lm_head(module)
        if state is not None:
            heads[id(state)] = (module, state)
    return heads


def _row_shapes(worker: object) -> tuple[int, ...]:
    config = getattr(worker, "vllm_config")
    max_rows = int(config.scheduler_config.max_num_seqs)
    speculative_config = config.speculative_config
    if speculative_config is not None:
        max_rows *= speculative_config.num_speculative_tokens + 1
    max_rows = max(1, max_rows)

    capture_sizes = config.compilation_config.cudagraph_capture_sizes or []
    shapes = sorted(
        {
            int(size)
            for size in capture_sizes
            if 0 < int(size) <= max_rows
        }
    )
    if not shapes:
        shapes = list(autotune_row_buckets(max(256, max_rows)))
    if 1 not in shapes:
        shapes.insert(0, 1)
    return tuple(shapes)


def hybrid_nvfp4_lm_head_warmup(worker: object) -> None:
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
        _, tuned_shapes = autotune_hybrid_nvfp4_lm_head(
            state,
            layer.weight,
            shapes,
        )
        warmup_hybrid_nvfp4_lm_head_kernels(
            state,
            layer.weight,
            tp_size=getattr(layer, "tp_size", 1),
        )
        logger.debug("Hybrid NVFP4 lm-head tuned row shapes: %s", tuned_shapes)

    torch.accelerator.synchronize()
    logger.info(
        "Warmed %d hybrid NVFP4 lm-head state(s) across %d row shapes in %.2fs.",
        len(heads),
        len(shapes),
        perf_counter() - started,
    )


__all__ = ["hybrid_nvfp4_lm_head_warmup"]
