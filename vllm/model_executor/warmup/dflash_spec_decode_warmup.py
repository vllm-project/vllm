# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Warmup DFlash/DSpark speculative decoding Triton kernels.

``_prepare_dflash_inputs_kernel`` (shared by the DFlash and DSpark
speculators) is JIT-compiled by Triton.  Its cache key is driven by three
``tl.constexpr`` values (``BLOCK_SIZE``, ``SAMPLE_FROM_ANCHOR``,
``PAD_SLOT_ID``) plus the grid dimensions ``(num_reqs, num_blocks)``, which
Triton specializes when a program-count axis is 1.

``BLOCK_SIZE`` is computed at runtime as
``min(256, next_power_of_2(max_tokens_per_req))`` and therefore varies with
batch composition (small for pure decode, 256 for prefill chunks), so every
reachable power of two needs its own cubin.  Without warmup the first
request in each shape pays a JIT latency spike.

This module enumerates the relevant ``BLOCK_SIZE`` values crossed with the
single-request / multi-request grid specializations and invokes the kernel
once each so Triton compiles and caches every specialization ahead of live
inference.  It is a no-op when DFlash/DSpark spec decoding is not configured.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from vllm.logger import init_logger

if TYPE_CHECKING:
    from vllm.v1.worker.gpu.spec_decode.dflash.speculator import DFlashSpeculator

logger = init_logger(__name__)

# Powers of two reachable by ``min(256, next_power_of_2(max_tokens_per_req))``.
# ``max_tokens_per_req = max_target_query_len + num_query_per_req`` ranges from
# a handful of tokens (decode) up to the max prefill chunk, so every value
# from 1 to 256 can occur.
_DFLASH_BLOCK_SIZES: tuple[int, ...] = (1, 2, 4, 8, 16, 32, 64, 128, 256)

# Grid specializations.  Triton specializes ``tl.num_programs`` when an axis
# is 1, so cover the four "which axes are 1" combinations.  Single-request
# decode is ``(1, 1)``; prefill of one long request is ``(1, many)``; a batch
# of small decodes is ``(many, 1)``; a mixed batch is ``(many, many)``.
_DFLASH_GRID_COMBOS: tuple[tuple[int, int], ...] = (
    (1, 1),
    (1, 8),
    (50, 1),
    (50, 8),
)

# Max sizes for the dummy buffers.  These only need to be large enough to keep
# the kernel in bounds; their exact non-1 values do not affect the cache key
# (Triton only specializes integer scalars that are exactly 1).
_DFLASH_WARMUP_MAX_REQS = 50
_DFLASH_WARMUP_MAX_TOKENS = 512


def _warmup_prepare_dflash_inputs_kernel(
    device: str,
    num_speculative_steps: int,
    num_query_per_req: int,
    sample_from_anchor: bool,
    parallel_drafting_token_id: int,
    block_size: int,
) -> None:
    """Exhaustively invoke ``_prepare_dflash_inputs_kernel``.

    The integer scalar arguments (``block_size``, ``block_table_stride``,
    ``num_query_per_req``, ``num_speculative_steps``,
    ``parallel_drafting_token_id``, ``max_num_reqs``, ``max_num_tokens``,
    ``max_model_len``) are constant per deployment, so they are passed with
    their configured values and need no enumeration (Triton only specializes
    integer scalars whose value is exactly 1, which is covered implicitly when
    the configured value happens to be 1).

    Total cache entries generated:
        len(_DFLASH_BLOCK_SIZES) * len(_DFLASH_GRID_COMBOS)
    """
    from vllm.v1.attention.backends.utils import PAD_SLOT_ID
    from vllm.v1.worker.gpu.spec_decode.dflash.speculator import (
        _prepare_dflash_inputs_kernel,
    )

    max_num_reqs = _DFLASH_WARMUP_MAX_REQS
    max_num_tokens = _DFLASH_WARMUP_MAX_TOKENS
    max_sampled = max_num_reqs * max(num_speculative_steps, 1)
    # block_table_stride is constant per deployment; pick any non-1 value so it
    # matches the non-1 specialization used at runtime.  A small stride keeps
    # the dummy block table compact while staying >= any block index touched.
    block_table_stride = max(block_size, 2)
    num_reqs_cap = _DFLASH_WARMUP_MAX_REQS

    logger.info(
        "Warming up _prepare_dflash_inputs_kernel: "
        "%d BLOCK_SIZE x %d grid combos = %d entries "
        "(num_speculative_steps=%d, num_query_per_req=%d, "
        "sample_from_anchor=%s, block_size=%d)",
        len(_DFLASH_BLOCK_SIZES),
        len(_DFLASH_GRID_COMBOS),
        len(_DFLASH_BLOCK_SIZES) * len(_DFLASH_GRID_COMBOS),
        num_speculative_steps,
        num_query_per_req,
        sample_from_anchor,
        block_size,
    )

    for bs in _DFLASH_BLOCK_SIZES:
        for num_reqs, num_blocks in _DFLASH_GRID_COMBOS:
            try:
                # Per-request inputs (indexed by req_idx in [0, num_reqs)).
                # Give each request exactly one context token so the kernel's
                # ``last_valid_pos = target_positions[valid_ctx_end - 1]``
                # load stays in bounds (avoids the negative index that an empty
                # context would produce).
                n = num_reqs if num_reqs > 0 else 1
                target_query_start_loc = torch.arange(
                    n + 1, dtype=torch.int32, device=device
                )
                target_positions = torch.zeros(
                    max(n, num_reqs_cap), dtype=torch.int64, device=device
                )
                idx_mapping = torch.arange(
                    max(n, num_reqs_cap), dtype=torch.int32, device=device
                )
                last_sampled = torch.zeros(
                    num_reqs_cap, dtype=torch.int64, device=device
                )
                next_prefill_tokens = torch.zeros(
                    num_reqs_cap, dtype=torch.int32, device=device
                )
                num_sampled = torch.ones(n, dtype=torch.int32, device=device)
                num_rejected = torch.zeros(n, dtype=torch.int32, device=device)
                block_table = torch.zeros(
                    num_reqs_cap,
                    block_table_stride,
                    dtype=torch.int32,
                    device=device,
                )

                # Outputs (sized for the padding loops, which run when
                # req_idx == num_reqs - 1 and write up to max_num_reqs /
                # max_num_tokens / max_sampled).
                out_input_ids = torch.zeros(
                    max_num_tokens, dtype=torch.int32, device=device
                )
                out_query_positions = torch.zeros(
                    max_num_tokens, dtype=torch.int64, device=device
                )
                out_query_start_loc = torch.zeros(
                    max_num_reqs + 1, dtype=torch.int32, device=device
                )
                out_seq_lens = torch.zeros(
                    max_num_reqs, dtype=torch.int32, device=device
                )
                out_query_slot_mapping = torch.zeros(
                    max_num_tokens, dtype=torch.int64, device=device
                )
                out_context_positions = torch.zeros(
                    max_num_tokens, dtype=torch.int64, device=device
                )
                out_context_slot_mapping = torch.zeros(
                    max_num_tokens, dtype=torch.int64, device=device
                )
                out_sample_indices = torch.zeros(
                    max_sampled, dtype=torch.int64, device=device
                )
                out_sample_pos = torch.zeros(
                    max_sampled, dtype=torch.int64, device=device
                )
                out_sample_idx_mapping = torch.zeros(
                    max_sampled, dtype=torch.int32, device=device
                )

                _prepare_dflash_inputs_kernel[(num_reqs, num_blocks)](
                    out_input_ids,
                    out_query_positions,
                    out_query_start_loc,
                    out_seq_lens,
                    out_query_slot_mapping,
                    out_context_positions,
                    out_context_slot_mapping,
                    out_sample_indices,
                    out_sample_pos,
                    out_sample_idx_mapping,
                    target_positions,
                    target_query_start_loc,
                    idx_mapping,
                    last_sampled,
                    next_prefill_tokens,
                    num_sampled,
                    num_rejected,
                    block_table,
                    block_table_stride,
                    parallel_drafting_token_id,
                    block_size,
                    num_query_per_req,
                    num_speculative_steps,
                    max_num_reqs,
                    max_num_tokens,
                    max_model_len=max_num_tokens,
                    SAMPLE_FROM_ANCHOR=sample_from_anchor,
                    PAD_SLOT_ID=PAD_SLOT_ID,
                    BLOCK_SIZE=bs,
                )
            except Exception:
                logger.warning(
                    "_prepare_dflash_inputs_kernel JIT failed for "
                    "BLOCK_SIZE=%d, num_reqs=%d, num_blocks=%d "
                    "(skipping this combo, continuing)",
                    bs,
                    num_reqs,
                    num_blocks,
                    exc_info=True,
                )
    torch.accelerator.synchronize()


def dflash_kernel_warmup(speculator: DFlashSpeculator) -> None:
    """Warm up DFlash/DSpark spec-decode Triton kernels.

    Args:
        speculator: The configured ``DFlashSpeculator`` (or ``DSpark``
            subclass).  All kernel parameters are read from it so the warmup
            matches the runtime specialization exactly.
    """
    device = speculator.device
    if device.type != "cuda":
        return

    kernel_block_sizes = getattr(speculator.block_tables, "kernel_block_sizes", None)
    if not kernel_block_sizes:
        logger.warning(
            "Skipping DFlash spec-decode warmup: block_tables not initialized."
        )
        return

    device_str = str(device)
    logger.info(
        "Warming up DFlash spec-decode kernels on %s "
        "(num_speculative_steps=%d, sample_from_anchor=%s)",
        device_str,
        speculator.num_speculative_steps,
        speculator.sample_from_anchor,
    )

    _warmup_prepare_dflash_inputs_kernel(
        device=device_str,
        num_speculative_steps=speculator.num_speculative_steps,
        num_query_per_req=speculator.num_query_per_req,
        sample_from_anchor=speculator.sample_from_anchor,
        parallel_drafting_token_id=speculator.parallel_drafting_token_id,
        block_size=kernel_block_sizes[0],
    )

    logger.info("DFlash spec-decode kernel warmup finished.")
