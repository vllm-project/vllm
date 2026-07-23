# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Warmup DFlash/DSpark speculative decoding Triton kernels.

``_prepare_dflash_inputs_kernel`` (shared by the DFlash and DSpark
speculators) is JIT-compiled by Triton.  Its cache key is driven by three
``tl.constexpr`` values (``BLOCK_SIZE``, ``SAMPLE_FROM_ANCHOR``,
``PAD_SLOT_ID``) plus:

* Grid ``(num_reqs, num_blocks)`` — ``tl.num_programs`` is specialized
  when an axis is 1.
* Pointer divisibility (16B alignment) — runtime tensors come from
  PyTorch's caching allocator (always 16B-aligned), so they all trigger
  the ``'D'`` specialization.
* Integer scalar divisibility — Triton marks a scalar ``'D'`` when its
  value is a multiple of 16, and ``''`` otherwise.  **This is the critical
  part**: warmup must use the *exact* runtime values of ``block_table_stride``,
  ``max_num_reqs``, ``max_num_tokens``, ``max_model_len``, etc. so the
  divisibility tags match.

``BLOCK_SIZE`` is computed at runtime as
``min(256, next_power_of_2(max_tokens_per_req))`` and therefore varies with
batch composition (small for pure decode, 256 for prefill chunks), so every
reachable power of two needs its own cubin.

This module enumerates the relevant ``BLOCK_SIZE`` values crossed with the
single-request / multi-request grid specializations, reading all
deployment-fixed scalar values from the live speculator so the warmup
matches the runtime specialization exactly (including divisibility tags).
It is a no-op when DFlash/DSpark spec decoding is not configured.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from vllm.logger import init_logger

if TYPE_CHECKING:
    from vllm.v1.worker.gpu.spec_decode.dflash.speculator import DFlashSpeculator

logger = init_logger(__name__)

# Powers of two reachable by ``min(256, next_power_of_2(max_tokens_per_req))``.
_DFLASH_BLOCK_SIZES: tuple[int, ...] = (1, 2, 4, 8, 16, 32, 64, 128, 256)

# Grid specializations.  Triton specializes ``tl.num_programs`` when an axis
# is 1, so cover the four "which axes are 1" combinations.
_DFLASH_GRID_COMBOS: tuple[tuple[int, int], ...] = (
    (1, 1),
    (1, 8),
    (50, 1),
    (50, 8),
)


def _alloc(
    size: int,
    dtype: torch.dtype = torch.int32,
    device: str = "cuda:0",
    fill_value: int = 0,
) -> torch.Tensor:
    """Allocate a 16B-aligned tensor for warmup.

    PyTorch's caching allocator returns 512B-aligned blocks, so any
    ``torch.empty(size)`` tensor has ``data_ptr() % 16 == 0``, which triggers
    Triton's ``'D'`` (divisible) pointer specialization — matching runtime
    tensors that also come from the caching allocator.
    """
    t = torch.empty(size, dtype=dtype, device=device)
    t.fill_(fill_value)
    return t


def _warmup_prepare_dflash_inputs_kernel(
    device: str,
    num_speculative_steps: int,
    num_query_per_req: int,
    sample_from_anchor: bool,
    parallel_drafting_token_id: int,
    block_size: int,
    block_table_stride: int,
    max_num_reqs: int,
    max_num_tokens: int,
    max_model_len: int,
) -> None:
    """Exhaustively invoke ``_prepare_dflash_inputs_kernel``.

    All integer scalar arguments are passed with their **exact runtime
    values** read from the live speculator, so Triton's divisibility
    specialization (``%16==0`` → ``'D'``, else ``''``) matches runtime
    exactly.

    Total cache entries generated:
        len(_DFLASH_BLOCK_SIZES) * len(_DFLASH_GRID_COMBOS)
    """
    from vllm.v1.attention.backends.utils import PAD_SLOT_ID
    from vllm.v1.worker.gpu.spec_decode.dflash.speculator import (
        _prepare_dflash_inputs_kernel,
    )

    max_sampled = max_num_reqs * max(num_speculative_steps, 1)

    logger.info(
        "Warming up _prepare_dflash_inputs_kernel: "
        "%d BLOCK_SIZE x %d grid combos = %d entries "
        "(num_speculative_steps=%d, num_query_per_req=%d, "
        "sample_from_anchor=%s, block_size=%d, block_table_stride=%d, "
        "max_num_reqs=%d, max_num_tokens=%d, max_model_len=%d)",
        len(_DFLASH_BLOCK_SIZES),
        len(_DFLASH_GRID_COMBOS),
        len(_DFLASH_BLOCK_SIZES) * len(_DFLASH_GRID_COMBOS),
        num_speculative_steps,
        num_query_per_req,
        sample_from_anchor,
        block_size,
        block_table_stride,
        max_num_reqs,
        max_num_tokens,
        max_model_len,
    )

    for bs in _DFLASH_BLOCK_SIZES:
        for num_reqs, num_blocks in _DFLASH_GRID_COMBOS:
            try:
                n = num_reqs if num_reqs > 0 else 1
                # Per-request inputs.
                target_query_start_loc = _alloc(n + 1, dtype=torch.int32, device=device)
                target_query_start_loc.copy_(
                    torch.arange(n + 1, dtype=torch.int32, device=device)
                )
                target_positions = _alloc(
                    max(n, max_num_reqs), dtype=torch.int64, device=device
                )
                idx_mapping = _alloc(
                    max(n, max_num_reqs), dtype=torch.int32, device=device
                )
                idx_mapping.copy_(
                    torch.arange(max(n, max_num_reqs), dtype=torch.int32, device=device)
                )
                last_sampled = _alloc(max_num_reqs, dtype=torch.int64, device=device)
                next_prefill_tokens = _alloc(
                    max_num_reqs, dtype=torch.int32, device=device
                )
                num_sampled = _alloc(n, dtype=torch.int32, device=device, fill_value=1)
                num_rejected = _alloc(n, dtype=torch.int32, device=device)
                block_table = _alloc(
                    max_num_reqs * block_table_stride,
                    dtype=torch.int32,
                    device=device,
                ).view(max_num_reqs, block_table_stride)

                # Outputs — sized for the padding loops, which run when
                # req_idx == num_reqs - 1 and write up to max_num_reqs /
                # max_num_tokens / max_sampled.
                out_input_ids = _alloc(max_num_tokens, dtype=torch.int32, device=device)
                out_query_positions = _alloc(
                    max_num_tokens, dtype=torch.int64, device=device
                )
                out_query_start_loc = _alloc(
                    max_num_reqs + 1, dtype=torch.int32, device=device
                )
                out_seq_lens = _alloc(max_num_reqs, dtype=torch.int32, device=device)
                out_query_slot_mapping = _alloc(
                    max_num_tokens, dtype=torch.int64, device=device
                )
                out_context_positions = _alloc(
                    max_num_tokens, dtype=torch.int64, device=device
                )
                out_context_slot_mapping = _alloc(
                    max_num_tokens, dtype=torch.int64, device=device
                )
                out_sample_indices = _alloc(
                    max_sampled, dtype=torch.int64, device=device
                )
                out_sample_pos = _alloc(max_sampled, dtype=torch.int64, device=device)
                out_sample_idx_mapping = _alloc(
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
                    max_model_len=max_model_len,
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

    Reads all deployment-fixed scalar values from the live speculator so
    the warmup matches the runtime specialization exactly, including
    Triton's integer divisibility tags (``%16==0`` → ``'D'``).
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

    # Read the actual block_table stride from the live block tables.
    input_block_tables = getattr(speculator.block_tables, "input_block_tables", None)
    if not input_block_tables:
        logger.warning(
            "Skipping DFlash spec-decode warmup: input_block_tables not initialized."
        )
        return
    block_table_stride = int(input_block_tables[0].stride(0))

    device_str = str(device)
    logger.info(
        "Warming up DFlash spec-decode kernels on %s "
        "(num_speculative_steps=%d, sample_from_anchor=%s, "
        "block_table_stride=%d, max_num_reqs=%d, max_num_tokens=%d, "
        "max_model_len=%d)",
        device_str,
        speculator.num_speculative_steps,
        speculator.sample_from_anchor,
        block_table_stride,
        speculator.max_num_reqs,
        speculator.max_num_tokens,
        speculator.max_model_len,
    )

    _warmup_prepare_dflash_inputs_kernel(
        device=device_str,
        num_speculative_steps=speculator.num_speculative_steps,
        num_query_per_req=speculator.num_query_per_req,
        sample_from_anchor=speculator.sample_from_anchor,
        parallel_drafting_token_id=speculator.parallel_drafting_token_id,
        block_size=kernel_block_sizes[0],
        block_table_stride=block_table_stride,
        max_num_reqs=speculator.max_num_reqs,
        max_num_tokens=speculator.max_num_tokens,
        max_model_len=speculator.max_model_len,
    )

    logger.info("DFlash spec-decode kernel warmup finished.")
