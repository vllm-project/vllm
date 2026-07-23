# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Warmup Eagle speculative decoding Triton kernels.

Eagle's ``eagle_prepare_next_token_padded_kernel`` and
``eagle_prepare_inputs_padded_kernel`` are JIT-compiled by Triton.  Triton
specializes on integer arguments whose runtime value equals 1 (turning
them into compile-time constants), so every distinct "which params are 1"
combination produces a separate cubin.  Without warmup the first request
in each shape pays a JIT latency spike.

This module enumerates the relevant parameter combinations and invokes
the kernels once each so Triton compiles and caches every specialization
ahead of live inference.  It is a no-op when Eagle spec decoding is not
configured.
"""

from __future__ import annotations

import itertools
from typing import TYPE_CHECKING

import torch

from vllm.logger import init_logger
from vllm.utils.math_utils import cdiv

if TYPE_CHECKING:
    from vllm.config import VllmConfig

logger = init_logger(__name__)

# Hybrid block factors (alloc_block_size // kernel_block_size) observed across
# vLLM KV-cache configs.  The production ``n_blocks_per_req`` is
# ``max_num_blocks_per_req * blocks_per_kv_block`` (block_table.py:79), and
# ``blocks_per_kv_block`` is 1 for the standard case and >1 when the allocator
# block size differs from the kernel block size.  Enumerate the common factors
# so the kernel is specialized for every reachable row width regardless of
# whether the drafter can see the main model's hybrid-block setting at warmup
# time.
_EAGLE_STEP_HYBRID_FACTORS = (1, 2, 4)


def _warmup_eagle_prepare_next_token_kernel(
    device: str,
    num_speculative_tokens: int,
) -> None:
    """Exhaustively invoke ``eagle_prepare_next_token_padded_kernel``.

    The kernel has four integer parameters that Triton will specialize on
    when their value is 1:

    * ``vocab_size``               – never 1 in practice, skip
    * ``num_sampled_tokens_per_req`` – equals ``num_speculative_tokens + 1``
    * ``num_reqs``                 – 1 for single-request batches
    * ``stride_sampled_token_ids``  – 1 when the tensor is a single column

    ``BLOCK_SIZE_TOKENS`` is a ``tl.constexpr`` set to
    ``next_power_of_2(num_sampled_tokens_per_req)``, so every distinct
    value needs its own cache entry.

    Total cache entries generated:
        2^3 (int combos) × len(BLOCK_SIZES) cache entries.
    """
    from vllm.v1.spec_decode.utils import eagle_prepare_next_token_padded_kernel

    VOCAB_SIZE = 129280  # DeepSeek V4; never specialized (always ≠ 1)

    num_sampled = num_speculative_tokens + 1
    # BLOCK_SIZE_TOKENS must be a power of 2 ≥ num_sampled.
    # Cover all values from 1 up to the block needed by num_sampled so the
    # cache also covers configurations with smaller num_speculative_tokens.
    max_block = 1
    while max_block < num_sampled:
        max_block *= 2
    BLOCK_SIZES = [1]
    b = 2
    while b <= max_block:
        BLOCK_SIZES.append(b)
        b *= 2

    # Each int param: (value when ==1, value when !=1).
    # Triton only differentiates "is 1" vs "is not 1"; the specific non-1
    # value does not affect the cache key.
    int_param_choices = [
        (1, num_sampled),  # num_sampled_tokens_per_req
        (1, 50),  # num_reqs (cover up to 50 reqs)
        (1, num_sampled),  # stride_sampled_token_ids
    ]

    combos = list(itertools.product(*int_param_choices))
    total = len(BLOCK_SIZES) * len(combos)
    logger.debug(
        "Warming up eagle_prepare_next_token_padded_kernel: "
        "%d BLOCK_SIZE × %d int combos = %d entries",
        len(BLOCK_SIZES),
        len(combos),
        total,
    )

    for bs in BLOCK_SIZES:
        for v_sampled, v_reqs, v_stride in combos:
            try:
                sampled = torch.zeros(
                    v_reqs,
                    max(v_stride, v_sampled),
                    dtype=torch.int32,
                    device=device,
                )
                discard_mask = torch.zeros(v_reqs, dtype=torch.bool, device=device)
                backup = torch.zeros(v_reqs, dtype=torch.int32, device=device)
                next_tok = torch.zeros(v_reqs, dtype=torch.int32, device=device)
                valid_count = torch.zeros(v_reqs, dtype=torch.int32, device=device)

                eagle_prepare_next_token_padded_kernel[(v_reqs,)](
                    sampled,
                    discard_mask,
                    backup,
                    next_tok,
                    valid_count,
                    VOCAB_SIZE,
                    v_sampled,
                    v_reqs,
                    v_stride,
                    BLOCK_SIZE_TOKENS=bs,
                )
            except Exception:
                logger.warning(
                    "eagle_prepare_next_token_padded_kernel JIT failed for "
                    "BLOCK_SIZE_TOKENS=%d, num_sampled=%d, num_reqs=%d, "
                    "stride=%d (skipping this combo, continuing)",
                    bs,
                    v_sampled,
                    v_reqs,
                    v_stride,
                    exc_info=True,
                )
    torch.accelerator.synchronize()


def _warmup_eagle_prepare_inputs_kernel(device: str) -> None:
    """Exhaustively invoke ``eagle_prepare_inputs_padded_kernel``.

    Only ``num_reqs`` can be specialized (when ==1).  Two cache entries
    cover all cases: single-request and multi-request.
    """
    from vllm.v1.spec_decode.utils import eagle_prepare_inputs_padded_kernel

    NUM_REQS_VALUES = [1, 50]
    logger.debug(
        "Warming up eagle_prepare_inputs_padded_kernel: %d int combos",
        len(NUM_REQS_VALUES),
    )

    for v_reqs in NUM_REQS_VALUES:
        try:
            cu_num_draft = torch.full((v_reqs,), 4, dtype=torch.int32, device=device)
            valid_count = torch.full((v_reqs,), 2, dtype=torch.int32, device=device)
            query_start_loc = torch.zeros(v_reqs + 1, dtype=torch.int32, device=device)
            for i in range(v_reqs):
                query_start_loc[i + 1] = query_start_loc[i] + 4
            token_indices = torch.zeros(v_reqs, dtype=torch.int32, device=device)
            num_rejected = torch.zeros(v_reqs, dtype=torch.int32, device=device)

            eagle_prepare_inputs_padded_kernel[(v_reqs,)](
                cu_num_draft,
                valid_count,
                query_start_loc,
                token_indices,
                num_rejected,
                v_reqs,
            )
        except Exception:
            logger.warning(
                "eagle_prepare_inputs_padded_kernel JIT failed for "
                "num_reqs=%d (skipping, continuing)",
                v_reqs,
                exc_info=True,
            )
    torch.accelerator.synchronize()


def _warmup_mtp_shared_head_rmsnorm_kernel(device: str) -> None:
    """Invoke ``_mtp_shared_head_rmsnorm_kernel`` once.

    DeepSeek V4 uses a fixed ``HIDDEN=4096``, so there is only one cache
    entry.  Calling it once ensures the kernel is compiled and cached.
    """
    from vllm.models.deepseek_v4.common.ops.fused_mtp_input_rmsnorm import (
        mtp_shared_head_rmsnorm,
    )

    HIDDEN = 4096
    x = torch.zeros(1, HIDDEN, dtype=torch.bfloat16, device=device)
    weight = torch.ones(HIDDEN, dtype=torch.bfloat16, device=device)
    mtp_shared_head_rmsnorm(x, weight, 1e-6)
    torch.accelerator.synchronize()


def _warmup_eagle_step_slot_mapping_metadata_kernel(
    device: str,
    block_size: int,
    max_model_len: int,
    total_cp_size: int,
) -> None:
    """Exhaustively invoke ``eagle_step_slot_mapping_metadata_kernel``.

    Four ``tl.constexpr`` values determine the specialization key:
    ``block_size``, ``max_model_len``, ``n_blocks_per_req``, ``PAD_ID``.
    ``block_size`` and ``max_model_len`` are model-config constants.
    ``PAD_ID`` is always ``-1``.  ``n_blocks_per_req`` comes from the *main*
    model's ``block_table_tensor.shape[1]`` (block_table.py:79-83):

        max_num_blocks_per_req = kv_cache_spec.max_num_blocks_per_req(
            vllm_config, max_model_len)          # includes CP for FullAttention
        n_blocks_per_req = max_num_blocks_per_req * blocks_per_kv_block

    ``blocks_per_kv_block`` is 1 for the standard case and >1 when the
    allocator block size differs from the kernel block size (hybrid blocks).
    Since the warmup cannot read the main model's hybrid-block factor at
    this stage, enumerate the union of CP-aware base counts (1 and the
    configured total CP size) crossed with common hybrid factors.

    Two runtime int args (``batch_size``, ``block_table_stride``) are
    specialized by Triton when their value equals 1, so every "which is 1"
    combination is also enumerated.

    Total cache entries:
        len(n_blocks_candidates) × 2^2 (int combos)
    """
    from vllm.v1.spec_decode.utils import (
        PADDING_SLOT_ID,
        eagle_step_slot_mapping_metadata_kernel,
    )

    # Enumerate n_blocks_per_req over CP × hybrid factors.
    cp_factors = {1, total_cp_size}
    n_blocks_candidates: set[int] = set()
    for cp in cp_factors:
        base = cdiv(max_model_len, block_size * cp)
        for hybrid in _EAGLE_STEP_HYBRID_FACTORS:
            n_blocks_candidates.add(max(base * hybrid, 1))

    # Runtime int args specialized when == 1.
    int_param_choices = [
        (1, 50),  # batch_size
        # block_table_stride equals n_blocks_per_req at runtime; only
        # "is 1" vs "is not 1" matters, so a single non-1 value suffices.
        (1, max(max(n_blocks_candidates), 2)),  # block_table_stride
    ]
    combos = list(itertools.product(*int_param_choices))

    total = len(n_blocks_candidates) * len(combos)
    logger.debug(
        "Warming up eagle_step_slot_mapping_metadata_kernel: "
        "%d n_blocks_per_req × %d int combos = %d entries "
        "(cp_factors=%s, hybrid_factors=%s)",
        len(n_blocks_candidates),
        len(combos),
        total,
        sorted(cp_factors),
        list(_EAGLE_STEP_HYBRID_FACTORS),
    )

    input_batch_size = 50
    for n_blocks_per_req in sorted(n_blocks_candidates):
        for v_batch, v_stride in combos:
            try:
                positions = torch.zeros(
                    input_batch_size, dtype=torch.int64, device=device
                )
                block_table = torch.zeros(
                    input_batch_size,
                    max(v_stride, 1),
                    dtype=torch.int32,
                    device=device,
                )
                seq_lens = torch.zeros(
                    input_batch_size, dtype=torch.int32, device=device
                )
                out_clamped = torch.zeros(
                    input_batch_size, dtype=torch.int64, device=device
                )
                out_slot = torch.zeros(
                    input_batch_size, dtype=torch.int64, device=device
                )

                eagle_step_slot_mapping_metadata_kernel[(input_batch_size,)](
                    positions,
                    block_table,
                    v_stride,
                    seq_lens,
                    out_clamped,
                    out_slot,
                    block_size=block_size,
                    max_model_len=max_model_len,
                    n_blocks_per_req=n_blocks_per_req,
                    PAD_ID=PADDING_SLOT_ID,
                    batch_size=v_batch,
                )
            except Exception:
                logger.warning(
                    "eagle_step_slot_mapping_metadata_kernel JIT failed for "
                    "n_blocks_per_req=%d, batch_size=%d, stride=%d "
                    "(skipping this combo, continuing)",
                    n_blocks_per_req,
                    v_batch,
                    v_stride,
                    exc_info=True,
                )
    torch.accelerator.synchronize()


def eagle_eagle_kernel_warmup(
    device: torch.device,
    num_speculative_tokens: int | None,
    vllm_config: VllmConfig,
    block_size: int | None = None,
    max_model_len: int | None = None,
) -> None:
    """Warm up Eagle spec-decode Triton kernels.

    Args:
        device: CUDA device to compile kernels for.
        num_speculative_tokens: Number of speculative tokens configured.
            If ``None`` or 0, Eagle kernels are not used and this is a
            no-op.
        vllm_config: VllmConfig, used to read ``parallel_config`` for CP
            factors that affect ``n_blocks_per_req`` in
            ``eagle_step_slot_mapping_metadata_kernel``.
        block_size: KV cache block size. Required to warmup
            ``eagle_step_slot_mapping_metadata_kernel``.
        max_model_len: Max model length. Required to warmup
            ``eagle_step_slot_mapping_metadata_kernel``.
    """
    if num_speculative_tokens is None or num_speculative_tokens <= 0:
        return

    if device.type != "cuda":
        return

    device_str = str(device)
    logger.info(
        "Warming up Eagle spec-decode kernels on %s (num_speculative_tokens=%d)",
        device_str,
        num_speculative_tokens,
    )

    _warmup_eagle_prepare_next_token_kernel(device_str, num_speculative_tokens)
    _warmup_eagle_prepare_inputs_kernel(device_str)
    _warmup_mtp_shared_head_rmsnorm_kernel(device_str)
    if block_size is not None and max_model_len is not None:
        parallel_config = vllm_config.parallel_config
        total_cp_size = (
            parallel_config.decode_context_parallel_size
            * parallel_config.prefill_context_parallel_size
        )
        _warmup_eagle_step_slot_mapping_metadata_kernel(
            device_str,
            block_size,
            max_model_len,
            total_cp_size,
        )

    logger.info("Eagle spec-decode kernel warmup finished.")
