# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.triton_utils import tl, triton

_DUMMY_APE_CACHE: dict[torch.device, torch.Tensor] = {}


def save_partial_states(
    kv: torch.Tensor,
    score: torch.Tensor,
    ape: torch.Tensor | None,
    positions: torch.Tensor,
    state_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    block_size: int,
    state_width: int,
    compress_ratio: int,
    pdl_kwargs: dict | None = None,
    dummy_ape: torch.Tensor | None = None,
) -> None:
    """Write packed [kv, score(+ape)] partial states into the compressor cache.

    One program per token; pads (slot_id == -1) are skipped.

    Args:
        ape: If None, stores raw score without APE addition (bf16 state_cache mode).
             APE will be added inside the compress kernel instead.
        dummy_ape: Reusable placeholder used only to satisfy the Triton kernel
             signature when ape is None. It is not read when SKIP_APE=True.
    """
    num_actual = slot_mapping.shape[0]
    head_size = kv.shape[-1]
    skip_ape = ape is None
    if skip_ape:
        if dummy_ape is None:
            dummy_ape = _DUMMY_APE_CACHE.get(kv.device)
            if dummy_ape is None:
                dummy_ape = torch.empty(1, 1, dtype=torch.float32, device=kv.device)
                _DUMMY_APE_CACHE[kv.device] = dummy_ape
        ape = dummy_ape

    _save_partial_states_kernel[(num_actual,)](
        kv,
        kv.stride(0),
        score,
        score.stride(0),
        ape,
        ape.stride(0),
        positions,
        state_cache,
        state_cache.stride(0),
        state_cache.stride(1),
        slot_mapping,
        block_size,
        HEAD_SIZE=head_size,
        TRITON_BLOCK_SIZE=triton.next_power_of_2(head_size),
        STATE_WIDTH=state_width,
        COMPRESS_RATIO=compress_ratio,
        SKIP_APE=skip_ape,
        **(pdl_kwargs or {}),
    )


@triton.jit
def _save_partial_states_kernel(
    kv_ptr,
    kv_stride,
    score_ptr,
    score_stride,
    ape_ptr,
    ape_stride,
    positions_ptr,
    state_cache_ptr,
    state_cache_stride0,
    state_cache_stride1,
    slot_mapping_ptr,
    block_size,
    HEAD_SIZE: tl.constexpr,
    TRITON_BLOCK_SIZE: tl.constexpr,
    # state_cache last dim packs [kv_state, score_state], each STATE_WIDTH wide.
    STATE_WIDTH: tl.constexpr,
    COMPRESS_RATIO: tl.constexpr,
    SKIP_APE: tl.constexpr,
):
    token_idx = tl.program_id(0)
    slot_id = tl.load(slot_mapping_ptr + token_idx)

    # Skip padded / invalid tokens (slot_id == -1 is the PAD sentinel used
    # by vLLM).  During CUDA graph replay the batch may contain padding
    # tokens whose slot_mapping is -1; writing to kv_state[-1] would be an
    # illegal memory access.
    if slot_id < 0:
        return

    block_idx = slot_id // block_size
    pos_in_block = slot_id % block_size
    base_ptr = (
        state_cache_ptr
        + block_idx * state_cache_stride0
        + pos_in_block * state_cache_stride1
    )

    block = tl.arange(0, TRITON_BLOCK_SIZE)
    mask = block < HEAD_SIZE

    kv = tl.load(kv_ptr + token_idx * kv_stride + block, mask=mask)
    tl.store(base_ptr + block, kv, mask=mask)

    score = tl.load(score_ptr + token_idx * score_stride + block, mask=mask)
    if SKIP_APE:
        # BF16 state_cache mode: store raw score without APE.
        # APE will be added inside the compress kernel.
        tl.store(base_ptr + STATE_WIDTH + block, score, mask=mask)
    else:
        # FP32 state_cache mode: fuse APE addition here.
        # score += ape[position % compress_ratio]
        position = tl.load(positions_ptr + token_idx)
        ape_row = position % COMPRESS_RATIO
        ape = tl.load(ape_ptr + ape_row * ape_stride + block, mask=mask)
        tl.store(base_ptr + STATE_WIDTH + block, score + ape, mask=mask)
