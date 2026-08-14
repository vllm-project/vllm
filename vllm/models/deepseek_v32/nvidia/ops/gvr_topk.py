# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.triton_utils import tl, triton

_TOPK = 2048
_BLOCK_SIZE = 256


@triton.jit
def _prepare_hints_kernel(
    previous_topk,
    state_valid,
    request_indices,
    seq_lens,
    hints,
    previous_stride,
    hint_stride,
    TOPK: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    block = tl.program_id(1)
    offsets = block * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < TOPK
    request_index = tl.load(request_indices + row)
    safe_request_index = tl.maximum(request_index, 0)
    seq_len = tl.maximum(tl.load(seq_lens + row), 1)
    has_state = (
        (request_index >= 0)
        & (seq_len > 1)
        & (tl.load(state_valid + safe_request_index) != 0)
    )
    previous = tl.load(
        previous_topk + safe_request_index * previous_stride + offsets,
        mask=mask & has_state,
        other=0,
    )
    cold_start = offsets * tl.maximum(seq_len - 1, 1) // TOPK
    hint = tl.where(has_state, previous, cold_start)
    tl.store(hints + row * hint_stride + offsets, hint, mask=mask)


@triton.jit
def _store_decode_state_kernel(
    output_indices,
    request_indices,
    previous_topk,
    state_valid,
    output_stride,
    previous_stride,
    TOPK: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    block = tl.program_id(1)
    offsets = block * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < TOPK
    request_index = tl.load(request_indices + row)
    valid_request = request_index >= 0
    safe_request_index = tl.maximum(request_index, 0)
    value = tl.load(
        output_indices + row * output_stride + offsets,
        mask=mask & valid_request,
        other=-1,
    )
    tl.store(
        previous_topk + safe_request_index * previous_stride + offsets,
        value,
        mask=mask & valid_request,
    )
    tl.store(
        state_valid + safe_request_index,
        1,
        mask=valid_request & (block == 0),
    )


@triton.jit
def _store_prefill_state_kernel(
    topk_indices,
    query_start_loc,
    request_indices,
    previous_topk,
    state_valid,
    topk_stride,
    previous_stride,
    request_offset,
    TOPK: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    prefill_request = tl.program_id(0)
    block = tl.program_id(1)
    request = request_offset + prefill_request
    offsets = block * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < TOPK
    request_index = tl.load(request_indices + request)
    valid_request = request_index >= 0
    safe_request_index = tl.maximum(request_index, 0)
    last_token = tl.load(query_start_loc + request + 1) - 1
    value = tl.load(
        topk_indices + last_token * topk_stride + offsets,
        mask=mask & valid_request & (last_token >= 0),
        other=-1,
    )
    tl.store(
        previous_topk + safe_request_index * previous_stride + offsets,
        value,
        mask=mask & valid_request & (last_token >= 0),
    )
    tl.store(
        state_valid + safe_request_index,
        1,
        mask=valid_request & (last_token >= 0) & (block == 0),
    )


def should_use_gvr_topk(num_rows: int, num_columns: int) -> bool:
    """Return whether a measured GVR tier beats vLLM's current selector."""
    if num_columns < 65536 or num_columns > 262144 or num_columns % 64 != 0:
        return False
    return num_rows >= 32


def prepare_gvr_hints(
    previous_topk: torch.Tensor,
    state_valid: torch.Tensor,
    request_indices: torch.Tensor,
    seq_lens: torch.Tensor,
    hints: torch.Tensor,
) -> torch.Tensor:
    num_rows = request_indices.numel()
    hint_view = hints[:num_rows]
    _prepare_hints_kernel[(num_rows, triton.cdiv(_TOPK, _BLOCK_SIZE))](
        previous_topk,
        state_valid,
        request_indices,
        seq_lens,
        hint_view,
        previous_topk.stride(0),
        hint_view.stride(0),
        TOPK=_TOPK,
        BLOCK_SIZE=_BLOCK_SIZE,
    )
    return hint_view


def gvr_topk(
    logits: torch.Tensor,
    seq_lens: torch.Tensor,
    request_indices: torch.Tensor,
    previous_topk: torch.Tensor,
    state_valid: torch.Tensor,
    output_indices: torch.Tensor,
) -> None:
    from .gvr_topk_cutedsl import GvrTopKKernel

    GvrTopKKernel.launch(
        logits,
        output_indices,
        seq_lens,
        output_indices,
        _TOPK,
        previous_topk=previous_topk,
        state_valid=state_valid,
        request_indices=request_indices,
        fuse_hint_prepare=True,
    )


def store_decode_gvr_state(
    output_indices: torch.Tensor,
    request_indices: torch.Tensor,
    previous_topk: torch.Tensor,
    state_valid: torch.Tensor,
) -> None:
    num_rows = request_indices.numel()
    _store_decode_state_kernel[(num_rows, triton.cdiv(_TOPK, _BLOCK_SIZE))](
        output_indices,
        request_indices,
        previous_topk,
        state_valid,
        output_indices.stride(0),
        previous_topk.stride(0),
        TOPK=_TOPK,
        BLOCK_SIZE=_BLOCK_SIZE,
    )


def store_prefill_gvr_state(
    topk_indices: torch.Tensor,
    query_start_loc: torch.Tensor,
    request_indices: torch.Tensor,
    previous_topk: torch.Tensor,
    state_valid: torch.Tensor,
    num_decodes: int,
    num_prefills: int,
) -> None:
    _store_prefill_state_kernel[(num_prefills, triton.cdiv(_TOPK, _BLOCK_SIZE))](
        topk_indices,
        query_start_loc,
        request_indices,
        previous_topk,
        state_valid,
        topk_indices.stride(0),
        previous_topk.stride(0),
        num_decodes,
        TOPK=_TOPK,
        BLOCK_SIZE=_BLOCK_SIZE,
    )
