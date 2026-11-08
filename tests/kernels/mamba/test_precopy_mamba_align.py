# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Equivalence test for `precopy_mamba_align_fused_kernel`.

The fused "align" pre-copy must migrate mamba state across block boundaries
with byte-identical semantics to the scalar V1 copy specs
(`get_conv_copy_spec` / `get_temporal_copy_spec`):

* conv state (SD layout, conv_width > 0): shift the sliding window by
  ``token_bias`` tokens -- ``state[bt[src_col], token_bias:]`` ->
  ``state[bt[dst_col], :conv_width - token_bias]``.
* temporal state (conv_width == 0): ``token_bias`` selects the accepted
  speculative column -- ``state[bt[src_col + token_bias]]`` ->
  ``state[bt[dst_col]]``.

The kernel must also no-op when ``src_col < 0`` (fresh request) or
``src_col == dst_col`` (no boundary crossed). Callers pass an explicit
``idx_mapping``; ``idx_mapping=None`` launches in batch order.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch

from vllm.model_executor.layers.mamba import mamba_utils as layer_mamba_utils
from vllm.platforms import current_platform
from vllm.v1.worker.mamba_utils import (
    _TEMPORAL_TILES,
    postprocess_mamba_fused_kernel,
    precopy_mamba_align_fused_kernel,
)

_parametrize: Callable[..., Callable[[Any], Any]]

try:
    import pytest

    _cuda_required = pytest.mark.skipif(
        not current_platform.is_cuda_alike(),
        reason="precopy_mamba_align_fused_kernel needs CUDA/Triton",
    )
    _parametrize = pytest.mark.parametrize
except ModuleNotFoundError:  # allow running directly as ``python <thisfile>``

    def _cuda_required(fn):
        return fn

    def _no_parametrize(_name, _values):
        def _deco(fn):
            return fn

        return _deco

    _parametrize = _no_parametrize


NUM_LAYERS = 3
CONV_WIDTH = 4  # conv_kernel - 1 + num_spec
CONV_DIM = 96
SSM_SHAPE = (4, 16, 16)
MAX_COLS = 8


def _build_state(num_blocks, device, conv_state_dim_first):
    """Per-layer (conv SD [nb, width, dim] bf16, ssm [nb, *shape] fp32) pools."""
    convs, ssms = [], []
    for _ in range(NUM_LAYERS):
        conv_shape = (
            (num_blocks, CONV_DIM, CONV_WIDTH)
            if conv_state_dim_first
            else (num_blocks, CONV_WIDTH, CONV_DIM)
        )
        convs.append(
            torch.randn(
                *conv_shape,
                dtype=torch.bfloat16,
                device=device,
            )
        )
        ssms.append(
            torch.randn(num_blocks, *SSM_SHAPE, dtype=torch.float32, device=device)
        )
    return convs, ssms


def _build_meta(convs, ssms, device, conv_state_dim_first):
    """Flattened per-(layer, state-type) metadata, ordered conv, ssm per layer."""
    n = NUM_LAYERS * 2
    base = torch.zeros(n, dtype=torch.int64, device=device)
    blk_stride = torch.zeros(n, dtype=torch.int64, device=device)
    elem = torch.zeros(n, dtype=torch.int32, device=device)
    inner = torch.zeros(n, dtype=torch.int64, device=device)
    width = torch.zeros(n, dtype=torch.int32, device=device)
    group = torch.zeros(n, dtype=torch.int32, device=device)
    drc = torch.zeros(n, dtype=torch.int32, device=device)  # DS rows (unused, SD)
    drs = torch.zeros(n, dtype=torch.int64, device=device)
    i = 0
    for layer in range(NUM_LAYERS):
        conv, ssm = convs[layer], ssms[layer]
        # conv (SD): width = size(1), inner = stride(1)
        base[i] = conv.data_ptr()
        blk_stride[i] = conv.stride(0) * conv.element_size()
        elem[i] = conv.element_size()
        if conv_state_dim_first:
            width[i] = conv.size(2)
            inner[i] = 1
            drc[i] = conv.size(1)
            drs[i] = conv.stride(1) * conv.element_size()
        else:
            width[i] = conv.size(1)
            inner[i] = conv.stride(1)
        i += 1
        # ssm (temporal): width = 0, inner = elems per block
        base[i] = ssm.data_ptr()
        blk_stride[i] = ssm.stride(0) * ssm.element_size()
        elem[i] = ssm.element_size()
        width[i] = 0
        inner[i] = ssm[0].numel()
        i += 1
    return base, blk_stride, elem, inner, width, group, drc, drs


def _reference(convs, ssms, bt, src_col, dst_col, bias, num_reqs, conv_dim_first):
    """Apply the V1 copy semantics on clones, reading from the pre-copy state."""
    conv_pre = [c.clone() for c in convs]
    ssm_pre = [s.clone() for s in ssms]
    conv_ref = [c.clone() for c in convs]
    ssm_ref = [s.clone() for s in ssms]
    for r in range(num_reqs):
        sc, dc, tb = int(src_col[r]), int(dst_col[r]), int(bias[r])
        if sc < 0 or sc == dc:
            continue
        sblk, dblk = int(bt[r, sc]), int(bt[r, dc])
        tblk = int(bt[r, sc + tb])  # temporal src column shifted by bias
        for layer in range(NUM_LAYERS):
            if conv_dim_first:
                conv_ref[layer][dblk, :, : CONV_WIDTH - tb] = conv_pre[layer][
                    sblk, :, tb:
                ]
            else:
                conv_ref[layer][dblk, : CONV_WIDTH - tb] = conv_pre[layer][sblk, tb:]
            ssm_ref[layer][dblk] = ssm_pre[layer][tblk]
    return conv_ref, ssm_ref


@_parametrize("conv_state_dim_first", [False, True])
@_parametrize("num_reqs", [1, 4, 16])
@_parametrize("token_bias", [0, 1, 2])
@_parametrize("has_idx_mapping", [True, False])
@_parametrize("temporal_tiles", [1, _TEMPORAL_TILES])
@_cuda_required
def test_precopy_matches_v1_copy_specs(
    num_reqs,
    token_bias,
    has_idx_mapping,
    conv_state_dim_first,
    temporal_tiles,
):
    device = torch.device("cuda")
    torch.manual_seed(0)
    # Distinct physical block per (req, col) so copies never alias.
    num_blocks = num_reqs * MAX_COLS + 1
    bt = torch.empty(num_reqs, MAX_COLS, dtype=torch.int32, device=device)
    for r in range(num_reqs):
        bt[r] = torch.arange(
            1 + r * MAX_COLS, 1 + (r + 1) * MAX_COLS, dtype=torch.int32, device=device
        )

    # Per-req columns: req 0 fresh (src=-1, skip), req 1 same block (skip),
    # the rest cross from col 1 -> col 0 with the given spec token bias.
    src_col = torch.full((num_reqs,), 1, dtype=torch.int32, device=device)
    dst_col = torch.zeros(num_reqs, dtype=torch.int32, device=device)
    bias = torch.full((num_reqs,), token_bias, dtype=torch.int32, device=device)
    if num_reqs >= 1:
        src_col[0] = -1  # fresh -> no copy
    if num_reqs >= 2:
        dst_col[1] = 1  # src_col == dst_col -> no copy

    convs, ssms = _build_state(num_blocks, device, conv_state_dim_first)
    conv_ref, ssm_ref = _reference(
        convs,
        ssms,
        bt.cpu(),
        src_col.cpu(),
        dst_col.cpu(),
        bias.cpu(),
        num_reqs,
        conv_state_dim_first,
    )

    base, blk_stride, elem, inner, width, group, drc, drs = _build_meta(
        convs, ssms, device, conv_state_dim_first
    )
    bt_ptrs = torch.tensor([bt.data_ptr()], dtype=torch.int64, device=device)
    # Non-identity mapping: block-table rows are request slots, not batch rows.
    idx_mapping = torch.arange(num_reqs, dtype=torch.int32, device=device).flip(0)
    grid = (num_reqs, NUM_LAYERS * 2, temporal_tiles)
    precopy_mamba_align_fused_kernel[grid](
        dst_col,
        src_col,
        bias,
        bt_ptrs,
        bt.stride(0),
        base,
        blk_stride,
        elem,
        inner,
        width,
        group,
        drc,
        drs,
        idx_mapping if has_idx_mapping else None,
        num_reqs,
        COPY_BLOCK_SIZE=1024,
        CONV_STATE_DIM_FIRST=conv_state_dim_first,
        HAS_IDX_MAPPING=has_idx_mapping,
        TEMPORAL_TILES=temporal_tiles,
    )
    torch.accelerator.synchronize()

    for layer in range(NUM_LAYERS):
        torch.testing.assert_close(convs[layer], conv_ref[layer], rtol=0, atol=0)
        torch.testing.assert_close(ssms[layer], ssm_ref[layer], rtol=0, atol=0)


@_parametrize("conv_state_dim_first", [False, True])
@_cuda_required
def test_v2_postprocess_indexes_block_tables_by_req_slot(conv_state_dim_first):
    """Under PP the V2 postprocess runs after later steps have re-gathered the
    batch-ordered tables, so it must index the request-slot tables by slot."""
    device = torch.device("cuda")
    torch.manual_seed(0)
    num_reqs = 4
    block_size = 8
    num_blocks = num_reqs * MAX_COLS + 1
    bt = torch.arange(1, num_blocks, dtype=torch.int32, device=device).view(
        num_reqs, MAX_COLS
    )
    # slot 0: copy col 1 -> 0; slot 1: src == dst; slot 2: not aligned;
    # slot 3: copy col 2 -> 0 with one accepted draft token.
    num_accepted = torch.tensor([1, 1, 1, 2], dtype=torch.int32, device=device)
    state_idx = torch.tensor([1, 0, 1, 2], dtype=torch.int32, device=device)
    new_num_computed = torch.tensor([8, 8, 5, 8], dtype=torch.int32, device=device)
    ref_src = torch.tensor([1, 0, -1, 2], dtype=torch.int32)
    ref_dst = torch.zeros(num_reqs, dtype=torch.int32)
    ref_bias = torch.tensor([0, 0, 0, 1], dtype=torch.int32)

    convs, ssms = _build_state(num_blocks, device, conv_state_dim_first)
    conv_ref, ssm_ref = _reference(
        convs,
        ssms,
        bt.cpu(),
        ref_src,
        ref_dst,
        ref_bias,
        num_reqs,
        conv_state_dim_first,
    )
    base, blk_stride, elem, inner, width, group, drc, drs = _build_meta(
        convs, ssms, device, conv_state_dim_first
    )
    bt_ptrs = torch.tensor([bt.data_ptr()], dtype=torch.int64, device=device)
    idx_mapping = torch.tensor([2, 0, 3, 1], dtype=torch.int32, device=device)
    postprocess_mamba_fused_kernel[(num_reqs, NUM_LAYERS * 2, 1)](
        num_accepted.clone(),
        state_idx,
        None,
        new_num_computed,
        None,
        bt_ptrs,
        bt.stride(0),
        base,
        blk_stride,
        elem,
        inner,
        width,
        group,
        drc,
        drs,
        num_accepted,
        idx_mapping,
        num_reqs,
        block_size=block_size,
        COPY_BLOCK_SIZE=1024,
        CONV_STATE_DIM_FIRST=conv_state_dim_first,
        HAS_IDX_MAPPING=True,
        PRECOMPUTED_NEW_COMPUTED=True,
        TEMPORAL_TILES=1,
    )
    torch.accelerator.synchronize()

    for layer in range(NUM_LAYERS):
        torch.testing.assert_close(convs[layer], conv_ref[layer], rtol=0, atol=0)
        torch.testing.assert_close(ssms[layer], ssm_ref[layer], rtol=0, atol=0)


def test_ds_conv_copy_spec_reproduces_multi_accept_assert(monkeypatch):
    monkeypatch.setattr(
        layer_mamba_utils,
        "is_conv_state_dim_first",
        lambda: True,
    )
    state = torch.empty((2, CONV_DIM, CONV_WIDTH), dtype=torch.bfloat16)

    with pytest.raises(AssertionError, match="num_accepted_tokens > 1"):
        layer_mamba_utils.get_conv_copy_spec(
            state=state,
            block_ids=[0, 1],
            cur_block_idx=0,
            num_accepted_tokens=3,
        )


if __name__ == "__main__":
    from itertools import product

    _CASES = product(
        (1, 4, 16),  # num_reqs
        (0, 1, 2),  # token_bias
        (True, False),  # has_idx_mapping
        (False, True),  # conv_state_dim_first
        (1, _TEMPORAL_TILES),  # temporal_tiles
    )
    for nr, tb, mapping, dim_first, tt in _CASES:
        test_precopy_matches_v1_copy_specs(nr, tb, mapping, dim_first, tt)
        print(
            f"OK num_reqs={nr} token_bias={tb} has_idx_mapping={mapping} "
            f"conv_dim_first={dim_first} temporal_tiles={tt}"
        )
