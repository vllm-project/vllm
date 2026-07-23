# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Equivalence test for ``precopy_mamba_align_fused_kernel``.

The V2 "align" pre-copy must migrate mamba state across block boundaries with
byte-identical semantics to the V1 copy specs (``get_conv_copy_spec`` /
``get_temporal_copy_spec``):

* conv state (SD layout, conv_width > 0): shift the sliding window by
  ``token_bias`` tokens -- ``state[bt[src_col], token_bias:]`` ->
  ``state[bt[dst_col], :conv_width - token_bias]``.
* temporal state (conv_width == 0): ``token_bias`` selects the accepted
  speculative column -- ``state[bt[src_col + token_bias]]`` ->
  ``state[bt[dst_col]]``.

The kernel must also no-op when ``src_col < 0`` (fresh request) or
``src_col == dst_col`` (no boundary crossed).
"""

from __future__ import annotations

import torch

from vllm.platforms import current_platform
from vllm.v1.worker.mamba_utils import precopy_mamba_align_fused_kernel

try:
    import pytest

    pytestmark = pytest.mark.skipif(
        not current_platform.is_cuda(),
        reason="precopy_mamba_align_fused_kernel needs CUDA/Triton",
    )
    _parametrize = pytest.mark.parametrize
except ModuleNotFoundError:  # allow running directly as ``python <thisfile>``
    pytest = None

    def _parametrize(_name, _values):
        def _deco(fn):
            return fn

        return _deco


NUM_LAYERS = 3
CONV_WIDTH = 4  # conv_kernel - 1 + num_spec
CONV_DIM = 96
SSM_SHAPE = (4, 16, 16)
MAX_COLS = 8


def _build_state(num_blocks, device):
    """Per-layer (conv SD [nb, width, dim] bf16, ssm [nb, *shape] fp32) pools."""
    convs, ssms = [], []
    for _ in range(NUM_LAYERS):
        convs.append(
            torch.randn(
                num_blocks, CONV_WIDTH, CONV_DIM, dtype=torch.bfloat16, device=device
            )
        )
        ssms.append(
            torch.randn(num_blocks, *SSM_SHAPE, dtype=torch.float32, device=device)
        )
    return convs, ssms


def _build_meta(convs, ssms, device):
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


def _reference(convs, ssms, bt, src_col, dst_col, bias, num_reqs):
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
            conv_ref[layer][dblk, : CONV_WIDTH - tb] = conv_pre[layer][sblk, tb:]
            ssm_ref[layer][dblk] = ssm_pre[layer][tblk]
    return conv_ref, ssm_ref


@_parametrize("num_reqs", [1, 4, 16])
@_parametrize("token_bias", [0, 1, 2])
def test_precopy_matches_v1_copy_specs(num_reqs, token_bias):
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

    convs, ssms = _build_state(num_blocks, device)
    conv_ref, ssm_ref = _reference(
        convs, ssms, bt.cpu(), src_col.cpu(), dst_col.cpu(), bias.cpu(), num_reqs
    )

    base, blk_stride, elem, inner, width, group, drc, drs = _build_meta(
        convs, ssms, device
    )
    bt_ptrs = torch.tensor([bt.data_ptr()], dtype=torch.int64, device=device)
    idx_mapping = torch.arange(num_reqs, dtype=torch.int32, device=device)
    grid = (num_reqs, NUM_LAYERS * 2)
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
        idx_mapping,
        num_reqs,
        COPY_BLOCK_SIZE=1024,
        CONV_STATE_DIM_FIRST=False,
    )
    torch.accelerator.synchronize()

    for layer in range(NUM_LAYERS):
        torch.testing.assert_close(convs[layer], conv_ref[layer], rtol=0, atol=0)
        torch.testing.assert_close(ssms[layer], ssm_ref[layer], rtol=0, atol=0)


if __name__ == "__main__":
    for nr in (1, 4, 16):
        for tb in (0, 1, 2):
            test_precopy_matches_v1_copy_specs(nr, tb)
            print(f"OK num_reqs={nr} token_bias={tb}")
