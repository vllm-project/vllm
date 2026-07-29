# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Equivalence test for ``precopy_mamba_align_fused_kernel``.

The fused "align" pre-copy must migrate mamba state across block boundaries
with byte-identical semantics to the scalar V1 copy specs
(``get_conv_copy_spec`` / ``get_temporal_copy_spec``):

* conv state (SD layout, conv_width > 0): shift the sliding window by
  ``token_bias`` tokens -- ``state[bt[src_col], token_bias:]`` ->
  ``state[bt[dst_col], :conv_width - token_bias]``.
* temporal state (conv_width == 0): ``token_bias`` selects the accepted
  speculative column -- ``state[bt[src_col + token_bias]]`` ->
  ``state[bt[dst_col]]``.

The kernel must also no-op when ``src_col < 0`` (fresh request) or
``src_col == dst_col`` (no boundary crossed). V2 callers pass an explicit
``idx_mapping``; V1 align preprocessing launches in batch order with
``idx_mapping=None``.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import torch

from vllm.model_executor.layers.mamba import mamba_utils as layer_mamba_utils
from vllm.platforms import current_platform
from vllm.v1.worker import mamba_utils as worker_mamba_utils
from vllm.v1.worker.mamba_utils import precopy_mamba_align_fused_kernel

try:
    import pytest

    _cuda_required = pytest.mark.skipif(
        not current_platform.is_cuda(),
        reason="precopy_mamba_align_fused_kernel needs CUDA/Triton",
    )
    _parametrize = pytest.mark.parametrize
except ModuleNotFoundError:  # allow running directly as ``python <thisfile>``
    pytest = None

    def _cuda_required(fn):
        return fn

    def _parametrize(_name, _values):
        def _deco(fn):
            return fn

        return _deco


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
@_cuda_required
def test_precopy_matches_v1_copy_specs(
    num_reqs, token_bias, has_idx_mapping, conv_state_dim_first
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
        idx_mapping if has_idx_mapping else None,
        num_reqs,
        COPY_BLOCK_SIZE=1024,
        CONV_STATE_DIM_FIRST=conv_state_dim_first,
        HAS_IDX_MAPPING=has_idx_mapping,
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


class _FakeCpuGpuBuffer:
    def __init__(self, n):
        self.np = np.zeros(n, dtype=np.int32)
        self.gpu = object()
        self.copy_sizes = []

    def copy_to_gpu(self, n=None):
        self.copy_sizes.append(n)
        return self.gpu


class _FakePrecopyContext:
    def __init__(self, n):
        self.is_initialized = True
        self.mamba_group_ids = [0]
        self.mamba_state_idx_buf = _FakeCpuGpuBuffer(n)
        self.precopy_src_col_buf = _FakeCpuGpuBuffer(n)
        self.precopy_token_bias_buf = _FakeCpuGpuBuffer(n)
        self.calls = []

    def initialize_from_forward_context(self, *args, **kwargs):
        raise AssertionError("test context is pre-initialized")

    def run_fused_precopy(
        self,
        *,
        num_reqs,
        state_idx_gpu,
        src_col_gpu,
        token_bias_gpu,
        idx_mapping,
    ):
        self.calls.append(
            {
                "num_reqs": num_reqs,
                "state_idx": self.mamba_state_idx_buf.np[:num_reqs].copy(),
                "src_col": self.precopy_src_col_buf.np[:num_reqs].copy(),
                "token_bias": self.precopy_token_bias_buf.np[:num_reqs].copy(),
                "idx_mapping": idx_mapping,
            }
        )


def _make_preprocess_case(token_bias):
    req_ids = ["fresh", "same", "cross_a", "cross_b"]
    scheduler_output = SimpleNamespace(
        finished_req_ids=set(),
        preempted_req_ids=set(),
        scheduled_cached_reqs=SimpleNamespace(resumed_req_ids=set()),
        num_scheduled_tokens={
            "fresh": 1,
            "same": 1,
            "cross_a": 1,
            "cross_b": 2,
        },
    )
    input_batch = SimpleNamespace(
        req_ids=req_ids,
        num_accepted_tokens_cpu=np.array(
            [token_bias + 1, token_bias + 1, token_bias + 1, 2],
            dtype=np.int32,
        ),
    )
    requests = {
        "fresh": SimpleNamespace(req_id="fresh", num_computed_tokens=0),
        "same": SimpleNamespace(req_id="same", num_computed_tokens=5),
        "cross_a": SimpleNamespace(req_id="cross_a", num_computed_tokens=8),
        "cross_b": SimpleNamespace(req_id="cross_b", num_computed_tokens=7),
    }
    mamba_state_idx = {"same": 1, "cross_a": 0, "cross_b": 1}
    return scheduler_output, input_batch, requests, mamba_state_idx


@_parametrize("token_bias", [1, 2])
def test_preprocess_fused_align_matches_scalar_bookkeeping(monkeypatch, token_bias):
    block_size = 4
    mamba_spec = SimpleNamespace(block_size=block_size, num_speculative_blocks=1)
    cache_config = SimpleNamespace(enable_prefix_caching=True)
    kv_cache_config = SimpleNamespace()
    scalar_copy_calls = []

    def fake_collect(
        copy_bufs,
        kv_cache_config,
        mamba_state_copy_funcs,
        mamba_group_ids,
        src_block_idx,
        dest_block_idx,
        accept_token_bias,
        req_state,
        forward_context,
    ):
        scalar_copy_calls.append(
            (
                req_state.req_id,
                src_block_idx,
                dest_block_idx,
                accept_token_bias,
            )
        )

    monkeypatch.setattr(worker_mamba_utils, "collect_mamba_copy_meta", fake_collect)
    monkeypatch.setattr(
        worker_mamba_utils, "do_mamba_copy_block", lambda copy_bufs: None
    )

    scalar_case = _make_preprocess_case(token_bias)
    fused_case = _make_preprocess_case(token_bias)

    scalar_copy_bufs = SimpleNamespace(
        mamba_group_ids=[0],
        mamba_spec=mamba_spec,
        offset=0,
    )
    worker_mamba_utils.preprocess_mamba(
        scheduler_output=scalar_case[0],
        kv_cache_config=kv_cache_config,
        cache_config=cache_config,
        mamba_state_idx=scalar_case[3],
        input_batch=scalar_case[1],
        requests=scalar_case[2],
        forward_context={},
        mamba_state_copy_funcs=(),
        copy_bufs=scalar_copy_bufs,
    )

    ctx = _FakePrecopyContext(len(fused_case[1].req_ids))
    fused_copy_bufs = SimpleNamespace(
        mamba_group_ids=[0],
        mamba_spec=mamba_spec,
        offset=0,
    )
    worker_mamba_utils.preprocess_mamba(
        scheduler_output=fused_case[0],
        kv_cache_config=kv_cache_config,
        cache_config=cache_config,
        mamba_state_idx=fused_case[3],
        input_batch=fused_case[1],
        requests=fused_case[2],
        forward_context={},
        mamba_state_copy_funcs=(),
        copy_bufs=fused_copy_bufs,
        align_ctx=ctx,
    )

    assert fused_case[3] == scalar_case[3]
    np.testing.assert_array_equal(
        fused_case[1].num_accepted_tokens_cpu,
        scalar_case[1].num_accepted_tokens_cpu,
    )
    assert scalar_copy_calls == [
        ("cross_a", 0, 2, token_bias),
        ("cross_b", 1, 2, 1),
    ]
    assert len(ctx.calls) == 1
    call = ctx.calls[0]
    assert call["num_reqs"] == len(fused_case[1].req_ids)
    assert call["idx_mapping"] is None
    np.testing.assert_array_equal(call["state_idx"], np.array([0, 1, 2, 2]))
    np.testing.assert_array_equal(call["src_col"], np.array([-1, -1, 0, 1]))
    np.testing.assert_array_equal(call["token_bias"], np.array([0, 0, token_bias, 1]))
    fused_copy_calls = [
        (req_id, int(src), int(dst), int(bias))
        for req_id, src, dst, bias in zip(
            fused_case[1].req_ids,
            call["src_col"],
            call["state_idx"],
            call["token_bias"],
        )
        if int(src) != -1 and int(src) != int(dst)
    ]
    assert fused_copy_calls == scalar_copy_calls


if __name__ == "__main__":
    for nr in (1, 4, 16):
        for tb in (0, 1, 2):
            for mapping in (True, False):
                for dim_first in (False, True):
                    test_precopy_matches_v1_copy_specs(nr, tb, mapping, dim_first)
                    print(
                        f"OK num_reqs={nr} token_bias={tb} "
                        f"has_idx_mapping={mapping} conv_dim_first={dim_first}"
                    )
