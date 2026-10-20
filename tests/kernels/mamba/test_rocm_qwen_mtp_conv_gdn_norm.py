# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Qwen GDN MTP verify kernel against stock conv, gated delta, and RMSNorm."""

from __future__ import annotations

import itertools

import pytest
import torch

from vllm.model_executor.layers.layernorm import RMSNormGated
from vllm.model_executor.layers.mamba.ops.causal_conv1d import (
    causal_conv1d_update,
)
from vllm.model_executor.layers.mamba.ops.rocm_qwen_mtp_conv_gdn_norm import (
    launch_conv_gdn_norm,
)
from vllm.platforms import current_platform
from vllm.third_party.flash_linear_attention.ops.fused_sigmoid_gating import (
    fused_sigmoid_gating_delta_rule_update,
)

pytestmark = pytest.mark.skipif(
    not current_platform.is_rocm(),
    reason="ROCm Qwen MTP conv+GDN+norm kernel",
)

H = 16
HV = 48
K = 128
V = 128
SHAPE = (H, HV, K, V)
DIM = 2 * H * K + HV * V
CONV_LEN = 6
EPS = 1e-6
OUTPUT_REL_L2 = 5e-4
SSM_ATOL = 1e-3
SSM_RTOL = 1e-3
# slot * (48 * 128 * 128) overflows int32 at slot 2731.
HIGH_BLOCK = 3000

CASES = [
    pytest.param(1, 1, id="t1"),
    pytest.param(2, 1, id="t2-accept1"),
    pytest.param(2, 2, id="t2-accept2"),
    pytest.param(3, 1, id="t3-accept1"),
    pytest.param(3, 2, id="t3-accept2"),
    pytest.param(4, 1, id="t4-accept1"),
    pytest.param(4, 2, id="t4-accept2"),
    pytest.param(4, 4, id="t4-accept4"),
    # A step where no request drafted still accepts up to num_spec + 1.
    pytest.param(1, 3, id="t1-accept3"),
    pytest.param(2, 4, id="t2-accept4"),
]


def _make_inputs(
    tokens: int,
    accepted: int,
    seed: int,
    shape: tuple[int, int, int, int] = SHAPE,
    dtype: torch.dtype = torch.bfloat16,
    activation: str = "silu",
):
    torch.manual_seed(seed)
    device = "cuda"
    h, hv, k, v = shape
    dim = 2 * h * k + hv * v
    width = max(tokens, accepted)
    mixed = 0.1 * torch.randn(tokens, dim, dtype=dtype, device=device)
    weight = 0.1 * torch.randn(dim, 4, dtype=dtype, device=device)
    conv_row = 0.1 * torch.randn(dim, CONV_LEN, dtype=dtype, device=device)
    ssm_rows = 0.01 * torch.randn(width, hv, v, k, dtype=torch.float32, device=device)
    # chunk leaves a and b non-contiguous, as split_ba does on the model path.
    ba = 0.1 * torch.randn(tokens, 2 * hv, dtype=dtype, device=device)
    b, a = ba.chunk(2, dim=-1)
    if tokens > 1:
        assert not a.is_contiguous()
        assert not b.is_contiguous()
    a_log = 0.5 * torch.randn(hv, dtype=torch.float32, device=device)
    dt_bias = 0.1 * torch.randn(hv, dtype=torch.float32, device=device)
    z = 0.1 * torch.randn(tokens, hv, v, dtype=dtype, device=device)
    norm_w = torch.randn(v, dtype=torch.float32, device=device)
    norm = RMSNormGated(
        v,
        eps=EPS,
        group_size=None,
        norm_before_gate=True,
        activation=activation,
        device=device,
        dtype=torch.float32,
    )
    norm.weight = torch.nn.Parameter(norm_w, requires_grad=False)
    accepted_t = torch.tensor([accepted], dtype=torch.int32, device=device)
    return mixed, weight, conv_row, ssm_rows, a, b, a_log, dt_bias, z, norm, accepted_t


def _pool(
    n_slots: int,
    shape: tuple[int, int, int, int] = SHAPE,
    dtype: torch.dtype = torch.bfloat16,
    token_major: bool = False,
):
    device = "cuda"
    h, hv, k, v = shape
    dim = 2 * h * k + hv * v
    if token_major:
        conv = torch.empty(
            n_slots, CONV_LEN, dim, dtype=dtype, device=device
        ).transpose(1, 2)
    else:
        conv = torch.empty(n_slots, dim, CONV_LEN, dtype=dtype, device=device)
    ssm = torch.empty(n_slots, hv, v, k, dtype=torch.float32, device=device)
    return conv, ssm


def _plant(conv, ssm, block: int, conv_row, ssm_rows) -> None:
    tokens = ssm_rows.shape[0]
    conv[block].copy_(conv_row)
    ssm[block : block + tokens].copy_(ssm_rows)


def _indices(block: int, tokens: int) -> torch.Tensor:
    return torch.arange(block, block + tokens, dtype=torch.int32, device="cuda").view(
        1, tokens
    )


def _run_stock(
    mixed,
    weight,
    conv,
    a,
    b,
    a_log,
    dt_bias,
    ssm,
    indices,
    accepted,
    z,
    norm: RMSNormGated,
    out,
) -> None:
    tokens = mixed.shape[0]
    _, hv, v, k = ssm.shape
    h = (mixed.shape[1] - hv * v) // (2 * k)
    cu = torch.tensor([0, tokens], dtype=torch.int32, device=mixed.device)
    y = causal_conv1d_update(
        mixed,
        conv,
        weight,
        None,
        "silu",
        conv_state_indices=indices[:, 0],
        num_accepted_tokens=accepted,
        query_start_loc=cu,
        max_query_len=tokens,
        validate_data=False,
    )
    q = y[:, : h * k].view(1, tokens, h, k)
    key = y[:, h * k : 2 * h * k].view(1, tokens, h, k)
    value = y[:, 2 * h * k :].view(1, tokens, hv, v)
    core, _ = fused_sigmoid_gating_delta_rule_update(
        A_log=a_log,
        a=a,
        b=b,
        dt_bias=dt_bias,
        q=q,
        k=key,
        v=value,
        scale=k**-0.5,
        initial_state=ssm,
        inplace_final_state=True,
        cu_seqlens=cu,
        ssm_state_indices=indices,
        num_accepted_tokens=accepted,
        use_qk_l2norm_in_kernel=True,
    )
    if core.dim() == 4:
        core = core.squeeze(0)
    out.copy_(norm.forward_cuda(core, z))


def _run_fused(
    mixed,
    weight,
    conv,
    a,
    b,
    a_log,
    dt_bias,
    ssm,
    indices,
    accepted,
    z,
    norm: RMSNormGated,
    out,
) -> None:
    launch_conv_gdn_norm(
        mixed,
        weight,
        conv,
        a,
        b,
        a_log,
        dt_bias,
        indices,
        accepted,
        ssm,
        out,
        z,
        norm.weight,
        EPS,
        out,
        gate_sigmoid=norm.activation == "sigmoid",
    )


def _snapshot(conv, ssm, block: int, rows: int):
    return (
        conv[block].clone(),
        ssm[block : block + rows].clone(),
    )


def _assert_close(actual, expected, conv_a, conv_e, ssm_a, ssm_e, where: str) -> None:
    assert not torch.isnan(actual).any(), f"NaN in fused output at {where}"
    output_error = (actual.float() - expected.float()).norm()
    output_relative_l2 = output_error / expected.float().norm().clamp_min(1e-20)
    assert output_relative_l2 < OUTPUT_REL_L2, (
        f"output relative L2 mismatch at {where}: {output_relative_l2.item():.6g}"
    )
    torch.testing.assert_close(conv_a, conv_e, atol=0, rtol=0)
    torch.testing.assert_close(ssm_a, ssm_e, atol=SSM_ATOL, rtol=SSM_RTOL)


def _compare_at_block(inputs, conv, ssm, block: int) -> tuple[torch.Tensor, ...]:
    mixed, weight, conv_row, ssm_rows, a, b, a_log, dt_bias, z, norm, accepted = inputs
    tokens = mixed.shape[0]
    width = ssm_rows.shape[0]
    indices = _indices(block, width)
    _plant(conv, ssm, block, conv_row, ssm_rows)

    fused_out = torch.empty_like(z)
    _run_fused(
        mixed.clone(),
        weight,
        conv,
        a,
        b,
        a_log,
        dt_bias,
        ssm,
        indices,
        accepted,
        z,
        norm,
        fused_out,
    )
    fused_conv, fused_ssm = _snapshot(conv, ssm, block, width)

    _plant(conv, ssm, block, conv_row, ssm_rows)
    stock_out = torch.empty_like(fused_out)
    _run_stock(
        mixed.clone(),
        weight,
        conv,
        a,
        b,
        a_log,
        dt_bias,
        ssm,
        indices,
        accepted,
        z,
        norm,
        stock_out,
    )
    stock_conv, stock_ssm = _snapshot(conv, ssm, block, width)
    where = f"block {block} T={tokens} accepted={int(accepted.item())}"
    _assert_close(
        fused_out, stock_out, fused_conv, stock_conv, fused_ssm, stock_ssm, where
    )
    return fused_out, fused_conv, fused_ssm


@pytest.mark.parametrize("tokens,accepted", CASES)
@torch.inference_mode()
def test_matches_stock_ops(tokens, accepted, default_vllm_config):
    del default_vllm_config
    inputs = _make_inputs(tokens, accepted, seed=100 + 10 * tokens + accepted)
    conv, ssm = _pool(max(tokens, accepted) + 2)
    _compare_at_block(inputs, conv, ssm, block=1)


@pytest.mark.parametrize("tokens,accepted", CASES)
@torch.inference_mode()
def test_high_block_id_matches_block_one(tokens, accepted, default_vllm_config):
    del default_vllm_config
    inputs = _make_inputs(tokens, accepted, seed=200 + 10 * tokens + accepted)
    conv, ssm = _pool(HIGH_BLOCK + max(tokens, accepted))
    low = _compare_at_block(inputs, conv, ssm, block=1)
    high = _compare_at_block(inputs, conv, ssm, block=HIGH_BLOCK)
    where = f"T={tokens} accepted={accepted}"
    torch.testing.assert_close(high[0], low[0], atol=0, rtol=0, msg=where)
    torch.testing.assert_close(high[1], low[1], atol=0, rtol=0, msg=where)
    torch.testing.assert_close(high[2], low[2], atol=0, rtol=0, msg=where)


@pytest.mark.parametrize(
    "shape,dtype",
    [
        pytest.param((16, 32, 128, 128), torch.bfloat16, id="h16-hv32"),
        pytest.param((8, 16, 64, 96), torch.bfloat16, id="k64-v96"),
        pytest.param((4, 8, 256, 128), torch.bfloat16, id="k256"),
        pytest.param(SHAPE, torch.float16, id="fp16"),
    ],
)
@pytest.mark.parametrize("tokens,accepted", [(1, 1), (4, 2)])
@torch.inference_mode()
def test_other_shapes_match_stock_ops(
    shape, dtype, tokens, accepted, default_vllm_config
):
    """Head counts, head sizes, and dtype come from the tensors."""
    del default_vllm_config
    inputs = _make_inputs(tokens, accepted, seed=400 + tokens, shape=shape, dtype=dtype)
    conv, ssm = _pool(tokens + 2, shape=shape, dtype=dtype)
    _compare_at_block(inputs, conv, ssm, block=1)


@pytest.mark.parametrize("tokens,accepted", [(1, 1), (4, 2)])
@torch.inference_mode()
def test_sigmoid_gate_matches_stock_ops(tokens, accepted, default_vllm_config):
    del default_vllm_config
    inputs = _make_inputs(tokens, accepted, seed=500 + tokens, activation="sigmoid")
    conv, ssm = _pool(tokens + 2)
    _compare_at_block(inputs, conv, ssm, block=1)


@pytest.mark.parametrize("tokens,accepted", [(1, 1), (4, 2)])
@torch.inference_mode()
def test_token_major_conv_state_matches_stock_ops(
    tokens, accepted, default_vllm_config
):
    """Conv state viewed as ``[slots, dim, len]`` over a ``[slots, len, dim]``
    allocation, as when conv state is not dim-first."""
    del default_vllm_config
    inputs = _make_inputs(tokens, accepted, seed=600 + tokens)
    conv, ssm = _pool(tokens + 2, token_major=True)
    _compare_at_block(inputs, conv, ssm, block=1)


def _compare_packed(
    batch: int,
    lengths: list[int],
    accepted: int,
    block0: int,
    pad: int = 0,
    null_seqs: int = 0,
) -> None:
    """``pad`` trailing rows sit outside every sequence, like graph padding;
    the varlen path must leave them untouched. The last ``null_seqs``
    sequences point at NULL_BLOCK_ID 0, like padded requests, and must
    produce zero output without touching slot 0."""
    torch.manual_seed(300 + batch * 17 + block0 + sum(lengths))
    device = "cuda"
    total = sum(lengths)
    width = max(lengths)
    rows_total = total + pad
    mixed = 0.1 * torch.randn(rows_total, DIM, dtype=torch.bfloat16, device=device)
    weight = 0.1 * torch.randn(DIM, 4, dtype=torch.bfloat16, device=device)
    ba = 0.1 * torch.randn(rows_total, 2 * HV, dtype=torch.bfloat16, device=device)
    b, a = ba.chunk(2, dim=-1)
    a_log = 0.5 * torch.randn(HV, dtype=torch.float32, device=device)
    dt_bias = 0.1 * torch.randn(HV, dtype=torch.float32, device=device)
    z = 0.1 * torch.randn(rows_total, HV, V, dtype=torch.bfloat16, device=device)
    norm_w = torch.randn(V, dtype=torch.float32, device=device)
    norm = RMSNormGated(
        V,
        eps=EPS,
        group_size=None,
        norm_before_gate=True,
        activation="silu",
        device=device,
        dtype=torch.float32,
    )
    norm.weight = torch.nn.Parameter(norm_w, requires_grad=False)
    accepted_t = torch.full((batch,), accepted, dtype=torch.int32, device=device)
    real = batch - null_seqs
    rows = []
    cursor = block0
    for length in lengths[:real]:
        rows.append(torch.arange(cursor, cursor + length, device=device))
        cursor += length
    rows += [
        torch.zeros(length, dtype=torch.long, device=device)
        for length in lengths[real:]
    ]
    indices = torch.stack(
        [torch.nn.functional.pad(row, (0, width - row.numel())) for row in rows]
    ).to(torch.int32)
    pool = cursor + 2
    conv_row = 0.1 * torch.randn(
        batch, DIM, CONV_LEN, dtype=torch.bfloat16, device=device
    )
    ssm_rows = [
        0.01 * torch.randn(length, HV, V, K, dtype=torch.float32, device=device)
        for length in lengths
    ]

    def plant(conv, ssm) -> None:
        conv[0].zero_()
        ssm[0].zero_()
        for seq, length in enumerate(lengths[:real]):
            block = int(indices[seq, 0])
            conv[block].copy_(conv_row[seq])
            ssm[block : block + length].copy_(ssm_rows[seq])

    conv_f, ssm_f = _pool(pool)
    conv_s, ssm_s = _pool(pool)
    plant(conv_f, ssm_f)
    plant(conv_s, ssm_s)
    fused_out = torch.zeros(rows_total, HV, V, dtype=torch.bfloat16, device=device)
    varlen = len(set(lengths)) > 1 or pad > 0
    cu_seqlens = None
    if varlen:
        cu_seqlens = torch.tensor(
            [0, *itertools.accumulate(lengths)], dtype=torch.int32, device=device
        )
    launch_conv_gdn_norm(
        mixed if varlen else mixed[:total],
        weight,
        conv_f,
        a if varlen else a[:total],
        b if varlen else b[:total],
        a_log,
        dt_bias,
        indices if varlen else indices[:, : lengths[0]],
        accepted_t,
        ssm_f,
        fused_out if varlen else fused_out[:total],
        z if varlen else z[:total],
        norm.weight,
        EPS,
        fused_out if varlen else fused_out[:total],
        gate_sigmoid=False,
        n_seq=batch,
        cu_seqlens=cu_seqlens,
    )
    assert not fused_out[total:].any(), "rows outside every sequence were written"
    real_rows = sum(lengths[:real])
    assert not fused_out[real_rows:].any(), "NULL_BLOCK_ID sequences wrote output"
    assert not conv_f[0].any() and not ssm_f[0].any(), "NULL_BLOCK_ID was written"

    offset = 0
    for seq, length in enumerate(lengths[:real]):
        block = int(indices[seq, 0])
        stock_out = torch.empty(length, HV, V, dtype=torch.bfloat16, device=device)
        _run_stock(
            mixed[offset : offset + length],
            weight,
            conv_s,
            a[offset : offset + length],
            b[offset : offset + length],
            a_log,
            dt_bias,
            ssm_s,
            indices[seq : seq + 1, :length],
            accepted_t[seq : seq + 1],
            z[offset : offset + length],
            norm,
            stock_out,
        )
        where = f"batch {batch} seq {seq} T={length} block {block}"
        _assert_close(
            fused_out[offset : offset + length],
            stock_out,
            conv_f[block],
            conv_s[block],
            ssm_f[block : block + length],
            ssm_s[block : block + length],
            where,
        )
        offset += length


@pytest.mark.parametrize("batch", [8, 32])
@pytest.mark.parametrize("tokens", [1, 4])
@torch.inference_mode()
def test_uniform_batch_matches_stock_ops(batch, tokens, default_vllm_config):
    del default_vllm_config
    _compare_packed(batch, [tokens] * batch, accepted=1, block0=1)


@pytest.mark.parametrize("batch", [4])
@torch.inference_mode()
def test_uniform_batch_high_block_id(batch, default_vllm_config):
    del default_vllm_config
    _compare_packed(batch, [4] * batch, accepted=2, block0=HIGH_BLOCK)


@torch.inference_mode()
def test_mixed_widths_match_stock_ops(default_vllm_config):
    del default_vllm_config
    _compare_packed(5, [4, 1, 4, 2, 3], accepted=1, block0=1)


@torch.inference_mode()
def test_mixed_widths_leave_padding_rows(default_vllm_config):
    del default_vllm_config
    _compare_packed(4, [2, 4, 1, 3], accepted=1, block0=HIGH_BLOCK, pad=5)


@pytest.mark.parametrize("lengths", [[4] * 6, [2, 4, 1, 3, 2, 2]])
@torch.inference_mode()
def test_null_block_sequences_are_skipped(lengths, default_vllm_config):
    del default_vllm_config
    _compare_packed(len(lengths), lengths, accepted=1, block0=1, null_seqs=2)
