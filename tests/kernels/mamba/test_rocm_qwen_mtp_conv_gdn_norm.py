# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Qwen3.8-27B MTP verify kernel against stock conv, gated delta, and RMSNorm."""

from __future__ import annotations

import pytest
import torch

from vllm.model_executor.layers.layernorm import RMSNormGated
from vllm.model_executor.layers.mamba.ops.causal_conv1d import (
    causal_conv1d_update,
)
from vllm.model_executor.layers.mamba.ops.rocm_qwen_mtp_conv_gdn_norm import (
    launch_conv_gdn_norm,
    launch_conv_gdn_norm_varlen,
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
]


def _make_inputs(tokens: int, accepted: int, seed: int):
    torch.manual_seed(seed)
    device = "cuda"
    mixed = 0.1 * torch.randn(tokens, DIM, dtype=torch.bfloat16, device=device)
    weight = 0.1 * torch.randn(DIM, 4, dtype=torch.bfloat16, device=device)
    conv_row = 0.1 * torch.randn(DIM, CONV_LEN, dtype=torch.bfloat16, device=device)
    ssm_rows = 0.01 * torch.randn(
        tokens, HV, V, K, dtype=torch.float32, device=device
    )
    # chunk leaves a and b non-contiguous, as split_ba does on the model path.
    ba = 0.1 * torch.randn(tokens, 2 * HV, dtype=torch.bfloat16, device=device)
    b, a = ba.chunk(2, dim=-1)
    if tokens > 1:
        assert not a.is_contiguous()
        assert not b.is_contiguous()
    a_log = 0.5 * torch.randn(HV, dtype=torch.float32, device=device)
    dt_bias = 0.1 * torch.randn(HV, dtype=torch.float32, device=device)
    z = 0.1 * torch.randn(tokens, HV, V, dtype=torch.bfloat16, device=device)
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
    accepted_t = torch.tensor([accepted], dtype=torch.int32, device=device)
    return mixed, weight, conv_row, ssm_rows, a, b, a_log, dt_bias, z, norm, accepted_t


def _pool(n_slots: int):
    device = "cuda"
    conv = torch.empty(n_slots, DIM, CONV_LEN, dtype=torch.bfloat16, device=device)
    ssm = torch.empty(n_slots, HV, V, K, dtype=torch.float32, device=device)
    return conv, ssm


def _plant(conv, ssm, block: int, conv_row, ssm_rows) -> None:
    tokens = ssm_rows.shape[0]
    conv[block].copy_(conv_row)
    ssm[block : block + tokens].copy_(ssm_rows)


def _indices(block: int, tokens: int) -> torch.Tensor:
    return torch.arange(
        block, block + tokens, dtype=torch.int32, device="cuda"
    ).view(1, tokens)


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
    q = y[:, : H * K].view(1, tokens, H, K)
    k = y[:, H * K : 2 * H * K].view(1, tokens, H, K)
    v = y[:, 2 * H * K :].view(1, tokens, HV, V)
    core, _ = fused_sigmoid_gating_delta_rule_update(
        A_log=a_log,
        a=a,
        b=b,
        dt_bias=dt_bias,
        q=q,
        k=k,
        v=v,
        scale=K**-0.5,
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
    tokens = mixed.shape[0]
    core = torch.empty(tokens, HV, V, dtype=torch.bfloat16, device=mixed.device)
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
        core,
        z,
        norm.weight,
        EPS,
        out,
        gate_sigmoid=False,
    )


def _snapshot(conv, ssm, block: int, tokens: int):
    return (
        conv[block].clone(),
        ssm[block : block + tokens].clone(),
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
    indices = _indices(block, tokens)
    _plant(conv, ssm, block, conv_row, ssm_rows)

    fused_out = torch.empty(tokens, HV, V, dtype=torch.bfloat16, device="cuda")
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
    fused_conv, fused_ssm = _snapshot(conv, ssm, block, tokens)

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
    stock_conv, stock_ssm = _snapshot(conv, ssm, block, tokens)
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
    conv, ssm = _pool(tokens + 2)
    _compare_at_block(inputs, conv, ssm, block=1)


@pytest.mark.parametrize("tokens,accepted", CASES)
@torch.inference_mode()
def test_high_block_id_matches_block_one(tokens, accepted, default_vllm_config):
    del default_vllm_config
    inputs = _make_inputs(tokens, accepted, seed=200 + 10 * tokens + accepted)
    conv, ssm = _pool(HIGH_BLOCK + tokens)
    low = _compare_at_block(inputs, conv, ssm, block=1)
    high = _compare_at_block(inputs, conv, ssm, block=HIGH_BLOCK)
    where = f"T={tokens} accepted={accepted}"
    torch.testing.assert_close(high[0], low[0], atol=0, rtol=0, msg=where)
    torch.testing.assert_close(high[1], low[1], atol=0, rtol=0, msg=where)
    torch.testing.assert_close(high[2], low[2], atol=0, rtol=0, msg=where)


def _compare_packed(batch: int, lengths: list[int], accepted: int, block0: int) -> None:
    torch.manual_seed(300 + batch * 17 + block0 + sum(lengths))
    device = "cuda"
    total = sum(lengths)
    width = max(lengths)
    mixed = 0.1 * torch.randn(total, DIM, dtype=torch.bfloat16, device=device)
    weight = 0.1 * torch.randn(DIM, 4, dtype=torch.bfloat16, device=device)
    ba = 0.1 * torch.randn(total, 2 * HV, dtype=torch.bfloat16, device=device)
    b, a = ba.chunk(2, dim=-1)
    a_log = 0.5 * torch.randn(HV, dtype=torch.float32, device=device)
    dt_bias = 0.1 * torch.randn(HV, dtype=torch.float32, device=device)
    z = 0.1 * torch.randn(total, HV, V, dtype=torch.bfloat16, device=device)
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
    rows = []
    cursor = block0
    for length in lengths:
        rows.append(torch.arange(cursor, cursor + length, device=device))
        cursor += length
    indices = torch.stack(
        [
            torch.nn.functional.pad(row, (0, width - row.numel()))
            for row in rows
        ]
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
        for seq, length in enumerate(lengths):
            block = int(indices[seq, 0])
            conv[block].copy_(conv_row[seq])
            ssm[block : block + length].copy_(ssm_rows[seq])

    conv_f, ssm_f = _pool(pool)
    conv_s, ssm_s = _pool(pool)
    plant(conv_f, ssm_f)
    plant(conv_s, ssm_s)
    fused_out = torch.empty(total, HV, V, dtype=torch.bfloat16, device=device)
    core = torch.empty_like(fused_out)
    if len(set(lengths)) == 1:
        launch_conv_gdn_norm(
            mixed, weight, conv_f, a, b, a_log, dt_bias, indices[:, : lengths[0]],
            accepted_t, ssm_f, core, z, norm.weight, EPS, fused_out,
            gate_sigmoid=False, n_seq=batch,
        )
    else:
        launch_conv_gdn_norm_varlen(
            mixed, weight, conv_f, a, b, a_log, dt_bias, indices, accepted_t,
            ssm_f, z, norm.weight, EPS, fused_out, lengths, gate_sigmoid=False,
        )

    offset = 0
    for seq, length in enumerate(lengths):
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
