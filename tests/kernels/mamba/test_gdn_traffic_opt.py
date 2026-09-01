# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the GDN traffic-opt weight-layout transforms (C9 de-interleave +
C6 tiny-GEMM concat).

Covers:
 - permutation round-trip vs the interleaved unpack (bit-exact)
 - GEMM pipeline equivalence old-vs-new at real serve dtypes (C9 must be
   bit-exact; C6 is torch.equal-first with ULP quantification on failure)
 - conv/gating decode kernels accepting row-strided mixed_qkv (the new
   regime: mixed_qkv is a view into the projection output)
 - MoE router+shared-expert-gate weight concat math
 - view-identity dispatch asserts (the unpack glue copies are gone by
   construction when the flags are on)
"""

import os

import pytest
import torch

from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import (
    QwenGatedDeltaNetAttention,
    build_ba_deinterleave_perm,
    build_qkvz_deinterleave_perm,
    gdn_concat_router_gate_enabled,
    gdn_concat_tiny_gemms_enabled,
    gdn_deinterleave_qkvz_enabled,
)

H, HV, K, V = 16, 32, 128, 128  # TP1 Qwen3-Next-80B GDN geometry
NG = H
NVG = HV // H
HIDDEN = 2048
QKVZ_N = 2 * H * K + 2 * HV * V  # 12288
QKV_N = 2 * H * K + HV * V  # 8192
BA_N = 2 * HV  # 64
T = 4  # spec tokens per seq (k=3 + bonus)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires CUDA"
)


class Duck:
    gqa_interleaved_layout = True
    num_k_heads = H
    num_v_heads = HV
    head_k_dim = K
    head_v_dim = V
    tp_size = 1
    key_dim = H * K
    value_dim = HV * V


def _unpack_interleaved(qkvz: torch.Tensor, ba: torch.Tensor):
    """The exact old (interleaved) unpack, via the module's own method."""
    fn = QwenGatedDeltaNetAttention.prepare_gdn_attention_core_inputs
    return fn(Duck(), qkvz, ba, qkvz.shape[0])


def _split_flat(qkvz: torch.Tensor, ba: torch.Tensor):
    """The new (de-interleaved) unpack: views only."""
    mixed_qkv, z = qkvz.split([QKV_N, HV * V], dim=-1)
    z = z.reshape(z.size(0), HV, V)
    b, a = ba.chunk(2, dim=-1)
    return mixed_qkv, z, b.contiguous(), a.contiguous()


def _drift_stats(x: torch.Tensor, y: torch.Tensor) -> dict:
    """Quantify bf16 drift: max abs diff, max diff relative to the tensor
    scale, and the fraction of exactly-equal elements. (A raw int16-bit
    "ULP" distance misreads sign flips near zero, so scale-relative abs
    diff is the honest metric for reassociation drift.)"""
    xf, yf = x.float(), y.float()
    d = (xf - yf).abs()
    scale = yf.abs().max().clamp_min(1e-12)
    return {
        "max_abs": d.max().item(),
        "max_rel_to_scale": (d.max() / scale).item(),
        "equal_frac": (x == y).float().mean().item(),
    }


def test_flag_defaults_and_off(monkeypatch):
    monkeypatch.delenv("VLLM_GDN_DEINTERLEAVE_QKVZ", raising=False)
    monkeypatch.delenv("VLLM_GDN_CONCAT_TINY_GEMMS", raising=False)
    monkeypatch.delenv("VLLM_GDN_CONCAT_ROUTER_GATE", raising=False)
    assert gdn_deinterleave_qkvz_enabled()
    assert gdn_concat_tiny_gemms_enabled()
    # router-gate fold is default OFF (measured in-graph negative unpadded,
    # marginal padded, plus routing-logit drift)
    assert not gdn_concat_router_gate_enabled()
    monkeypatch.setenv("VLLM_GDN_DEINTERLEAVE_QKVZ", "0")
    monkeypatch.setenv("VLLM_GDN_CONCAT_TINY_GEMMS", "0")
    monkeypatch.setenv("VLLM_GDN_CONCAT_ROUTER_GATE", "1")
    assert not gdn_deinterleave_qkvz_enabled()
    assert not gdn_concat_tiny_gemms_enabled()
    assert gdn_concat_router_gate_enabled()


@pytest.mark.parametrize("ntok", [1, 7, 128])
def test_deinterleave_perm_round_trip(ntok):
    """Permuting the projection output columns must reproduce exactly what
    the interleaved unpack produced (bit-exact, pure data movement)."""
    torch.manual_seed(0)
    dev = torch.device("cuda")
    qkvz = torch.randn(ntok, QKVZ_N, dtype=torch.bfloat16, device=dev)
    ba = torch.randn(ntok, BA_N, dtype=torch.bfloat16, device=dev)

    mq_old, z_old, b_old, a_old = _unpack_interleaved(qkvz, ba)

    perm_qkvz = build_qkvz_deinterleave_perm(NG, K, V, NVG, device=dev)
    perm_ba = build_ba_deinterleave_perm(NG, NVG, device=dev)
    mq_new, z_new, b_new, a_new = _split_flat(qkvz[:, perm_qkvz], ba[:, perm_ba])

    assert torch.equal(mq_old, mq_new)
    assert torch.equal(z_old, z_new)
    assert torch.equal(b_old, b_new)
    assert torch.equal(a_old, a_new)


@pytest.mark.parametrize("ntok", [16, 128])
def test_gemm_pipeline_c9_bit_exact(ntok):
    """C9 alone: row-permuted weights -> same GEMM shape -> outputs must be
    bit-exact against the old two-GEMM + interleaved-unpack pipeline."""
    torch.manual_seed(1)
    dev = torch.device("cuda")
    w_qkvz = torch.randn(QKVZ_N, HIDDEN, dtype=torch.bfloat16, device=dev) * 0.02
    w_ba = torch.randn(BA_N, HIDDEN, dtype=torch.bfloat16, device=dev) * 0.02
    h = torch.randn(ntok, HIDDEN, dtype=torch.bfloat16, device=dev)

    qkvz_old = torch.nn.functional.linear(h, w_qkvz)
    ba_old = torch.nn.functional.linear(h, w_ba)
    mq_old, z_old, b_old, a_old = _unpack_interleaved(qkvz_old, ba_old)

    perm_qkvz = build_qkvz_deinterleave_perm(NG, K, V, NVG, device=dev)
    perm_ba = build_ba_deinterleave_perm(NG, NVG, device=dev)
    qkvz_new = torch.nn.functional.linear(h, w_qkvz[perm_qkvz].contiguous())
    ba_new = torch.nn.functional.linear(h, w_ba[perm_ba].contiguous())
    mq_new, z_new, b_new, a_new = _split_flat(qkvz_new, ba_new)

    assert torch.equal(mq_old, mq_new), "C9 must be bit-exact (row perm only)"
    assert torch.equal(z_old, z_new)
    assert torch.equal(b_old, b_new)
    assert torch.equal(a_old, a_new)
    # dispatch assert: the new unpack is views into the GEMM output
    assert mq_new.data_ptr() == qkvz_new.data_ptr()
    assert z_new.data_ptr() == qkvz_new.data_ptr() + QKV_N * qkvz_new.element_size()


@pytest.mark.parametrize("ntok", [16, 128])
def test_gemm_pipeline_c6_concat(ntok):
    """C6 (+C9): one fused [12352, hidden] GEMM. torch.equal first; if the
    larger N picks a different tile config, quantify the ULP delta (must
    stay ULP-class) and report."""
    torch.manual_seed(2)
    dev = torch.device("cuda")
    w_qkvz = torch.randn(QKVZ_N, HIDDEN, dtype=torch.bfloat16, device=dev) * 0.02
    w_ba = torch.randn(BA_N, HIDDEN, dtype=torch.bfloat16, device=dev) * 0.02
    h = torch.randn(ntok, HIDDEN, dtype=torch.bfloat16, device=dev)

    perm_qkvz = build_qkvz_deinterleave_perm(NG, K, V, NVG, device=dev)
    perm_ba = build_ba_deinterleave_perm(NG, NVG, device=dev)
    w_qkvz_p = w_qkvz[perm_qkvz].contiguous()
    w_ba_p = w_ba[perm_ba].contiguous()

    ref_qkvz = torch.nn.functional.linear(h, w_qkvz_p)
    ref_ba = torch.nn.functional.linear(h, w_ba_p)

    w_fused = torch.cat([w_qkvz_p, w_ba_p], dim=0)
    out = torch.nn.functional.linear(h, w_fused)
    got_qkvz = out[:, :QKVZ_N]
    got_ba = out[:, QKVZ_N:]

    bitexact = torch.equal(got_qkvz, ref_qkvz) and torch.equal(got_ba, ref_ba)
    if not bitexact:
        s_qkvz = _drift_stats(got_qkvz.contiguous(), ref_qkvz)
        s_ba = _drift_stats(got_ba.contiguous(), ref_ba)
        print(
            f"\nC6 in_proj concat NOT bit-exact at ntok={ntok}: "
            f"qkvz={s_qkvz} ba={s_ba}"
        )
        assert s_qkvz["max_rel_to_scale"] <= 2**-7, (
            "C6 in_proj drift exceeds ULP-class; investigate before shipping"
        )
        assert s_ba["max_rel_to_scale"] <= 2**-7
    # dispatch assert: both consumers are views of ONE GEMM output
    assert got_qkvz.data_ptr() == out.data_ptr()
    assert got_ba.data_ptr() == out.data_ptr() + QKVZ_N * out.element_size()


@pytest.mark.parametrize("ntok", [16, 128])
def test_moe_gate_concat(ntok):
    """C6 router-gate fold: [512+1, hidden] fused logits vs the two
    separate GEMMs. torch.equal first, ULP quantification on failure."""
    torch.manual_seed(3)
    dev = torch.device("cuda")
    w_gate = torch.randn(512, HIDDEN, dtype=torch.bfloat16, device=dev) * 0.02
    w_seg = torch.randn(1, HIDDEN, dtype=torch.bfloat16, device=dev) * 0.02
    h = torch.randn(ntok, HIDDEN, dtype=torch.bfloat16, device=dev)

    ref_logits = torch.nn.functional.linear(h, w_gate)
    ref_seg = torch.nn.functional.linear(h, w_seg)

    # padded fold (N 513 -> 528), as the runner builds it
    combined = torch.cat([w_gate, w_seg], dim=0)
    pad = (-combined.shape[0]) % 16
    combined = torch.cat(
        [combined, combined.new_zeros(pad, combined.shape[1])], dim=0
    )
    fused = torch.nn.functional.linear(h, combined)
    got_logits = fused[:, :512].contiguous()
    got_seg = fused[:, 512:513]

    bitexact = torch.equal(got_logits, ref_logits) and torch.equal(
        got_seg.contiguous(), ref_seg
    )
    if not bitexact:
        # KNOWN + REPORTED: N=512 vs N=513 picks a different cuBLAS
        # reduction, so reassociation drift is expected. The bound below is
        # ULP-class relative to the logit scale; near-zero logits may flip
        # sign. Whether this flips near-tie top-k routing picks is
        # validated by the greedy-sha e2e comparison (report both).
        s_l = _drift_stats(got_logits, ref_logits)
        s_s = _drift_stats(got_seg.contiguous(), ref_seg)
        print(
            f"\nMoE gate concat NOT bit-exact at ntok={ntok}: "
            f"logits={s_l} segate={s_s}"
        )
        assert s_l["max_rel_to_scale"] <= 2**-7
        assert s_s["max_rel_to_scale"] <= 2**-7
    # relocated sigmoid scaling is the same op on the same values
    se_out = torch.randn(ntok, HIDDEN, dtype=torch.bfloat16, device=dev)
    old = torch.nn.functional.sigmoid(ref_seg) * se_out
    new = torch.sigmoid(got_seg) * se_out
    if torch.equal(ref_seg, got_seg.contiguous()):
        assert torch.equal(old, new)


@pytest.mark.parametrize("batch", [4, 32])
def test_decode_kernels_accept_row_strided_qkv(batch):
    """The new regime feeds the conv+gating decode pair a row-strided
    mixed_qkv view (token stride 12352, stride(-1)==1). Outputs and final
    states must be bit-identical to the contiguous input."""
    from vllm.third_party.flash_linear_attention.ops import (
        fused_sigmoid_gating_delta_rule_update,
    )
    from vllm.model_executor.layers.mamba.ops.causal_conv1d import (
        causal_conv1d_update,
    )

    torch.manual_seed(4)
    dev = torch.device("cuda")
    ntok = batch * T
    DIM = QKV_N
    W = 4
    WT = 16
    NB = batch * WT + 16

    # row-strided view: mixed_qkv occupies the first DIM cols of a wider
    # (fused-GEMM-shaped) buffer
    wide = torch.randn(ntok, QKVZ_N + BA_N, dtype=torch.bfloat16, device=dev)
    x_strided = wide[:, :DIM]
    x_contig = x_strided.contiguous()
    assert x_strided.stride(0) == QKVZ_N + BA_N

    a = torch.rand(ntok, HV, dtype=torch.bfloat16, device=dev)
    b = torch.rand(ntok, HV, dtype=torch.bfloat16, device=dev)
    A_log = torch.rand(HV, dtype=torch.float32, device=dev)
    dtb = torch.rand(HV, dtype=torch.float32, device=dev)
    wgt = torch.rand(DIM, W, dtype=torch.bfloat16, device=dev)
    bias = torch.rand(DIM, dtype=torch.bfloat16, device=dev)
    cu = torch.arange(0, ntok + 1, T, dtype=torch.int32, device=dev)
    acc = torch.full((batch,), T, dtype=torch.int32, device=dev)

    ids = torch.randperm(NB - 1, device=dev)[: batch * WT].int() + 1
    table = ids.view(batch, WT).contiguous()
    pa = torch.stack(
        [
            torch.zeros(batch, dtype=torch.int32, device=dev),
            torch.ones(batch, dtype=torch.int32, device=dev),
        ],
        1,
    ).contiguous()

    outs = {}
    for name, x in (("strided", x_strided), ("contig", x_contig)):
        torch.manual_seed(5)  # identical pools per arm
        conv_pool = torch.rand(
            NB, DIM, W - 1 + T - 1, dtype=torch.bfloat16, device=dev
        )
        ssm_pool = torch.rand(NB, HV, V, K, dtype=torch.bfloat16, device=dev)
        conv_out = causal_conv1d_update(
            x,
            conv_pool,
            wgt,
            bias,
            "silu",
            conv_state_indices=table,
            packed_anchors=pa,
            num_accepted_tokens=acc,
            query_start_loc=cu,
            max_query_len=T,
        )
        core, _ = fused_sigmoid_gating_delta_rule_update(
            A_log=A_log,
            a=a,
            b=b,
            dt_bias=dtb,
            mixed_qkv=conv_out,
            num_qk_heads=H,
            head_qk_dim=K,
            num_v_heads=HV,
            head_v_dim=V,
            initial_state=ssm_pool,
            inplace_final_state=True,
            cu_seqlens=cu,
            block_table=table,
            packed_anchors=pa,
            num_accepted_tokens=acc,
            use_qk_l2norm_in_kernel=True,
        )
        outs[name] = (conv_out.clone(), core.clone(), conv_pool, ssm_pool)

    assert torch.equal(outs["strided"][0], outs["contig"][0]), "conv out"
    assert torch.equal(outs["strided"][1], outs["contig"][1]), "core out"
    assert torch.equal(outs["strided"][2], outs["contig"][2]), "conv state"
    assert torch.equal(outs["strided"][3], outs["contig"][3]), "ssm state"


def test_gating_kernel_strided_direct():
    """fused_sigmoid_gating also directly accepts a strided mixed_qkv
    (decode paths that skip conv rewrites)."""
    from vllm.third_party.flash_linear_attention.ops import (
        fused_sigmoid_gating_delta_rule_update,
    )

    torch.manual_seed(6)
    dev = torch.device("cuda")
    batch, ntok = 8, 8 * T
    DIM = QKV_N
    WT = 8
    NB = batch * WT + 16

    wide = torch.randn(ntok, QKVZ_N + BA_N, dtype=torch.bfloat16, device=dev)
    x_strided = wide[:, :DIM]
    x_contig = x_strided.contiguous()

    a = torch.rand(ntok, HV, dtype=torch.bfloat16, device=dev)
    b = torch.rand(ntok, HV, dtype=torch.bfloat16, device=dev)
    A_log = torch.rand(HV, dtype=torch.float32, device=dev)
    dtb = torch.rand(HV, dtype=torch.float32, device=dev)
    cu = torch.arange(0, ntok + 1, T, dtype=torch.int32, device=dev)
    acc = torch.full((batch,), T, dtype=torch.int32, device=dev)
    ids = torch.randperm(NB - 1, device=dev)[: batch * WT].int() + 1
    table = ids.view(batch, WT).contiguous()
    pa = torch.stack(
        [
            torch.zeros(batch, dtype=torch.int32, device=dev),
            torch.ones(batch, dtype=torch.int32, device=dev),
        ],
        1,
    ).contiguous()

    results = []
    for x in (x_strided, x_contig):
        torch.manual_seed(7)
        ssm_pool = torch.rand(NB, HV, V, K, dtype=torch.bfloat16, device=dev)
        core, _ = fused_sigmoid_gating_delta_rule_update(
            A_log=A_log,
            a=a,
            b=b,
            dt_bias=dtb,
            mixed_qkv=x,
            num_qk_heads=H,
            head_qk_dim=K,
            num_v_heads=HV,
            head_v_dim=V,
            initial_state=ssm_pool,
            inplace_final_state=True,
            cu_seqlens=cu,
            block_table=table,
            packed_anchors=pa,
            num_accepted_tokens=acc,
            use_qk_l2norm_in_kernel=True,
        )
        results.append((core.clone(), ssm_pool.clone()))

    assert torch.equal(results[0][0], results[1][0])
    assert torch.equal(results[0][1], results[1][1])
