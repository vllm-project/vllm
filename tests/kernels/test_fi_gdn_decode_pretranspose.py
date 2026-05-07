# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for FlashInfer ``gated_delta_rule_decode_pretranspose``.

Two groups of tests:

1. *Regression*: the CuTe DSL kernel
   ``gdn_decode_kernel_small_batch_pretranspose`` (legacy fp32 path)
   used to compute per-slot element offsets in int32, so
   ``pool_idx * ssm_state.stride(0) >= 2**31`` overflowed and the
   kernel issued an illegal global memory access. With vLLM's mamba
   KV-cache slot stride 540672 (Qwen3.5-35B-A3B fp32 SSM state),
   the threshold is ``pool_idx >= ceil(2**31 / 540672) == 3973``.
   Upstream fix: https://github.com/flashinfer-ai/flashinfer/pull/3230

2. *Parity*: the FI pretranspose decode kernel must produce the same
   outputs and SSM-state updates as the Triton recurrent reference
   ``fused_recurrent_gated_delta_rule`` with explicit gating.
"""

from __future__ import annotations

import pytest
import torch

flashinfer = pytest.importorskip("flashinfer.gdn_decode")
gated_delta_rule_decode_pretranspose = flashinfer.gated_delta_rule_decode_pretranspose


# GDN dimensions for Qwen/Qwen3.5-35B-A3B (the configuration we observed
# crashing during ``vllm bench serve``). Other Qwen3.5 variants set
# different head counts (e.g., 397B-A17B-NVFP4 uses HV=64).
B = 4  # batch size; the bug triggers as long as B >= 1, so a small B is enough
H = 16  # linear_num_key_heads
HV = 32  # linear_num_value_heads
K = 128  # linear_key_head_dim
V = 128  # linear_value_head_dim

# vLLM mamba KV-cache padded slot stride for the (HV, V, K) SSM state on
# Qwen/Qwen3.5-35B-A3B (TP=1, fp32 SSM state, single B200).
# Per-slot data is HV*V*K = 524288 fp32 elements; the pool packs an extra
# row's worth of HV*V*K + extra to colocate conv-state, giving 540672.
SLOT_STRIDE = 540672  # fp32 elements

INT32_MAX = 2**31 - 1
# Smallest pool_idx that overflows int32 element-offset arithmetic.
OVERFLOW_IDX = (INT32_MAX + 1 + SLOT_STRIDE - 1) // SLOT_STRIDE  # ceil-div


def _make_state_with_padded_pool_stride(
    pool_size: int, device: torch.device
) -> torch.Tensor:
    """Allocate ssm_state with vLLM's padded slot stride layout."""
    inner_per_slot = HV * V * K  # 524288 elements
    pad = SLOT_STRIDE - inner_per_slot  # 16384 elements per slot
    assert pad >= 0 and pad % (V * K) == 0
    pad_hv_rows = pad // (V * K)
    big = torch.empty(
        (pool_size, HV + pad_hv_rows, V, K),
        dtype=torch.float32,
        device=device,
    )
    big.zero_()
    state = big[:, :HV, :, :]
    assert state.shape == (pool_size, HV, V, K)
    assert state.stride() == (SLOT_STRIDE, V * K, K, 1), state.stride()
    # Pin underlying allocation for the lifetime of the view.
    state._owner = big  # type: ignore[attr-defined]
    return state


def _call_fi(ssm_state: torch.Tensor, pool_idx: int, device: torch.device) -> None:
    """Single FI ``gated_delta_rule_decode_pretranspose`` call.

    Mirrors the wrapper in ``gdn_linear_attn.py`` exactly: q/k/v/a/b are
    constructed via the same view-only transpose / unsqueeze ops.
    """
    q1bhk = torch.zeros((1, B, H, K), dtype=torch.bfloat16, device=device)
    k1bhk = torch.zeros((1, B, H, K), dtype=torch.bfloat16, device=device)
    v1bhv = torch.zeros((1, B, HV, V), dtype=torch.bfloat16, device=device)
    a_bhv = torch.zeros((B, HV), dtype=torch.bfloat16, device=device)
    b_bhv = torch.zeros((B, HV), dtype=torch.bfloat16, device=device)

    q = q1bhk.transpose(0, 1)  # (B, 1, H, K) non-contig view
    k = k1bhk.transpose(0, 1)
    v = v1bhv.transpose(0, 1)
    a = a_bhv.unsqueeze(1)
    b = b_bhv.unsqueeze(1)

    A_log = torch.zeros(HV, dtype=torch.float32, device=device)
    dt_bias = torch.zeros(HV, dtype=torch.bfloat16, device=device)

    state_indices = torch.full((B,), pool_idx, dtype=torch.int32, device=device)
    out = torch.zeros((B, 1, HV, V), dtype=torch.bfloat16, device=device)

    gated_delta_rule_decode_pretranspose(
        q=q,
        k=k,
        v=v,
        state=None,
        A_log=A_log,
        a=a,
        dt_bias=dt_bias,
        b=b,
        scale=K**-0.5,
        output=out,
        use_qk_l2norm=True,
        initial_state=ssm_state,
        initial_state_indices=state_indices,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.skipif(
    torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] < 9,
    reason="FlashInfer GDN decode requires SM90+",
)
def test_fi_gdn_decode_pretranspose_low_pool_idx_ok():
    """Sanity: the kernel works for pool indices below the int32 threshold.

    This is the case that vLLM hits on small workloads (``--max-concurrency``
    well under the threshold).
    """
    device = torch.device("cuda")
    pool_size = 8192
    state = _make_state_with_padded_pool_stride(pool_size, device)

    # All indices well below the OVERFLOW threshold -- should just work.
    safe_idx = OVERFLOW_IDX // 4
    assert safe_idx * SLOT_STRIDE < INT32_MAX
    _call_fi(ssm_state=state, pool_idx=safe_idx, device=device)
    torch.accelerator.synchronize()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.skipif(
    torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] < 9,
    reason="FlashInfer GDN decode requires SM90+",
)
def test_fi_gdn_decode_pretranspose_high_pool_idx_no_overflow():
    """Regression test for the int32 element-offset overflow.

    Uses ``pool_idx == OVERFLOW_IDX`` (smallest index whose element offset
    exceeds 2**31). On unpatched flashinfer this triggered an illegal
    global memory access; the upstream fix
    (https://github.com/flashinfer-ai/flashinfer/pull/3230) widens the
    offset arithmetic to int64. The call must complete cleanly.
    """
    device = torch.device("cuda")
    # Pool must be large enough to physically contain the overflowing
    # index (the alloc itself is fine; only the kernel's internal
    # offset arithmetic used to overflow).
    pool_size = OVERFLOW_IDX + 32
    state = _make_state_with_padded_pool_stride(pool_size, device)

    assert pool_size > OVERFLOW_IDX, "pool too small for chosen index"
    assert OVERFLOW_IDX * SLOT_STRIDE > INT32_MAX, "expected to overflow int32"
    assert (OVERFLOW_IDX - 1) * SLOT_STRIDE <= INT32_MAX, "pre-overflow index"

    _call_fi(ssm_state=state, pool_idx=OVERFLOW_IDX, device=device)
    # Synchronize to surface any asynchronous CUDA error from the kernel.
    torch.accelerator.synchronize()


# ---------------------------------------------------------------------------
# Parity test: FlashInfer pretranspose decode vs. Triton recurrent reference.
# Mirrors tests/kernels/test_fused_recurrent_packed_decode.py and
# tests/kernels/test_fused_sigmoid_gating_delta_rule.py: same inputs through
# both kernels, then assert close on outputs and final SSM state.
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.skipif(
    torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] < 9,
    reason="FlashInfer GDN decode requires SM90+",
)
@pytest.mark.parametrize(
    "state_dtype",
    [
        # fp32: exercises the legacy CuTe DSL pretranspose kernel.
        torch.float32,
        # bf16 with K=V=128: exercises the bf16 fast path.
        torch.bfloat16,
    ],
)
@pytest.mark.parametrize("qkv_dtype", [torch.bfloat16])
@pytest.mark.parametrize("num_reqs", [1, 4, 32])
def test_fi_gdn_decode_matches_triton_reference(
    state_dtype: torch.dtype,
    qkv_dtype: torch.dtype,
    num_reqs: int,
) -> None:
    """FlashInfer ``gated_delta_rule_decode_pretranspose`` must produce
    the same outputs and SSM-state updates as the Triton reference
    ``fused_recurrent_gated_delta_rule`` with explicit gating.
    """
    from vllm.model_executor.layers.fla.ops import (
        fused_recurrent_gated_delta_rule,
    )

    torch.manual_seed(0)
    device = torch.device("cuda")
    pool_size = num_reqs * 2

    # Inputs (one decode token per request).
    q = torch.randn((1, num_reqs, H, K), dtype=qkv_dtype, device=device) * 0.1
    k = torch.randn((1, num_reqs, H, K), dtype=qkv_dtype, device=device) * 0.1
    v = torch.randn((1, num_reqs, HV, V), dtype=qkv_dtype, device=device) * 0.1
    a = torch.randn((num_reqs, HV), dtype=qkv_dtype, device=device) * 0.1
    b = torch.randn((num_reqs, HV), dtype=qkv_dtype, device=device) * 0.1
    A_log = torch.randn(HV, dtype=torch.float32, device=device) * 0.1
    dt_bias = torch.randn(HV, dtype=torch.float32, device=device) * 0.1

    # Initial SSM state: random, never zero, in FI's [pool, HV, V, K] layout.
    initial_state = (
        torch.randn((pool_size, HV, V, K), dtype=state_dtype, device=device) * 0.05
    )

    # Per-request pool indices (1..num_reqs; reserve slot 0 unused).
    state_indices = torch.arange(1, num_reqs + 1, dtype=torch.int32, device=device)
    cu_seqlens = torch.arange(0, num_reqs + 1, dtype=torch.int32, device=device)

    # ----- Reference: explicit gating + Triton recurrent kernel -----
    state_ref = initial_state.clone()
    x = a.float() + dt_bias.float()
    softplus_x = torch.where(
        x <= 20.0, torch.log1p(torch.exp(torch.clamp(x, max=20.0))), x
    )
    g_ref = (-torch.exp(A_log.float()) * softplus_x).unsqueeze(0)
    beta_ref = torch.sigmoid(b.float()).to(qkv_dtype).unsqueeze(0)
    out_ref, state_ref = fused_recurrent_gated_delta_rule(
        q=q,
        k=k,
        v=v,
        g=g_ref,
        beta=beta_ref,
        scale=K**-0.5,
        initial_state=state_ref,
        inplace_final_state=True,
        cu_seqlens=cu_seqlens,
        ssm_state_indices=state_indices,
        use_qk_l2norm_in_kernel=True,
    )

    # ----- FI: copy-free unpack to [B, 1, ...], no explicit gating -----
    state_fi = initial_state.clone()
    out_fi = torch.empty((num_reqs, 1, HV, V), dtype=qkv_dtype, device=device)
    gated_delta_rule_decode_pretranspose(
        q=q.transpose(0, 1),
        k=k.transpose(0, 1),
        v=v.transpose(0, 1),
        state=None,
        A_log=A_log,
        a=a.unsqueeze(1),
        dt_bias=dt_bias,
        b=b.unsqueeze(1),
        scale=K**-0.5,
        output=out_fi,
        use_qk_l2norm=True,
        initial_state=state_fi,
        initial_state_indices=state_indices,
    )

    # Compare. Triton recurrent and FI CuTe kernels accumulate in different
    # orders so we use the same tolerance as test_fused_recurrent_packed_decode.
    if state_dtype == torch.float32:
        atol, rtol = 1e-2, 1e-2
    else:  # bf16 fast path is intrinsically lower precision
        atol, rtol = 5e-2, 5e-2
    out_fi_ref = out_ref.transpose(0, 1)  # ref is (1, B, HV, V) -> (B, 1, HV, V)
    torch.testing.assert_close(out_fi, out_fi_ref, atol=atol, rtol=rtol)
    torch.testing.assert_close(state_fi, state_ref, atol=atol, rtol=rtol)
