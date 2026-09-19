# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Correctness of the RDNA3.5 gated delta net kernel.

The kernel under test is `csrc/rocm/rdna35_gdn_chunked.cu`.
"""

import contextlib
import math

import pytest
import torch
import torch.nn.functional as F

from vllm.platforms import current_platform

if not current_platform.is_rocm():
    pytest.skip(
        reason="gdn_chunked is a ROCm kernel.",
        allow_module_level=True,
    )

from vllm.platforms.rocm import on_gfx115x  # noqa: E402

if not on_gfx115x():
    pytest.skip(
        reason="gdn_chunked is RDNA3.5 only.",
        allow_module_level=True,
    )

import vllm.envs as envs  # noqa: E402
from vllm.third_party.flash_linear_attention.ops import (  # noqa: E402
    chunk_gated_delta_rule,
)
from vllm.third_party.flash_linear_attention.ops import (  # noqa: E402
    rocm_rdna35_gdn_chunked as rocm_gdn,
)

HEAD_DIM = 128


def test_op_is_registered():
    """The op must be present on gfx115x, not merely skipped.

    Without this, a build that omitted the kernel would fall back to Triton and
    every test below would still pass.
    """
    assert hasattr(torch.ops, "_rocm_C")
    assert hasattr(torch.ops._rocm_C, "gdn_chunked")


@contextlib.contextmanager
def _hip_kernel_disabled():
    """Force ``chunk_gated_delta_rule`` onto the Triton kernels."""
    saved_hip = envs.VLLM_GDN_HIP
    envs.VLLM_GDN_HIP = False
    rocm_gdn._available.cache_clear()
    # If this ever stops taking effect the reference below becomes the kernel
    # under test, and every comparison passes by construction.
    assert not rocm_gdn._available()
    try:
        yield
    finally:
        envs.VLLM_GDN_HIP = saved_hip
        rocm_gdn._available.cache_clear()


def _make_inputs(
    seq_lens, num_k_heads, gqa_ratio, state_dtype=torch.float32, dtype=torch.bfloat16
):
    num_v_heads = num_k_heads * gqa_ratio
    num_seqs = len(seq_lens)
    cu_seqlens = torch.zeros(num_seqs + 1, device="cuda", dtype=torch.int32)
    cu_seqlens[1:] = torch.tensor(seq_lens, device="cuda", dtype=torch.int32).cumsum(0)
    total = int(cu_seqlens[-1].item())

    # q and k reach the kernel l2-normalised, and the conditioning of (I + A)
    # depends on it.
    q = F.normalize(
        torch.randn(1, total, num_k_heads, HEAD_DIM, device="cuda", dtype=dtype),
        p=2,
        dim=-1,
    )
    k = F.normalize(torch.randn_like(q), p=2, dim=-1)
    v = torch.randn(1, total, num_v_heads, HEAD_DIM, device="cuda", dtype=dtype)
    a = torch.randn(1, total, num_v_heads, device="cuda", dtype=dtype)
    b = torch.randn_like(a)

    # Upstream FLA GatedDeltaNet synthetic initialisation.
    A = torch.empty(num_v_heads, device="cuda", dtype=torch.float32).uniform_(0, 16)
    A_log = torch.log(A)
    dt = torch.exp(
        torch.rand(num_v_heads, device="cuda", dtype=torch.float32)
        * (math.log(0.1) - math.log(0.001))
        + math.log(0.001)
    )
    dt = torch.clamp(dt, min=1e-4)
    dt_bias = dt + torch.log(-torch.expm1(-dt))
    g = -A_log.exp().view(1, 1, num_v_heads) * F.softplus(
        a.float() + dt_bias.view(1, 1, num_v_heads)
    )
    beta = torch.sigmoid(b.float())
    initial_state = (
        torch.randn(
            num_seqs,
            num_v_heads,
            HEAD_DIM,
            HEAD_DIM,
            device="cuda",
            dtype=state_dtype,
        )
        * 0.05
    )
    return q, k, v, g, beta, initial_state, cu_seqlens


def _rel_rms(got: torch.Tensor, ref: torch.Tensor) -> float:
    """Relative RMS error, scaled by the tensor as a whole."""
    got32, ref32 = got.float(), ref.float()
    denom = max(ref32.pow(2).mean().sqrt().item(), 1e-9)
    return ((got32 - ref32).pow(2).mean().sqrt() / denom).item()


def _run_both(q, k, v, g, beta, initial_state, cu_seqlens, output_final_state=True):
    common = dict(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        output_final_state=output_final_state,
        cu_seqlens=cu_seqlens,
        use_qk_l2norm_in_kernel=False,
    )
    state = None if initial_state is None else initial_state.clone()
    with _hip_kernel_disabled():
        ref = chunk_gated_delta_rule(initial_state=state, **common)
    state = None if initial_state is None else initial_state.clone()
    got = chunk_gated_delta_rule(initial_state=state, **common)
    return got, ref


@pytest.mark.parametrize("gqa_ratio", [1, 2, 3, 4])
@pytest.mark.parametrize(
    "seq_lens",
    [
        [512],
        [32, 33, 64, 1],  # pins the 32-token chunk boundary
        [1, 300, 64, 65, 200],
        [137, 1, 941, 512],
        # Plain decodes are reclassified as prefill when speculative decoding
        # is active, arriving as a batch of single-token sequences.
        [1] * 8,
    ],
)
@torch.inference_mode()
def test_matches_triton_chain(gqa_ratio, seq_lens):
    inputs = _make_inputs(seq_lens, num_k_heads=8, gqa_ratio=gqa_ratio)
    (o, state), (ref_o, ref_state) = _run_both(*inputs)

    # Both paths are bf16 end to end, so they agree only to bf16 precision.
    assert _rel_rms(o, ref_o) < 2e-2
    assert _rel_rms(state, ref_state) < 2e-2


@torch.inference_mode()
def test_none_initial_state_matches_zeros():
    """``initial_state=None`` must behave like an explicit zero state."""
    q, k, v, g, beta, initial_state, cu_seqlens = _make_inputs(
        [300, 64], num_k_heads=8, gqa_ratio=2
    )
    (o_none, state_none), _ = _run_both(q, k, v, g, beta, None, cu_seqlens)
    (o_zero, state_zero), _ = _run_both(
        q, k, v, g, beta, torch.zeros_like(initial_state), cu_seqlens
    )
    torch.testing.assert_close(o_none, o_zero)
    torch.testing.assert_close(state_none, state_zero)


@torch.inference_mode()
def test_output_final_state_false_returns_none():
    """Suppressing the final state must not change the output."""
    q, k, v, g, beta, initial_state, cu_seqlens = _make_inputs(
        [300, 64], num_k_heads=8, gqa_ratio=2
    )
    (o_with, _), _ = _run_both(q, k, v, g, beta, initial_state, cu_seqlens)
    (o_without, state_without), _ = _run_both(
        q, k, v, g, beta, initial_state, cu_seqlens, output_final_state=False
    )
    assert state_without is None
    torch.testing.assert_close(o_with, o_without)


@pytest.mark.parametrize("split", [32, 33])
@torch.inference_mode()
def test_continuation_split_matches_monolithic(split):
    """Continuations preserve outputs across aligned and unaligned splits."""
    total = 97
    q, k, v, g, beta, initial_state, cu_seqlens = _make_inputs(
        [total], num_k_heads=8, gqa_ratio=2
    )
    common = dict(use_qk_l2norm_in_kernel=False, output_final_state=True)

    o_mono, state_mono = chunk_gated_delta_rule(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        initial_state=initial_state.clone(),
        cu_seqlens=cu_seqlens,
        **common,
    )

    state_before = initial_state.clone()
    prefix_cu = torch.tensor([0, split], device="cuda", dtype=torch.int32)
    o1, state1 = chunk_gated_delta_rule(
        q=q[:, :split],
        k=k[:, :split],
        v=v[:, :split],
        g=g[:, :split],
        beta=beta[:, :split],
        initial_state=initial_state,
        cu_seqlens=prefix_cu,
        **common,
    )
    assert torch.equal(initial_state, state_before), "initial state must not mutate"

    suffix_cu = torch.tensor([0, total - split], device="cuda", dtype=torch.int32)
    o2, state2 = chunk_gated_delta_rule(
        q=q[:, split:],
        k=k[:, split:],
        v=v[:, split:],
        g=g[:, split:],
        beta=beta[:, split:],
        initial_state=state1,
        cu_seqlens=suffix_cu,
        **common,
    )

    o_split = torch.cat([o1, o2], dim=1)
    assert _rel_rms(o_split, o_mono) < 1e-3
    assert _rel_rms(state2, state_mono) < 1e-3


@pytest.mark.parametrize(
    "gate,beta_value", [(0.0, 1.0), (-0.0003, 0.9997), (-4.0, 0.0)]
)
@torch.inference_mode()
def test_adversarial_gate_beta_matches_fp64_reference(gate, beta_value):
    """Correlated keys and gate extremes agree with a token-wise reference."""
    torch.manual_seed(0)
    total, num_k_heads, gqa_ratio = 65, 2, 3
    num_v_heads = num_k_heads * gqa_ratio
    cu_seqlens = torch.tensor([0, total], device="cuda", dtype=torch.int32)
    dtype = torch.bfloat16

    base = F.normalize(
        torch.randn(1, 1, num_k_heads, HEAD_DIM, device="cuda"), p=2, dim=-1
    )
    noise = 0.02 * torch.randn(1, total, num_k_heads, HEAD_DIM, device="cuda")
    k = F.normalize((base + noise).to(dtype), p=2, dim=-1)
    q = F.normalize(
        torch.randn(1, total, num_k_heads, HEAD_DIM, device="cuda", dtype=dtype),
        p=2,
        dim=-1,
    )
    v = torch.randn(1, total, num_v_heads, HEAD_DIM, device="cuda", dtype=dtype)
    beta = torch.full((1, total, num_v_heads), beta_value, device="cuda")
    g = torch.full((1, total, num_v_heads), gate, device="cuda")
    initial_state = (
        torch.randn(1, num_v_heads, HEAD_DIM, HEAD_DIM, device="cuda") * 0.05
    )
    scale = HEAD_DIM**-0.5

    from vllm.third_party.flash_linear_attention.ops.rocm_rdna35_gdn_chunked import (
        chunk_gdn_hip_fwd,
    )

    o, final_state = chunk_gdn_hip_fwd(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        scale=scale,
        initial_state=initial_state,
        cu_seqlens=cu_seqlens,
    )
    assert torch.isfinite(o).all()
    assert torch.isfinite(final_state).all()

    ref_o, ref_state = _reference_delta_rule_fp64(
        q, k, v, g, beta, initial_state, cu_seqlens, scale
    )
    torch.testing.assert_close(o.double(), ref_o, rtol=1e-2, atol=1e-5)
    torch.testing.assert_close(final_state.double(), ref_state, rtol=1e-3, atol=1e-5)


def _reference_delta_rule_fp64(q, k, v, g, beta, initial_state, cu_seqlens, scale):
    """Independent per-token fp64 recurrence; deliberately not the kernel path."""
    num_seqs = len(cu_seqlens) - 1
    num_v_heads, head_dim = v.shape[-2], v.shape[-1]
    num_k_heads = q.shape[-2]
    gqa = num_v_heads // num_k_heads
    q64, k64, v64, g64, beta64 = (t.double() for t in (q, k, v, g, beta))
    o = torch.zeros_like(v, dtype=torch.float64)
    final_state = torch.zeros(
        num_seqs, num_v_heads, head_dim, head_dim, dtype=torch.float64, device=v.device
    )
    for s in range(num_seqs):
        start, end = int(cu_seqlens[s]), int(cu_seqlens[s + 1])
        for hv in range(num_v_heads):
            h = (
                initial_state[s, hv].double()
                if initial_state is not None
                else v.new_zeros(head_dim, head_dim, dtype=torch.float64)
            )
            hk = hv // gqa
            for t in range(start, end):
                qt = q64[0, t, hk] * scale
                kt = k64[0, t, hk]
                vt = v64[0, t, hv]
                h = h * g64[0, t, hv].exp()
                v_tilde = (vt - h @ kt) * beta64[0, t, hv]
                h = h + torch.outer(v_tilde, kt)
                o[0, t, hv] = h @ qt
            final_state[s, hv] = h
    return o, final_state


@pytest.mark.parametrize("gqa_ratio", [1, 3])
@pytest.mark.parametrize("seq_lens", [[512], [32, 33, 64, 1], [137, 1, 941]])
@torch.inference_mode()
def test_matches_triton_chain_fp16(gqa_ratio, seq_lens):
    """fp16 inputs take the same kernel; AWQ checkpoints ship fp16.

    gqa_ratio 3 covers the 48-value-head / 16-key-head shape of
    Qwen3.6-27B-AWQ-INT4, which is what put fp16 on this path.
    """
    inputs = _make_inputs(
        seq_lens, num_k_heads=8, gqa_ratio=gqa_ratio, dtype=torch.float16
    )
    (o, state), (ref_o, ref_state) = _run_both(*inputs)

    assert o.dtype == torch.float16
    # fp16 carries 10 mantissa bits to bf16's 7, so the two paths agree more
    # tightly here than the bf16 case above.
    assert _rel_rms(o, ref_o) < 2e-2
    assert _rel_rms(state, ref_state) < 2e-2


@pytest.mark.parametrize("state_dtype", [torch.float32, torch.bfloat16])
@torch.inference_mode()
def test_initial_state_dtypes(state_dtype):
    """The wrapper casts the incoming state to fp32."""
    inputs = _make_inputs(
        [300, 64], num_k_heads=8, gqa_ratio=2, state_dtype=state_dtype
    )
    (o, state), (ref_o, ref_state) = _run_both(*inputs)
    assert state.dtype == torch.float32
    assert state.shape == ref_state.shape
    assert _rel_rms(o, ref_o) < 2e-2
    assert _rel_rms(state, ref_state) < 2e-2


@torch.inference_mode()
def test_core_attn_out_aliasing():
    """``out`` is a view into a caller-owned buffer; it must not overrun it."""
    from vllm.third_party.flash_linear_attention.ops.rocm_rdna35_gdn_chunked import (
        chunk_gdn_hip_fwd,
    )

    q, k, v, g, beta, initial_state, cu_seqlens = _make_inputs(
        [300], num_k_heads=8, gqa_ratio=2
    )
    scale = HEAD_DIM**-0.5
    ref_o, _ = chunk_gdn_hip_fwd(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        scale=scale,
        initial_state=initial_state,
        cu_seqlens=cu_seqlens,
    )

    slack = 1024
    buf = torch.full((v.numel() + slack,), float("nan"), device="cuda", dtype=v.dtype)
    o, _ = chunk_gdn_hip_fwd(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        scale=scale,
        initial_state=initial_state,
        cu_seqlens=cu_seqlens,
        core_attn_out=buf,
    )
    assert torch.equal(o.reshape(-1), buf[: v.numel()])
    assert torch.isnan(buf[v.numel() :]).all(), "kernel wrote past the buffer"
    torch.testing.assert_close(o, ref_o)


@torch.inference_mode()
def test_opcheck():
    """Schema, aliasing annotations and fake implementation agree."""
    q, k, v, g, beta, initial_state, cu_seqlens = _make_inputs(
        [300], num_k_heads=8, gqa_ratio=2
    )
    out = torch.empty_like(v).squeeze(0)
    final_state = q.new_empty(1, v.shape[-2], HEAD_DIM, HEAD_DIM, dtype=torch.float32)
    torch.library.opcheck(
        torch.ops._rocm_C.gdn_chunked,
        (
            q.squeeze(0),
            k.squeeze(0),
            v.squeeze(0),
            g.squeeze(0).float(),
            beta.squeeze(0).float(),
            initial_state.float().contiguous(),
            cu_seqlens.to(torch.int32),
            out,
            final_state,
            HEAD_DIM**-0.5,
        ),
    )


@torch.inference_mode()
def test_declines_unsupported():
    """The gate must decline every case the kernel asserts on."""
    from vllm.third_party.flash_linear_attention.ops.rocm_rdna35_gdn_chunked import (
        is_hip_gdn_supported,
    )

    q, _, v, _, _, _, cu_seqlens = _make_inputs([64], num_k_heads=8, gqa_ratio=2)

    assert is_hip_gdn_supported(q, v, cu_seqlens)
    assert is_hip_gdn_supported(q.half(), v.half(), cu_seqlens)
    assert not is_hip_gdn_supported(q, v, None)
    # The kernel reads one element type, so a mixed pair has to fall back.
    assert not is_hip_gdn_supported(q.half(), v, cu_seqlens)
    assert not is_hip_gdn_supported(q.float(), v.float(), cu_seqlens)
    assert not is_hip_gdn_supported(q[..., :64], v[..., :64], cu_seqlens)
    # value heads not a multiple of key heads
    assert not is_hip_gdn_supported(q[:, :, :3], v[:, :, :5], cu_seqlens)
