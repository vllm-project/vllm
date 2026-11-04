# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.platforms import current_platform
from vllm.third_party.flash_linear_attention.ops import (
    fused_recurrent_gated_delta_rule,
    fused_recurrent_gated_delta_rule_packed_decode,
)
from vllm.third_party.flash_linear_attention.ops.fused_sigmoid_gating import (
    fused_sigmoid_gating_delta_rule_update,
)

DEVICE = current_platform.device_type

pytestmark = pytest.mark.skipif(
    not (current_platform.is_cuda_alike() or current_platform.is_xpu()),
    reason="Gated delta rule Triton kernels require a CUDA-alike or XPU device.",
)


@pytest.mark.parametrize("strided_mixed_qkv", [False, True])
@pytest.mark.parametrize(
    "batch,num_key_heads,num_value_heads,dtype,state_dtype",
    [
        (32, 4, 8, torch.float16, torch.float16),
        (32, 4, 8, torch.bfloat16, torch.bfloat16),
        (32, 4, 8, torch.float32, torch.float32),
        (1, 16, 64, torch.bfloat16, torch.float32),
        (9, 4, 12, torch.bfloat16, torch.float32),
        (1, 4, 12, torch.bfloat16, torch.bfloat16),
    ],
)
def test_fused_recurrent_packed_decode_matches_reference(
    dtype: torch.dtype,
    strided_mixed_qkv: bool,
    batch: int,
    num_key_heads: int,
    num_value_heads: int,
    state_dtype: torch.dtype,
):
    generator = torch.Generator(device=DEVICE).manual_seed(0)

    # Small but representative GDN config (Qwen3Next defaults are K=128, V=128).
    B = batch
    H = num_key_heads
    HV = num_value_heads
    K = 128
    V = 128
    qkv_dim = 2 * (H * K) + (HV * V)

    device = torch.device(DEVICE)

    if strided_mixed_qkv:
        # Simulate a packed view into a larger projection buffer:
        # mixed_qkv.stride(0) > mixed_qkv.shape[1]
        proj = torch.randn(
            (B, qkv_dim + 64), device=device, dtype=dtype, generator=generator
        )
        mixed_qkv = proj[:, :qkv_dim]
    else:
        mixed_qkv = torch.randn(
            (B, qkv_dim), device=device, dtype=dtype, generator=generator
        )

    a = torch.randn((B, HV), device=device, dtype=dtype, generator=generator)
    b = torch.randn((B, HV), device=device, dtype=dtype, generator=generator)
    A_log = torch.randn((HV,), device=device, dtype=dtype, generator=generator)
    dt_bias = torch.randn((HV,), device=device, dtype=dtype, generator=generator)

    # Continuous batching indices (include PAD_SLOT_ID=-1 cases). Index 0 is
    # reserved as NULL_BLOCK_ID (CUDA graph padding), so valid slots start at 1.
    ssm_state_indices = torch.arange(1, B + 1, device=device, dtype=torch.int32)
    if B > 3:
        ssm_state_indices[-3:] = -1
        ssm_state_indices[-1] = 0

    state0 = torch.randn(
        (B + 1, HV, V, K),
        device=device,
        dtype=state_dtype,
        generator=generator,
    )
    state_ref = state0.clone()
    state_packed = state0.clone()

    out_packed = torch.empty((B, 1, HV, V), device=device, dtype=dtype)

    # Reference path: materialize contiguous Q/K/V + explicit gating.
    q, k, v = torch.split(mixed_qkv, [H * K, H * K, HV * V], dim=-1)
    q = q.view(B, H, K).unsqueeze(1).contiguous()
    k = k.view(B, H, K).unsqueeze(1).contiguous()
    v = v.view(B, HV, V).unsqueeze(1).contiguous()

    x = a.float() + dt_bias.float()
    softplus_x = torch.where(
        x <= 20.0, torch.log1p(torch.exp(torch.clamp(x, max=20.0))), x
    )
    g = (-torch.exp(A_log.float()) * softplus_x).unsqueeze(1)
    beta = torch.sigmoid(b.float()).unsqueeze(1)

    out_ref, state_ref = fused_recurrent_gated_delta_rule(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        scale=K**-0.5,
        initial_state=state_ref,
        inplace_final_state=True,
        cu_seqlens=None,
        ssm_state_indices=ssm_state_indices,
        use_qk_l2norm_in_kernel=True,
    )

    # Packed path: fused gating + recurrent directly from packed mixed_qkv.
    fused_recurrent_gated_delta_rule_packed_decode(
        mixed_qkv=mixed_qkv,
        a=a,
        b=b,
        A_log=A_log,
        dt_bias=dt_bias,
        scale=K**-0.5,
        initial_state=state_packed,
        out=out_packed,
        ssm_state_indices=ssm_state_indices,
        use_qk_l2norm_in_kernel=True,
    )

    atol = 2e-2 if dtype != torch.float32 else 1e-4
    rtol = 1e-2 if dtype != torch.float32 else 1e-4
    # Output rows for PAD_SLOT_ID entries are never written (uninitialized in
    # both paths), so compare only the valid rows.
    valid = ssm_state_indices > 0
    torch.testing.assert_close(out_packed[valid], out_ref[valid], rtol=rtol, atol=atol)
    torch.testing.assert_close(state_packed, state_ref, rtol=rtol, atol=atol)


def _spec_and_packed_step(
    *, q, k, v, a, b, A_log, dt_bias, state_spec, state_packed, indices, scale
):
    out_spec, _ = fused_sigmoid_gating_delta_rule_update(
        A_log=A_log,
        a=a,
        b=b,
        dt_bias=dt_bias,
        q=q,
        k=k,
        v=v,
        scale=scale,
        initial_state=state_spec,
        inplace_final_state=True,
        ssm_state_indices=indices,
        use_qk_l2norm_in_kernel=True,
    )
    HV = v.shape[-2]
    mixed_qkv = torch.cat((q[0, 0].flatten(), k[0, 0].flatten(), v[0, 0].flatten()))
    out_packed = torch.empty((1, 1, HV, v.shape[-1]), device=q.device, dtype=q.dtype)
    fused_recurrent_gated_delta_rule_packed_decode(
        mixed_qkv=mixed_qkv.unsqueeze(0),
        a=a,
        b=b,
        A_log=A_log,
        dt_bias=dt_bias,
        scale=scale,
        initial_state=state_packed,
        out=out_packed,
        ssm_state_indices=indices,
        use_qk_l2norm_in_kernel=True,
    )
    return out_spec, out_packed


@pytest.mark.parametrize("state_dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("num_tokens", [1, 2, 7])
def test_packed_decode_matches_serial_speculative_recurrence(
    num_tokens: int, state_dtype: torch.dtype
):
    """Packed decode must match the recurrent kernel used for verification.

    Both paths now launch the same kernel with the same tile configuration, so
    the comparison is a bitwise one on every architecture.
    """
    torch.manual_seed(0)
    device = torch.device(DEVICE)
    dtype = torch.bfloat16
    H, HV, K, V = 16, 48, 128, 128
    scale = K**-0.5
    q = torch.randn(1, num_tokens, H, K, device=device, dtype=dtype)
    k = torch.randn_like(q)
    v = torch.randn(1, num_tokens, HV, V, device=device, dtype=dtype)
    a = torch.randn(num_tokens, HV, device=device, dtype=dtype)
    b = torch.randn_like(a)
    A_log = torch.randn(HV, device=device) * 0.1
    dt_bias = torch.randn(HV, device=device) * 0.1
    initial_state = torch.randn(2, HV, V, K, device=device, dtype=state_dtype) * 0.05
    state_spec = initial_state.clone()
    state_packed = initial_state.clone()
    state_indices = torch.ones(1, device=device, dtype=torch.int32)
    out_spec = []
    out_packed = []

    for token in range(num_tokens):
        spec, packed = _spec_and_packed_step(
            q=q[:, token : token + 1],
            k=k[:, token : token + 1],
            v=v[:, token : token + 1],
            a=a[token : token + 1],
            b=b[token : token + 1],
            A_log=A_log,
            dt_bias=dt_bias,
            state_spec=state_spec,
            state_packed=state_packed,
            indices=state_indices,
            scale=scale,
        )
        out_spec.append(spec)
        out_packed.append(packed)

    torch.testing.assert_close(
        torch.cat(out_packed, dim=1), torch.cat(out_spec, dim=1), rtol=0, atol=0
    )
    torch.testing.assert_close(state_packed[1], state_spec[1], rtol=0, atol=0)


def test_packed_decode_is_batch_size_invariant():
    """A request must advance identically alone or alongside other requests.

    The tile configuration depends only on ``(K, V, state dtype)`` and every
    program owns one ``(sequence, value-head)`` tile, so the number of
    sequences in the launch must not change any arithmetic.
    """
    torch.manual_seed(0)
    device = torch.device(DEVICE)
    dtype = torch.bfloat16
    H, HV, K, V = 2, 4, 128, 128
    qkv_dim = 2 * H * K + HV * V
    scale = K**-0.5
    A_log = torch.randn(HV, device=device) * 0.1
    dt_bias = torch.randn(HV, device=device) * 0.1

    num_seqs = 8
    mixed_qkv = torch.randn(num_seqs, qkv_dim, device=device, dtype=dtype)
    a = torch.randn(num_seqs, HV, device=device, dtype=dtype)
    b = torch.randn_like(a)
    state = torch.randn(num_seqs + 1, HV, V, K, device=device, dtype=dtype)

    out_batched = torch.empty(num_seqs, 1, HV, V, device=device, dtype=dtype)
    state_batched = state.clone()
    fused_recurrent_gated_delta_rule_packed_decode(
        mixed_qkv=mixed_qkv,
        a=a,
        b=b,
        A_log=A_log,
        dt_bias=dt_bias,
        scale=scale,
        initial_state=state_batched,
        out=out_batched,
        ssm_state_indices=torch.arange(
            1, num_seqs + 1, device=device, dtype=torch.int32
        ),
        use_qk_l2norm_in_kernel=True,
    )

    out_single = torch.empty(1, 1, HV, V, device=device, dtype=dtype)
    state_single = state[:2].clone()
    fused_recurrent_gated_delta_rule_packed_decode(
        mixed_qkv=mixed_qkv[:1],
        a=a[:1],
        b=b[:1],
        A_log=A_log,
        dt_bias=dt_bias,
        scale=scale,
        initial_state=state_single,
        out=out_single,
        ssm_state_indices=torch.ones(1, device=device, dtype=torch.int32),
        use_qk_l2norm_in_kernel=True,
    )

    torch.testing.assert_close(out_single, out_batched[:1], rtol=0, atol=0)
    torch.testing.assert_close(state_single[1], state_batched[1], rtol=0, atol=0)


def test_packed_decode_matches_batched_speculative_verification():
    """One multi-token spec-verification launch must equal N packed decode
    launches, which is how the same request is advanced on spec and non-spec
    steps.

    Only fp32 state is compared: Qwen GDN's ``mamba_ssm_dtype`` is ``float32``
    in the model config, so this is the production configuration. A
    reduced-precision cache cannot match by construction, because a multi-token
    launch keeps the state in registers across tokens while per-token launches
    round-trip it.
    """
    torch.manual_seed(0)
    device = torch.device(DEVICE)
    dtype = torch.bfloat16
    num_seqs, tokens_per_seq, H, HV, K, V = 3, 4, 16, 48, 128, 128
    scale = K**-0.5
    total = num_seqs * tokens_per_seq
    q = torch.randn(1, total, H, K, device=device, dtype=dtype)
    k = torch.randn_like(q)
    v = torch.randn(1, total, HV, V, device=device, dtype=dtype)
    a = torch.randn(1, total, HV, device=device, dtype=dtype)
    b = torch.randn_like(a)
    A_log = torch.randn(HV, device=device) * 0.1
    dt_bias = torch.randn(HV, device=device) * 0.1
    slots = torch.arange(1, num_seqs + 1, device=device, dtype=torch.int32)
    # Every timestep of a sequence advances the same slot.
    indices = torch.as_strided(slots, (num_seqs, tokens_per_seq), (1, 0))
    cu_seqlens = torch.arange(
        0, total + 1, tokens_per_seq, device=device, dtype=torch.int32
    )
    num_accepted = torch.full(
        (num_seqs,), tokens_per_seq, device=device, dtype=torch.int32
    )

    state_spec = torch.randn(num_seqs + 1, HV, V, K, device=device, dtype=torch.float32)
    state_packed = state_spec.clone()
    out_spec, _ = fused_sigmoid_gating_delta_rule_update(
        A_log=A_log,
        a=a,
        b=b,
        dt_bias=dt_bias,
        q=q,
        k=k,
        v=v,
        scale=scale,
        initial_state=state_spec,
        inplace_final_state=True,
        cu_seqlens=cu_seqlens,
        ssm_state_indices=indices,
        num_accepted_tokens=num_accepted,
        use_qk_l2norm_in_kernel=True,
    )

    out_packed = []
    for step in range(tokens_per_seq):
        tokens = torch.arange(
            step, total, tokens_per_seq, device=device, dtype=torch.long
        )
        step_out = torch.empty((num_seqs, 1, HV, V), device=device, dtype=dtype)
        fused_recurrent_gated_delta_rule_packed_decode(
            mixed_qkv=torch.cat(
                (
                    q[0, tokens].reshape(num_seqs, H * K),
                    k[0, tokens].reshape(num_seqs, H * K),
                    v[0, tokens].reshape(num_seqs, HV * V),
                ),
                dim=-1,
            ),
            a=a[0, tokens],
            b=b[0, tokens],
            A_log=A_log,
            dt_bias=dt_bias,
            scale=scale,
            initial_state=state_packed,
            out=step_out,
            ssm_state_indices=slots,
            use_qk_l2norm_in_kernel=True,
        )
        out_packed.append(step_out)

    torch.testing.assert_close(
        torch.stack(out_packed, dim=1).reshape(out_spec.shape),
        out_spec,
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        state_packed[1 : num_seqs + 1], state_spec[1 : num_seqs + 1], rtol=0, atol=0
    )


def test_decode_paths_share_one_kernel_config(monkeypatch: pytest.MonkeyPatch):
    """Both decode entry points must resolve the same tile/warp config."""
    import vllm.third_party.flash_linear_attention.ops.recurrent_decode as rd

    calls = []
    original = rd._select_decode_config_cached

    def spy(*args, **kwargs):
        config = original(*args, **kwargs)
        calls.append(config)
        return config

    monkeypatch.setattr(rd, "_select_decode_config_cached", spy)

    device = torch.device(DEVICE)
    dtype = torch.bfloat16
    num_seqs, tokens_per_seq, H, HV, K, V = 4, 4, 16, 48, 128, 128
    total = num_seqs * tokens_per_seq
    state = torch.randn(num_seqs + 1, HV, V, K, device=device)
    params = torch.randn(HV, device=device)
    slots = torch.arange(1, num_seqs + 1, device=device, dtype=torch.int32)

    q = torch.randn(1, total, H, K, device=device, dtype=dtype)
    fused_sigmoid_gating_delta_rule_update(
        A_log=params,
        a=torch.randn(total, HV, device=device, dtype=dtype),
        b=torch.randn(total, HV, device=device, dtype=dtype),
        dt_bias=params,
        q=q,
        k=torch.randn_like(q),
        v=torch.randn(1, total, HV, V, device=device, dtype=dtype),
        scale=K**-0.5,
        initial_state=state,
        cu_seqlens=torch.arange(
            0, total + 1, tokens_per_seq, device=device, dtype=torch.int32
        ),
        ssm_state_indices=torch.as_strided(slots, (num_seqs, tokens_per_seq), (1, 0)),
        use_qk_l2norm_in_kernel=True,
    )
    fused_recurrent_gated_delta_rule_packed_decode(
        mixed_qkv=torch.randn(num_seqs, 2 * H * K + HV * V, device=device, dtype=dtype),
        a=torch.randn(num_seqs, HV, device=device, dtype=dtype),
        b=torch.randn(num_seqs, HV, device=device, dtype=dtype),
        A_log=params,
        dt_bias=params,
        scale=K**-0.5,
        initial_state=state,
        out=torch.empty(num_seqs, 1, HV, V, device=device, dtype=dtype),
        ssm_state_indices=slots,
        use_qk_l2norm_in_kernel=True,
    )

    assert len(calls) == 2
    assert calls[0] == calls[1], calls


def test_packed_decode_keeps_beta_in_fp32():
    device = torch.device(DEVICE)
    dtype = torch.bfloat16

    mixed_qkv = torch.ones((1, 3), device=device, dtype=dtype)
    a = torch.zeros((1, 1), device=device, dtype=dtype)
    b = torch.full((1, 1), 0.5, device=device, dtype=dtype)
    params = torch.zeros((1,), device=device, dtype=dtype)
    ssm_state_indices = torch.ones((1,), device=device, dtype=torch.int32)

    state_packed = torch.zeros((2, 1, 1, 1), device=device, dtype=torch.float32)
    out_packed = torch.empty((1, 1, 1, 1), device=device, dtype=dtype)

    fused_recurrent_gated_delta_rule_packed_decode(
        mixed_qkv=mixed_qkv,
        a=a,
        b=b,
        A_log=params,
        dt_bias=params,
        scale=1.0,
        initial_state=state_packed,
        out=out_packed,
        ssm_state_indices=ssm_state_indices,
    )

    expected_beta = torch.sigmoid(b.float()).squeeze()
    torch.testing.assert_close(
        state_packed[1, 0, 0, 0], expected_beta, rtol=1e-6, atol=1e-6
    )


def test_packed_decode_supports_large_batch_head_grid():
    B, H, HV, K, V = 1024, 8, 64, 1, 1
    device = torch.device(DEVICE)
    gates = torch.empty((B, HV), device=device)
    params = torch.empty((HV,), device=device)
    out = torch.empty((B, 1, HV, V), device=device)

    fused_recurrent_gated_delta_rule_packed_decode(
        mixed_qkv=torch.empty((B, 2 * H * K + HV * V), device=device),
        a=gates,
        b=gates,
        A_log=params,
        dt_bias=params,
        scale=1.0,
        initial_state=torch.empty((1, HV, V, K), device=device),
        out=out,
        ssm_state_indices=torch.zeros((B,), device=device, dtype=torch.int32),
    )

    assert torch.count_nonzero(out).item() == 0


@pytest.mark.parametrize("num_tokens", [1, 9])
@torch.inference_mode()
def test_packed_decode_strided_ba_bitwise(num_tokens: int) -> None:
    """Packed b/a chunk views (row stride 2*HV) must match contiguous inputs
    bitwise."""
    torch.manual_seed(0)
    dtype = torch.bfloat16
    device = torch.device(DEVICE)
    H, HV, K, V = 2, 4, 128, 128
    qkv_dim = 2 * H * K + HV * V

    qkvz = torch.randn(num_tokens, qkv_dim + HV * V, dtype=dtype, device=device)
    mixed_qkv = qkvz[:, :qkv_dim]
    ba = torch.randn(num_tokens, 2 * HV, dtype=dtype, device=device)
    b_view, a_view = ba.chunk(2, dim=-1)
    A_log = torch.randn(HV, dtype=torch.float32, device=device)
    dt_bias = torch.randn(HV, dtype=torch.float32, device=device)
    state = torch.randn(num_tokens + 1, HV, V, K, dtype=torch.float32, device=device)
    indices = torch.arange(1, num_tokens + 1, dtype=torch.int32, device=device)

    out_strided = torch.empty(num_tokens, 1, HV, V, dtype=dtype, device=device)
    state_strided = state.clone()
    fused_recurrent_gated_delta_rule_packed_decode(
        mixed_qkv=mixed_qkv,
        a=a_view,
        b=b_view,
        A_log=A_log,
        dt_bias=dt_bias,
        scale=K**-0.5,
        initial_state=state_strided,
        out=out_strided,
        ssm_state_indices=indices,
        use_qk_l2norm_in_kernel=True,
    )

    out_contig = torch.empty(num_tokens, 1, HV, V, dtype=dtype, device=device)
    state_contig = state.clone()
    fused_recurrent_gated_delta_rule_packed_decode(
        mixed_qkv=mixed_qkv.contiguous(),
        a=a_view.contiguous(),
        b=b_view.contiguous(),
        A_log=A_log,
        dt_bias=dt_bias,
        scale=K**-0.5,
        initial_state=state_contig,
        out=out_contig,
        ssm_state_indices=indices,
        use_qk_l2norm_in_kernel=True,
    )

    torch.testing.assert_close(out_strided, out_contig, rtol=0, atol=0)
    torch.testing.assert_close(state_strided, state_contig, rtol=0, atol=0)
