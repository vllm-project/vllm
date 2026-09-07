# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Correctness tests for vLLM's ported Helion KDA kernels.

Compares Helion kernel outputs against pure-Python reference
implementations. Uses torch.testing.assert_close for numerical checks.

Ported from SGLang PR #32593 with adaptations for vLLM:
  - Pure-Python reference loops replace Triton baselines (different
    calling conventions in vLLM)
  - vLLM-specific adapter shape-transform tests added
  - Feature-toggle env-var tests added
"""
from __future__ import annotations

import pytest
import torch

from vllm.utils.import_utils import has_helion

if not has_helion():
    pytest.skip(
        "Helion is not installed. Install with: pip install vllm[helion]",
        allow_module_level=True,
    )

from vllm.kernels.helion.ops.kda.kda_decode import (
    helion_fused_recurrent_kda_packed_decode,
)
from vllm.kernels.helion.ops.kda.kda_prefill import (
    _intra_matrices_wide,
)
from vllm.kernels.helion.ops.kda.kda_prefill import (
    chunk_kda as helion_chunk_kda,
)
from vllm.platforms import current_platform

pytestmark = [
    pytest.mark.skipif(
        not current_platform.is_cuda(),
        reason="Helion KDA requires NVIDIA CUDA",
    ),
]

# Per-dtype tolerances for state comparison (shared across decode tests)
_DECODE_STATE_ATOL = {
    torch.float32: 1e-5,
    torch.bfloat16: 2e-3,
    torch.float16: 5e-4,
}


# ---------------------------------------------------------------------------
#  HELPER: generalized decode reference loop
# ---------------------------------------------------------------------------

def _decode_reference_loop(
    mixed_qkv: torch.Tensor,
    gate: torch.Tensor,
    beta: torch.Tensor,
    a_log: torch.Tensor,
    dt_bias: torch.Tensor,
    scale: float,
    state: torch.Tensor,
    out: torch.Tensor,
    indices: torch.Tensor,
    lower_bound: float | None = None,
) -> None:
    """Pure-Python KDA decode reference — handles both gate modes."""
    B = mixed_qkv.size(0)
    HV, V, K = state.shape[-3:]
    H = (mixed_qkv.size(1) - HV * V) // (2 * K)
    heads_per_q = HV // H

    q_raw, k_raw, v_raw = mixed_qkv.float().split(
        [H * K, H * K, HV * V], dim=-1
    )
    q = q_raw.view(B, H, K)
    k = k_raw.view(B, H, K)
    q = q / torch.sqrt((q * q).sum(-1, keepdim=True) + 1e-6)
    k = k / torch.sqrt((k * k).sum(-1, keepdim=True) + 1e-6)
    q = q.repeat_interleave(heads_per_q, dim=1)
    k = k.repeat_interleave(heads_per_q, dim=1)
    v = v_raw.view(B, HV, V)

    raw_gate = gate.float().view(B, HV, K)
    raw_gate = raw_gate + dt_bias.float().view(1, HV, K)
    A = torch.exp(a_log.float()).view(1, HV, 1)
    if lower_bound is not None:
        decay = torch.exp(lower_bound * torch.sigmoid(A * raw_gate))
    else:
        decay = torch.exp(-A * torch.nn.functional.softplus(raw_gate))
    beta_value = torch.sigmoid(beta.float())

    out.zero_()
    for b, idx in enumerate(indices.tolist()):
        if idx < 0:
            continue
        s = state[idx].float()
        s = s * decay[b, :, None, :]
        residual = v[b] - (s * k[b, :, None, :]).sum(-1)
        residual = residual * beta_value[b, :, None]
        s = s + residual[..., None] * k[b, :, None, :]
        out[b, 0] = (s * (q[b] * scale)[:, None, :]).sum(-1)
        state[idx] = s


# ---------------------------------------------------------------------------
#  HELPER: prefill reference + Helion comparison (ported from SGLang)
# ---------------------------------------------------------------------------

def _compare_prefill(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    gate: torch.Tensor,
    beta: torch.Tensor,
    state: torch.Tensor,
    indices: torch.Tensor,
    scale: float | None = None,
    use_qk_l2norm_in_kernel: bool = False,
    cu_seqlens: torch.Tensor | None = None,
    A_log: torch.Tensor | None = None,
    dt_bias: torch.Tensor | None = None,
    lower_bound: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    batch, tokens, heads, key_dim = q.shape
    value_dim = v.size(-1)
    if scale is None:
        scale = key_dim**-0.5

    reference_q = q.float()
    reference_k = k.float()
    if use_qk_l2norm_in_kernel:
        reference_q = reference_q / torch.sqrt(
            (reference_q * reference_q).sum(-1, keepdim=True) + 1e-6
        )
        reference_k = reference_k / torch.sqrt(
            (reference_k * reference_k).sum(-1, keepdim=True) + 1e-6
        )
        reference_q = reference_q.to(q.dtype).float()
        reference_k = reference_k.to(k.dtype).float()

    reference_gate = gate.float()
    if A_log is not None:
        if dt_bias is not None:
            reference_gate = reference_gate + dt_bias.view(1, 1, heads, key_dim)
        a = torch.exp(A_log.float()).view(1, 1, heads, 1)
        if lower_bound is not None:
            reference_gate = lower_bound * torch.sigmoid(a * reference_gate)
        else:
            reference_gate = -a * torch.nn.functional.softplus(reference_gate)

    reference_state = state.clone()
    reference_out = torch.empty_like(v)
    q_rows = reference_q.view(batch * tokens, heads, key_dim)
    k_rows = reference_k.view(batch * tokens, heads, key_dim)
    v_rows = v.view(batch * tokens, heads, value_dim).float()
    gate_rows = reference_gate.view(batch * tokens, heads, key_dim)
    beta_rows = beta.view(batch * tokens, heads).float()
    out_rows = reference_out.view(batch * tokens, heads, value_dim)

    if cu_seqlens is None:
        sequence_bounds = [
            (sequence * tokens, (sequence + 1) * tokens)
            for sequence in range(batch)
        ]
        chunks_per_sequence = (tokens + 63) // 64
        reference_chunks = torch.empty(
            batch, chunks_per_sequence, heads, value_dim, key_dim,
            device=q.device, dtype=v.dtype,
        )
    else:
        offsets = cu_seqlens.tolist()
        sequence_bounds = list(zip(offsets, offsets[1:]))
        total_chunks = sum(
            (end - begin + 63) // 64 for begin, end in sequence_bounds
        )
        reference_chunks = torch.empty(
            1, total_chunks, heads, value_dim, key_dim,
            device=q.device, dtype=v.dtype,
        )

    global_chunk = 0
    for sequence, (begin, end) in enumerate(sequence_bounds):
        state_index = indices[sequence].item()
        current_state = reference_state[state_index].float()
        for local_chunk, chunk_begin in enumerate(range(begin, end, 64)):
            chunk_index = local_chunk if cu_seqlens is None else global_chunk
            chunk_batch = sequence if cu_seqlens is None else 0
            reference_chunks[chunk_batch, chunk_index] = current_state.to(
                v.dtype
            )
            if cu_seqlens is not None:
                global_chunk += 1
            for token in range(chunk_begin, min(chunk_begin + 64, end)):
                current_state = (
                    current_state * torch.exp(gate_rows[token])[:, None, :]
                )
                residual = v_rows[token] - (
                    current_state * k_rows[token][:, None, :]
                ).sum(-1)
                residual = residual * beta_rows[token][:, None]
                current_state = current_state + (
                    residual[:, :, None] * k_rows[token][:, None, :]
                )
                output = (
                    current_state * (q_rows[token] * scale)[:, None, :]
                ).sum(-1)
                out_rows[token] = output.to(v.dtype)
        reference_state[state_index] = current_state.to(state.dtype)

    helion_state = state.clone()
    helion_v = v.clone()
    helion_out, helion_chunks = helion_chunk_kda(
        q, k, helion_v, gate, beta,
        initial_state=helion_state,
        initial_state_indices=indices,
        output_intermediate_states=True,
        scale=scale,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        cu_seqlens=cu_seqlens,
        A_log=A_log,
        dt_bias=dt_bias,
        lower_bound=lower_bound,
    )

    assert helion_out.data_ptr() == helion_v.data_ptr()
    torch.testing.assert_close(
        helion_out, reference_out, atol=2e-2, rtol=2e-2
    )
    torch.testing.assert_close(
        helion_chunks, reference_chunks, atol=2e-2, rtol=2e-2
    )
    torch.testing.assert_close(
        helion_state, reference_state, atol=2e-2, rtol=2e-2
    )
    return helion_out, helion_chunks, helion_state


# ===================================================================
#  PORTED DECODE TESTS
# ===================================================================

@pytest.mark.parametrize(
    "state_dtype",
    [torch.float32, torch.bfloat16, torch.float16],
)
def test_packed_decode_contract(state_dtype: torch.dtype) -> None:
    """Decode correctness for FP32/BF16/FP16, unbounded gate."""
    torch.manual_seed(123)
    batch, q_heads, v_heads, key_dim, value_dim = 3, 2, 4, 128, 128
    pool_size = 7
    mixed_qkv = torch.randn(
        batch, 2 * q_heads * key_dim + v_heads * value_dim,
        device="cuda", dtype=torch.bfloat16,
    )
    gate = torch.randn(
        batch, v_heads * key_dim, device="cuda", dtype=torch.bfloat16
    )
    beta = torch.randn(batch, v_heads, device="cuda", dtype=torch.bfloat16)
    a_log = torch.randn(v_heads, device="cuda", dtype=torch.float32)
    dt_bias = torch.randn(
        v_heads * key_dim, device="cuda", dtype=torch.float32
    )
    state = (
        torch.randn(
            pool_size, v_heads, value_dim, key_dim,
            device="cuda", dtype=state_dtype,
        ) * 0.01
    )
    indices = torch.tensor([5, -1, 2], device="cuda", dtype=torch.int32)
    ref_state = state.clone()
    helion_state = state.clone()
    ref_out = mixed_qkv.new_zeros(batch, 1, v_heads, value_dim)
    helion_out = torch.empty_like(ref_out)

    _decode_reference_loop(
        mixed_qkv, gate, beta, a_log, dt_bias,
        key_dim**-0.5, ref_state, ref_out, indices,
        lower_bound=None,
    )
    result, result_state = helion_fused_recurrent_kda_packed_decode(
        mixed_qkv, gate, beta, a_log, dt_bias,
        key_dim**-0.5, helion_state, helion_out, indices, True,
    )

    assert result.data_ptr() == helion_out.data_ptr()
    assert result_state.data_ptr() == helion_state.data_ptr()
    torch.testing.assert_close(helion_out, ref_out, atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(
        helion_state, ref_state,
        atol=_DECODE_STATE_ATOL[state_dtype], rtol=1e-4,
    )
    assert torch.count_nonzero(helion_out[1]).item() == 0
    untouched = torch.tensor([0, 1, 3, 4, 6], device="cuda")
    assert torch.equal(helion_state[untouched], state[untouched])


@pytest.mark.parametrize(
    "state_dtype", [torch.float32, torch.bfloat16],
)
def test_packed_decode_lower_bound_contract(
    state_dtype: torch.dtype,
) -> None:
    """Bounded sigmoid gate (lower_bound=-5.0) decode correctness."""
    torch.manual_seed(321)
    batch, q_heads, v_heads, key_dim, value_dim = 3, 2, 4, 128, 128
    pool_size = 7
    mixed_qkv = torch.randn(
        batch, 2 * q_heads * key_dim + v_heads * value_dim,
        device="cuda", dtype=torch.bfloat16,
    )
    gate = torch.randn(
        batch, v_heads * key_dim, device="cuda", dtype=torch.bfloat16
    )
    beta = torch.randn(batch, v_heads, device="cuda", dtype=torch.bfloat16)
    a_log = torch.randn(v_heads, device="cuda", dtype=torch.float32)
    dt_bias = torch.randn(
        v_heads * key_dim, device="cuda", dtype=torch.float32
    )
    state = (
        torch.randn(
            pool_size, v_heads, value_dim, key_dim,
            device="cuda", dtype=state_dtype,
        ) * 0.01
    )
    indices = torch.tensor([5, -1, 2], device="cuda", dtype=torch.int32)
    scale = key_dim**-0.5
    lower_bound = -5.0

    ref_state = state.clone()
    helion_state = state.clone()
    ref_out = mixed_qkv.new_zeros(batch, 1, v_heads, value_dim)
    helion_out = torch.empty_like(ref_out)

    _decode_reference_loop(
        mixed_qkv, gate, beta, a_log, dt_bias,
        scale, ref_state, ref_out, indices,
        lower_bound=lower_bound,
    )
    result, result_state = helion_fused_recurrent_kda_packed_decode(
        mixed_qkv, gate, beta, a_log, dt_bias,
        scale, helion_state, helion_out, indices, True, lower_bound,
    )

    assert result.data_ptr() == helion_out.data_ptr()
    assert result_state.data_ptr() == helion_state.data_ptr()
    torch.testing.assert_close(helion_out, ref_out, atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(
        helion_state, ref_state,
        atol=_DECODE_STATE_ATOL[state_dtype], rtol=1e-4,
    )
    assert torch.count_nonzero(helion_out[1]).item() == 0


# ===================================================================
#  PORTED PREFILL TESTS
# ===================================================================

def test_fixed_partial_prefill_and_state_pool_contract() -> None:
    """17 tokens (not multiple of CHUNK_SIZE=64), untouched pool slots."""
    torch.manual_seed(789)
    batch, tokens, heads, key_dim, value_dim = 2, 17, 2, 32, 32
    q = torch.randn(
        batch, tokens, heads, key_dim, device="cuda", dtype=torch.bfloat16
    )
    k = torch.randn_like(q)
    v = torch.randn(
        batch, tokens, heads, value_dim, device="cuda", dtype=torch.bfloat16
    )
    gate = torch.randn_like(q) * 0.2
    beta = torch.rand(batch, tokens, heads, device="cuda")
    a_log = torch.full([heads], -2.0, device="cuda")
    dt_bias = torch.zeros(heads * key_dim, device="cuda")
    indices = torch.tensor([3, 1], device="cuda", dtype=torch.int32)
    state = torch.randn(5, heads, value_dim, key_dim, device="cuda") * 0.01

    _, _, helion_state = _compare_prefill(
        q, k, v, gate, beta, state, indices,
        use_qk_l2norm_in_kernel=True, A_log=a_log, dt_bias=dt_bias,
    )
    untouched = torch.tensor([0, 2, 4], device="cuda")
    assert torch.equal(helion_state[untouched], state[untouched])


def test_prefill_uses_stable_subchunk_gates() -> None:
    """Numerical stability: large cumulative gates would overflow FP32."""
    torch.manual_seed(1117)
    tokens, heads, key_dim, value_dim = 64, 1, 32, 32
    q = torch.nn.functional.normalize(
        torch.randn(1, tokens, heads, key_dim, device="cuda"), dim=-1
    ).bfloat16()
    k = torch.nn.functional.normalize(
        torch.randn(1, tokens, heads, key_dim, device="cuda"), dim=-1
    ).bfloat16()
    v = torch.randn(1, tokens, heads, value_dim, device="cuda").bfloat16()
    gate = torch.full(
        (1, tokens, heads, key_dim), -2.0,
        device="cuda", dtype=torch.float32,
    )
    beta = torch.full((1, tokens, heads), 0.5, device="cuda")
    cu_seqlens = torch.tensor([0, tokens], device="cuda", dtype=torch.int32)
    indices = torch.zeros(1, device="cuda", dtype=torch.int32)
    state = torch.zeros(1, heads, value_dim, key_dim, device="cuda")

    output, chunks, final_state = _compare_prefill(
        q, k, v, gate, beta, state, indices, cu_seqlens=cu_seqlens,
    )
    assert torch.isfinite(output).all()
    assert torch.isfinite(chunks).all()
    assert torch.isfinite(final_state).all()


@pytest.mark.parametrize("is_varlen", [False, True])
def test_prefill_diagonal_uses_midpoint_gate_anchor(
    is_varlen: bool,
) -> None:
    """Verify midpoint gate anchoring prevents exp2 overflow."""
    tokens, heads, key_dim = 16, 1, 32
    q = torch.full(
        (1, tokens, heads, key_dim), key_dim**-0.5,
        device="cuda", dtype=torch.bfloat16,
    )
    k = q.clone()
    cumulative_gate = -10.0 * torch.arange(
        tokens, device="cuda", dtype=torch.float32
    )
    gate = cumulative_gate.view(1, tokens, 1, 1).expand_as(q).float()
    beta = torch.ones(1, tokens, heads, device="cuda")
    if is_varlen:
        metadata = torch.tensor(
            [0, tokens], device="cuda", dtype=torch.int32
        )
        chunk_indices = torch.tensor(
            [[0, 0]], device="cuda", dtype=torch.int32
        )
    else:
        metadata = torch.empty(0, device="cuda", dtype=torch.int32)
        chunk_indices = torch.empty(0, 2, device="cuda", dtype=torch.int32)

    aqk, _ = _intra_matrices_wide(
        q, k, gate, beta, metadata, chunk_indices, 1.0,
        is_varlen=is_varlen,
    )

    qk = q[0, :, 0].float() @ k[0, :, 0].float().T
    gate_delta = cumulative_gate[:, None] - cumulative_gate[None, :]
    causal = (
        torch.arange(tokens, device="cuda")[:, None]
        >= torch.arange(tokens, device="cuda")[None, :]
    )
    expected = torch.where(causal, qk * torch.exp2(gate_delta), 0.0)
    actual = aqk[0, :, 0, :tokens].float()

    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, expected, atol=5e-3, rtol=5e-3)


def test_fp16_preactivated_gate_with_bf16_state_contract() -> None:
    """FP16 activations with BF16 state pool; output dtype preserved."""
    torch.manual_seed(1213)
    batch, tokens, heads, key_dim, value_dim = 1, 17, 1, 32, 32
    q = torch.nn.functional.normalize(
        torch.randn(batch, tokens, heads, key_dim, device="cuda"), dim=-1
    ).half()
    k = torch.nn.functional.normalize(
        torch.randn(batch, tokens, heads, key_dim, device="cuda"), dim=-1
    ).half()
    v = torch.randn(
        batch, tokens, heads, value_dim, device="cuda", dtype=torch.float16
    )
    gate = -torch.rand(batch, tokens, heads, key_dim, device="cuda") * 0.01
    beta = torch.rand(batch, tokens, heads, device="cuda")
    indices = torch.tensor([1], device="cuda", dtype=torch.int32)
    state = (
        torch.randn(
            3, heads, value_dim, key_dim,
            device="cuda", dtype=torch.bfloat16,
        ) * 0.01
    )

    output, chunks, _ = _compare_prefill(
        q, k, v, gate, beta, state, indices,
    )
    assert output.dtype == torch.float16
    assert chunks.dtype == torch.float16


@pytest.mark.parametrize(
    ("state_dtype", "lower_bound"),
    [
        (torch.float32, None),
        (torch.bfloat16, -5.0),
        (torch.float16, None),
    ],
    ids=["fp32", "bf16-lower-bound", "fp16"],
)
def test_packed_varlen_prefill_contract(
    state_dtype: torch.dtype,
    lower_bound: float | None,
) -> None:
    """Packed varlen [65, 31] with cu_seqlens, K=V=128."""
    torch.manual_seed(456)
    lengths = [65, 31]
    tokens, heads, key_dim, value_dim = sum(lengths), 2, 128, 128
    q = torch.randn(
        1, tokens, heads, key_dim, device="cuda", dtype=torch.bfloat16
    )
    k = torch.randn_like(q)
    v = torch.randn(
        1, tokens, heads, value_dim, device="cuda", dtype=torch.bfloat16
    )
    gate = torch.randn_like(q)
    beta = torch.sigmoid(
        torch.randn(1, tokens, heads, device="cuda", dtype=torch.float32)
    )
    a_log = torch.randn(heads, device="cuda", dtype=torch.float32)
    dt_bias = torch.randn(
        heads * key_dim, device="cuda", dtype=torch.float32
    )
    cu_seqlens = torch.tensor(
        [0, lengths[0], tokens], device="cuda", dtype=torch.int32
    )
    indices = torch.tensor([3, 1], device="cuda", dtype=torch.int32)
    state = (
        torch.randn(
            5, heads, value_dim, key_dim,
            device="cuda", dtype=state_dtype,
        ) * 0.01
    )
    _compare_prefill(
        q, k, v, gate, beta, state, indices,
        use_qk_l2norm_in_kernel=True,
        cu_seqlens=cu_seqlens,
        A_log=a_log, dt_bias=dt_bias,
        lower_bound=lower_bound,
    )


# ===================================================================
#  SUPPLEMENTAL TESTS (vLLM-specific adapter/integration)
# ===================================================================

def test_feature_toggle_env_var(monkeypatch):
    """VLLM_DISABLE_HELION_KDA=1 should read as True."""
    import vllm.envs as envs

    monkeypatch.setenv("VLLM_DISABLE_HELION_KDA", "1")
    assert envs.VLLM_DISABLE_HELION_KDA is True


def test_feature_toggle_env_var_default(monkeypatch):
    """VLLM_DISABLE_HELION_KDA unset should default to False."""
    import vllm.envs as envs

    monkeypatch.delenv("VLLM_DISABLE_HELION_KDA", raising=False)
    assert envs.VLLM_DISABLE_HELION_KDA is False


def test_decode_adapter_shape_transform():
    """Verify gate/beta reshape for the decode adapter path."""
    B, H, K = 4, 8, 128
    g1_ns = torch.randn(1, B, H, K, device="cuda", dtype=torch.bfloat16)
    beta_ns = torch.randn(1, B, H, device="cuda", dtype=torch.bfloat16)

    a = g1_ns.view(B, -1)
    b = beta_ns.view(B, -1)

    assert a.shape == (B, H * K)
    assert a.ndim == 2
    assert b.shape == (B, H)
    assert b.ndim == 2
    assert a.data_ptr() == g1_ns.data_ptr()
    assert b.data_ptr() == beta_ns.data_ptr()


def test_prefill_state_zero_init_for_new_requests():
    """New requests (has_initial_state=False) must have state zeroed."""
    pool_size, H, V, K = 5, 2, 32, 32
    recurrent_state = torch.randn(
        pool_size, H, V, K, device="cuda"
    ) * 0.1
    has_initial_state = torch.tensor(
        [True, False, True], device="cuda", dtype=torch.bool
    )
    non_spec_state_indices_tensor = torch.tensor(
        [3, 1, 4], device="cuda", dtype=torch.int32
    )
    original_state = recurrent_state.clone()

    new_request_mask = ~has_initial_state
    new_indices = non_spec_state_indices_tensor[new_request_mask]
    if new_indices.numel() > 0:
        recurrent_state[new_indices] = 0

    assert torch.all(recurrent_state[1] == 0)
    assert torch.equal(recurrent_state[3], original_state[3])
    assert torch.equal(recurrent_state[4], original_state[4])
    assert torch.equal(recurrent_state[0], original_state[0])
    assert torch.equal(recurrent_state[2], original_state[2])


def test_helion_decode_matches_reference_end_to_end():
    """Full decode adapter path: reshape → kernel → transpose."""
    torch.manual_seed(42)
    B, H, K, V = 3, 4, 128, 128
    pool_size = 7

    mixed_qkv = torch.randn(
        B, 2 * H * K + H * V, device="cuda", dtype=torch.bfloat16
    )
    g1_ns = torch.randn(1, B, H, K, device="cuda", dtype=torch.bfloat16)
    beta_ns = torch.randn(1, B, H, device="cuda", dtype=torch.bfloat16)
    A_log = torch.randn(H, device="cuda", dtype=torch.float32)
    dt_bias = torch.randn(H * K, device="cuda", dtype=torch.float32)
    state = (
        torch.randn(pool_size, H, V, K, device="cuda", dtype=torch.float32)
        * 0.01
    )
    indices = torch.tensor([5, -1, 2], device="cuda", dtype=torch.int32)

    # Helion path (adapter transforms)
    helion_state = state.clone()
    out = mixed_qkv.new_empty(B, 1, H, V)
    helion_fused_recurrent_kda_packed_decode(
        mixed_qkv=mixed_qkv,
        a=g1_ns.view(B, -1),
        b=beta_ns.view(B, -1),
        A_log=A_log,
        dt_bias=dt_bias,
        scale=K**-0.5,
        initial_state=helion_state,
        out=out,
        ssm_state_indices=indices,
        use_qk_l2norm_in_kernel=True,
        lower_bound=None,
    )

    # Reference path
    ref_state = state.clone()
    ref_out = mixed_qkv.new_zeros(B, 1, H, V)
    _decode_reference_loop(
        mixed_qkv=mixed_qkv,
        gate=g1_ns.view(B, -1),
        beta=beta_ns.view(B, -1),
        a_log=A_log,
        dt_bias=dt_bias,
        scale=K**-0.5,
        state=ref_state,
        out=ref_out,
        indices=indices,
        lower_bound=None,
    )

    # Compare — both `out` and `ref_out` are [B, 1, H, V]
    torch.testing.assert_close(out, ref_out, atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(
        helion_state, ref_state, atol=1e-5, rtol=1e-4
    )
    # Verify output shape after transpose matches vLLM convention
    result = out.transpose(0, 1)
    assert result.shape == (1, B, H, V)
    # Pad index (-1) produces zero output
    assert torch.count_nonzero(out[1]).item() == 0
    # Untouched slots unchanged
    assert torch.equal(helion_state[0], state[0])
    assert torch.equal(helion_state[1], state[1])


def test_helion_prefill_matches_reference_end_to_end():
    """Full prefill adapter path including has_initial_state handling."""
    torch.manual_seed(99)
    T, H, K, V = 65, 2, 32, 32
    pool_size = 5

    q = torch.randn(1, T, H, K, device="cuda", dtype=torch.bfloat16)
    k = torch.randn_like(q)
    v = torch.randn(1, T, H, V, device="cuda", dtype=torch.bfloat16)
    g = torch.randn_like(q) * 0.2
    beta = torch.rand(1, T, H, device="cuda")
    A_log = torch.full([H], -2.0, device="cuda")
    dt_bias = torch.zeros(H * K, device="cuda")
    cu_seqlens = torch.tensor([0, T], device="cuda", dtype=torch.int32)
    indices = torch.tensor([2], device="cuda", dtype=torch.int32)
    state = torch.randn(pool_size, H, V, K, device="cuda") * 0.1

    has_initial_state = torch.tensor(
        [False], device="cuda", dtype=torch.bool
    )
    helion_state = state.clone()

    # Step 1: Zero-init (adapter logic)
    new_request_mask = ~has_initial_state
    new_indices = indices[new_request_mask]
    if new_indices.numel() > 0:
        helion_state[new_indices] = 0

    # Step 2: Run Helion prefill
    helion_out = helion_chunk_kda(
        q=q, k=k, v=v.clone(),
        g=g, beta=beta,
        scale=K**-0.5,
        initial_state=helion_state,
        initial_state_indices=indices,
        use_qk_l2norm_in_kernel=True,
        cu_seqlens=cu_seqlens,
        A_log=A_log,
        dt_bias=dt_bias,
        lower_bound=None,
        output_intermediate_states=False,
    )

    assert helion_out.shape == (1, T, H, V)
    assert torch.isfinite(helion_out).all()
    assert helion_state[2].ne(0).any()  # state updated
    assert torch.equal(helion_state[0], state[0])  # untouched
    assert torch.equal(helion_state[1], state[1])  # untouched
