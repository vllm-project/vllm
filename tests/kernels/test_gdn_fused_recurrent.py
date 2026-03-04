# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Unit tests for the GDN (GatedDeltaNet) fused recurrent kernel.

Covers:
  - Basic forward correctness against a pure-PyTorch reference (B=1)
  - Continuous batching with valid ssm_state_indices
  - Regression test for PAD_SLOT_ID (-1) in ssm_state_indices (PR #33326)
  - Variable-length sequences via cu_seqlens + ssm_state_indices
  - Zero-length sequences
  - Inplace final state update
  - Speculative decoding path with num_accepted_tokens
  - Headwise beta variant
  - Grouped value attention (HV > H)
  - Half-precision inputs
"""

import pytest
import torch
import torch.nn.functional as F

from vllm.model_executor.layers.fla.ops.fused_recurrent import (
    fused_recurrent_gated_delta_rule_fwd,
)
from vllm.utils.torch_utils import set_random_seed
from vllm.v1.attention.backends.utils import PAD_SLOT_ID

# ---------------------------------------------------------------------------
# Pure-PyTorch reference implementation
# ---------------------------------------------------------------------------


def gated_delta_rule_recurrent_ref(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float,
    initial_state: torch.Tensor | None = None,
    is_beta_headwise: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Reference implementation of the gated delta rule recurrence.

    Args:
        q: [B, T, H, K]
        k: [B, T, H, K]
        v: [B, T, HV, V]
        g: [B, T, HV]  -- gating in log space
        beta: [B, T, HV] or [B, T, HV, V] if headwise
        scale: scalar
        initial_state: [B, HV, V, K] or None
        is_beta_headwise: if True, beta has shape [B, T, HV, V]
    Returns:
        o: [B, T, HV, V]
        final_state: [B, HV, V, K]
    """
    B, T, H, K = q.shape
    HV = v.shape[2]
    V = v.shape[3]
    assert HV % H == 0
    gva_ratio = HV // H

    device = q.device
    dtype = torch.float32

    q = q.float()
    k = k.float()
    v = v.float()
    g = g.float()
    beta = beta.float()

    if initial_state is not None:
        h = initial_state.clone().float()
    else:
        h = torch.zeros(B, HV, V, K, device=device, dtype=dtype)

    o = torch.zeros(B, T, HV, V, device=device, dtype=dtype)

    for b in range(B):
        for t in range(T):
            for hv in range(HV):
                h_idx = hv // gva_ratio
                b_g = g[b, t, hv].item()
                h[b, hv] *= torch.exp(torch.tensor(b_g, dtype=dtype))

                b_k = k[b, t, h_idx]
                b_v = v[b, t, hv]
                v_delta = b_v - (h[b, hv] * b_k.unsqueeze(0)).sum(dim=1)

                # For headwise beta, beta[b,t,hv] is a [V] vector;
                # for scalar beta, beta[b,t,hv] is a scalar.
                # Both cases are handled identically by broadcasting.
                v_delta = v_delta * beta[b, t, hv]

                h[b, hv] += v_delta.unsqueeze(1) * b_k.unsqueeze(0)

                b_q = q[b, t, h_idx] * scale
                o[b, t, hv] = (h[b, hv] * b_q.unsqueeze(0)).sum(dim=1)

    return o, h


def gated_delta_rule_ref_per_seq(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float,
    cu_seqlens: torch.Tensor,
    initial_states: torch.Tensor,
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    """
    Reference for variable-length sequences, returning per-sequence outputs
    and final states.

    q, k, v, g, beta: [1, total_T, ...]
    cu_seqlens: [N+1]
    initial_states: [N, HV, V, K]

    Returns:
        all_o: [1, total_T, HV, V]
        final_states: list of [1, HV, V, K] tensors per sequence
    """
    N = len(cu_seqlens) - 1
    all_o = torch.zeros_like(v, dtype=torch.float32)
    final_states = []

    for i in range(N):
        start = cu_seqlens[i].item()
        end = cu_seqlens[i + 1].item()
        seq_len = end - start

        if seq_len == 0:
            final_states.append(initial_states[i : i + 1].float())
            continue

        o_i, fs = gated_delta_rule_recurrent_ref(
            q[:, start:end],
            k[:, start:end],
            v[:, start:end],
            g[:, start:end],
            beta[:, start:end],
            scale,
            initial_states[i : i + 1],
        )
        all_o[:, start:end] = o_i
        final_states.append(fs)

    return all_o, final_states


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

DEVICE = "cuda:0"


def make_gdn_inputs(
    B: int,
    T: int,
    H: int,
    HV: int,
    K: int,
    V_dim: int,
    dtype: torch.dtype = torch.float32,
    seed: int = 42,
):
    """Create random GDN inputs for a single-batch (B=1) scenario."""
    set_random_seed(seed)
    device = DEVICE
    q = torch.randn(B, T, H, K, device=device, dtype=dtype)
    k = F.normalize(torch.randn(B, T, H, K, device=device, dtype=dtype), p=2, dim=-1)
    v = torch.randn(B, T, HV, V_dim, device=device, dtype=dtype)
    g = F.logsigmoid(torch.randn(B, T, HV, device=device, dtype=dtype))
    beta = torch.rand(B, T, HV, device=device, dtype=dtype).sigmoid()
    initial_state = torch.randn(B, HV, V_dim, K, device=device, dtype=dtype) * 0.1
    scale = K**-0.5
    return q, k, v, g, beta, initial_state, scale


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "T,H,HV,K,V",
    [
        (1, 1, 1, 16, 16),
        (4, 2, 2, 16, 16),
        (8, 2, 4, 32, 16),
        (16, 4, 4, 16, 32),
    ],
)
@pytest.mark.parametrize("dtype", [torch.float32])
@torch.inference_mode()
def test_basic_forward(T, H, HV, K, V, dtype):
    """Test basic forward pass (B=1, no cu_seqlens) against reference."""
    B = 1
    q, k, v, g, beta, initial_state, scale = make_gdn_inputs(B, T, H, HV, K, V, dtype)

    o_kernel, fs_kernel = fused_recurrent_gated_delta_rule_fwd(
        q=q.contiguous(),
        k=k.contiguous(),
        v=v.contiguous(),
        g=g.contiguous(),
        beta=beta.contiguous(),
        scale=scale,
        initial_state=initial_state.clone(),
        inplace_final_state=False,
    )

    o_ref, fs_ref = gated_delta_rule_recurrent_ref(
        q, k, v, g, beta, scale, initial_state.clone()
    )

    torch.testing.assert_close(o_kernel.float(), o_ref, atol=1e-3, rtol=1e-3)
    # Non-inplace: final state at position T-1
    torch.testing.assert_close(
        fs_kernel[T - 1 : T].float(), fs_ref, atol=1e-3, rtol=1e-3
    )


@pytest.mark.parametrize(
    "num_seqs,H,HV,K,V",
    [
        (2, 2, 2, 16, 16),
        (4, 2, 2, 16, 16),
        (3, 2, 4, 16, 16),
        (8, 4, 4, 16, 32),
    ],
)
@pytest.mark.parametrize("dtype", [torch.float32])
@torch.inference_mode()
def test_multi_seq_decode(num_seqs, H, HV, K, V, dtype):
    """Test decode (T=1) with multiple sequences via continuous batching.

    This is the production path for GDN decode in vLLM: each sequence
    has exactly 1 token, ssm_state_indices is 1D of shape [num_seqs].
    """
    T = 1  # Decode: single token per sequence
    total_T = num_seqs * T
    scale = K**-0.5
    set_random_seed(42)

    q = torch.randn(1, total_T, H, K, device=DEVICE, dtype=dtype)
    k = F.normalize(
        torch.randn(1, total_T, H, K, device=DEVICE, dtype=dtype),
        p=2,
        dim=-1,
    )
    v = torch.randn(1, total_T, HV, V, device=DEVICE, dtype=dtype)
    g = F.logsigmoid(torch.randn(1, total_T, HV, device=DEVICE, dtype=dtype))
    beta = torch.rand(1, total_T, HV, device=DEVICE, dtype=dtype).sigmoid()
    initial_state = torch.randn(num_seqs, HV, V, K, device=DEVICE, dtype=dtype) * 0.1

    cu_seqlens = torch.arange(0, num_seqs + 1, dtype=torch.long, device=DEVICE) * T
    ssm_state_indices = torch.arange(num_seqs, dtype=torch.int32, device=DEVICE)

    state_backup = initial_state.clone()

    o_kernel, fs_kernel = fused_recurrent_gated_delta_rule_fwd(
        q=q.contiguous(),
        k=k.contiguous(),
        v=v.contiguous(),
        g=g.contiguous(),
        beta=beta.contiguous(),
        scale=scale,
        initial_state=initial_state,
        inplace_final_state=True,
        cu_seqlens=cu_seqlens,
        ssm_state_indices=ssm_state_indices,
    )

    o_ref, fs_list = gated_delta_rule_ref_per_seq(
        q, k, v, g, beta, scale, cu_seqlens, state_backup
    )

    torch.testing.assert_close(o_kernel.float(), o_ref, atol=1e-3, rtol=1e-3)
    for i in range(num_seqs):
        torch.testing.assert_close(
            fs_kernel[i : i + 1].float(),
            fs_list[i],
            atol=1e-3,
            rtol=1e-3,
        )


@pytest.mark.parametrize("num_valid", [1, 2, 3])
@torch.inference_mode()
def test_pad_slot_id_regression(num_valid):
    """
    Regression test for PR #33326: PAD_SLOT_ID (-1) in ssm_state_indices
    must not cause a crash (negative memory offset).

    Simulates CUDAGraph padding where some sequences have PAD_SLOT_ID = -1.
    """
    T = 1
    H, HV, K, V = 2, 2, 16, 16
    total_seqs = 4
    scale = K**-0.5

    set_random_seed(42)

    total_T = num_valid * T
    q = torch.randn(1, total_T, H, K, device=DEVICE, dtype=torch.float32)
    k = F.normalize(
        torch.randn(1, total_T, H, K, device=DEVICE, dtype=torch.float32),
        p=2,
        dim=-1,
    )
    v = torch.randn(1, total_T, HV, V, device=DEVICE, dtype=torch.float32)
    g = F.logsigmoid(torch.randn(1, total_T, HV, device=DEVICE, dtype=torch.float32))
    beta = torch.rand(1, total_T, HV, device=DEVICE, dtype=torch.float32).sigmoid()

    num_slots = num_valid + 2
    state_pool = (
        torch.randn(num_slots, HV, V, K, device=DEVICE, dtype=torch.float32) * 0.1
    )
    state_pool_backup = state_pool.clone()

    # First num_valid are valid, rest are PAD_SLOT_ID
    indices = list(range(num_valid)) + [PAD_SLOT_ID] * (total_seqs - num_valid)
    ssm_state_indices = torch.tensor(indices, dtype=torch.int32, device=DEVICE)

    # Valid seqs have T=1 token, padded seqs have 0 tokens
    seqlens = [0]
    cumsum = 0
    for i in range(total_seqs):
        if i < num_valid:
            cumsum += T
        seqlens.append(cumsum)
    cu_seqlens = torch.tensor(seqlens, dtype=torch.long, device=DEVICE)

    # This should NOT crash (the bug from PR #33326 was a crash here)
    o_kernel, fs_kernel = fused_recurrent_gated_delta_rule_fwd(
        q=q.contiguous(),
        k=k.contiguous(),
        v=v.contiguous(),
        g=g.contiguous(),
        beta=beta.contiguous(),
        scale=scale,
        initial_state=state_pool,
        inplace_final_state=True,
        cu_seqlens=cu_seqlens,
        ssm_state_indices=ssm_state_indices,
    )

    # Verify valid sequences produce correct output
    for i in range(num_valid):
        start = cu_seqlens[i].item()
        end = cu_seqlens[i + 1].item()
        slot = ssm_state_indices[i].item()

        init_i = state_pool_backup[slot : slot + 1].clone()
        _, fs_ref_i = gated_delta_rule_recurrent_ref(
            q[:, start:end],
            k[:, start:end],
            v[:, start:end],
            g[:, start:end],
            beta[:, start:end],
            scale,
            init_i,
        )
        torch.testing.assert_close(
            fs_kernel[slot : slot + 1].float(),
            fs_ref_i,
            atol=1e-3,
            rtol=1e-3,
        )


@torch.inference_mode()
def test_pad_slot_id_all_padded():
    """
    Test with ALL sequences padded (PAD_SLOT_ID).
    The kernel should not crash even when every state index is -1.
    """
    H, HV, K, V = 2, 2, 16, 16
    total_seqs = 4
    scale = K**-0.5

    set_random_seed(42)

    # No actual tokens
    q = torch.randn(1, 0, H, K, device=DEVICE, dtype=torch.float32)
    k = F.normalize(
        torch.randn(1, 0, H, K, device=DEVICE, dtype=torch.float32),
        p=2,
        dim=-1,
    )
    v = torch.randn(1, 0, HV, V, device=DEVICE, dtype=torch.float32)
    g = F.logsigmoid(torch.randn(1, 0, HV, device=DEVICE, dtype=torch.float32))
    beta = torch.rand(1, 0, HV, device=DEVICE, dtype=torch.float32).sigmoid()

    num_slots = 4
    state_pool = (
        torch.randn(num_slots, HV, V, K, device=DEVICE, dtype=torch.float32) * 0.1
    )

    ssm_state_indices = torch.full(
        (total_seqs,), PAD_SLOT_ID, dtype=torch.int32, device=DEVICE
    )
    cu_seqlens = torch.zeros(total_seqs + 1, dtype=torch.long, device=DEVICE)

    # Should not crash
    o_kernel, fs_kernel = fused_recurrent_gated_delta_rule_fwd(
        q=q.contiguous(),
        k=k.contiguous(),
        v=v.contiguous(),
        g=g.contiguous(),
        beta=beta.contiguous(),
        scale=scale,
        initial_state=state_pool,
        inplace_final_state=True,
        cu_seqlens=cu_seqlens,
        ssm_state_indices=ssm_state_indices,
    )


@pytest.mark.parametrize(
    "num_seqs",
    [2, 3, 5, 8],
)
@torch.inference_mode()
def test_variable_num_seqs_decode(num_seqs):
    """Test decode mode with various numbers of sequences.

    Each sequence has T=1 (decode). This tests the IS_VARLEN +
    IS_CONTINUOUS_BATCHING path that is used in production.
    """
    T = 1
    H, HV, K, V = 2, 2, 16, 16
    N = num_seqs
    total_T = N * T
    scale = K**-0.5

    set_random_seed(42)

    q = torch.randn(1, total_T, H, K, device=DEVICE, dtype=torch.float32)
    k = F.normalize(
        torch.randn(1, total_T, H, K, device=DEVICE, dtype=torch.float32),
        p=2,
        dim=-1,
    )
    v = torch.randn(1, total_T, HV, V, device=DEVICE, dtype=torch.float32)
    g = F.logsigmoid(torch.randn(1, total_T, HV, device=DEVICE, dtype=torch.float32))
    beta = torch.rand(1, total_T, HV, device=DEVICE, dtype=torch.float32).sigmoid()

    initial_state = torch.randn(N, HV, V, K, device=DEVICE, dtype=torch.float32) * 0.1

    cu_seqlens = torch.arange(0, N + 1, dtype=torch.long, device=DEVICE)
    ssm_state_indices = torch.arange(N, dtype=torch.int32, device=DEVICE)

    state_backup = initial_state.clone()

    o_kernel, fs_kernel = fused_recurrent_gated_delta_rule_fwd(
        q=q.contiguous(),
        k=k.contiguous(),
        v=v.contiguous(),
        g=g.contiguous(),
        beta=beta.contiguous(),
        scale=scale,
        initial_state=initial_state,
        inplace_final_state=True,
        cu_seqlens=cu_seqlens,
        ssm_state_indices=ssm_state_indices,
    )

    o_ref, fs_list = gated_delta_rule_ref_per_seq(
        q, k, v, g, beta, scale, cu_seqlens, state_backup
    )

    torch.testing.assert_close(o_kernel.float(), o_ref, atol=1e-3, rtol=1e-3)
    for i in range(N):
        torch.testing.assert_close(
            fs_kernel[i : i + 1].float(),
            fs_list[i],
            atol=1e-3,
            rtol=1e-3,
        )


@torch.inference_mode()
def test_zero_length_sequence():
    """Test that zero-length sequences are handled correctly (early return).

    Uses decode mode (T=1) with some sequences having 0 tokens.
    The kernel should skip zero-length sequences and process valid ones.
    """
    H, HV, K, V = 2, 2, 16, 16
    scale = K**-0.5

    set_random_seed(42)

    # 3 sequences: seq0 has 1 token, seq1 has 0 tokens, seq2 has 1 token
    total_T = 2
    N = 3
    q = torch.randn(1, total_T, H, K, device=DEVICE, dtype=torch.float32)
    k = F.normalize(
        torch.randn(1, total_T, H, K, device=DEVICE, dtype=torch.float32),
        p=2,
        dim=-1,
    )
    v = torch.randn(1, total_T, HV, V, device=DEVICE, dtype=torch.float32)
    g = F.logsigmoid(torch.randn(1, total_T, HV, device=DEVICE, dtype=torch.float32))
    beta = torch.rand(1, total_T, HV, device=DEVICE, dtype=torch.float32).sigmoid()

    initial_state = torch.randn(N, HV, V, K, device=DEVICE, dtype=torch.float32) * 0.1

    # seq0: tokens [0,1), seq1: tokens [1,1), seq2: tokens [1,2)
    cu_seqlens = torch.tensor([0, 1, 1, 2], dtype=torch.long, device=DEVICE)
    ssm_state_indices = torch.tensor([0, 1, 2], dtype=torch.int32, device=DEVICE)

    state_backup = initial_state.clone()

    o_kernel, fs_kernel = fused_recurrent_gated_delta_rule_fwd(
        q=q.contiguous(),
        k=k.contiguous(),
        v=v.contiguous(),
        g=g.contiguous(),
        beta=beta.contiguous(),
        scale=scale,
        initial_state=initial_state,
        inplace_final_state=True,
        cu_seqlens=cu_seqlens,
        ssm_state_indices=ssm_state_indices,
    )

    # Verify seq0 (1 token) against reference
    o_ref_seq0, fs_ref_seq0 = gated_delta_rule_recurrent_ref(
        q[:, :1],
        k[:, :1],
        v[:, :1],
        g[:, :1],
        beta[:, :1],
        scale,
        state_backup[0:1].clone(),
    )
    torch.testing.assert_close(
        o_kernel[:, :1].float(), o_ref_seq0, atol=1e-3, rtol=1e-3
    )
    torch.testing.assert_close(
        fs_kernel[0:1].float(), fs_ref_seq0, atol=1e-3, rtol=1e-3
    )

    # seq1 has 0 tokens -- its state should be unchanged
    torch.testing.assert_close(
        fs_kernel[1:2].float(), state_backup[1:2].float(), atol=0, rtol=0
    )

    # Verify seq2 (1 token) against reference
    o_ref_seq2, fs_ref_seq2 = gated_delta_rule_recurrent_ref(
        q[:, 1:2],
        k[:, 1:2],
        v[:, 1:2],
        g[:, 1:2],
        beta[:, 1:2],
        scale,
        state_backup[2:3].clone(),
    )
    torch.testing.assert_close(
        o_kernel[:, 1:2].float(), o_ref_seq2, atol=1e-3, rtol=1e-3
    )
    torch.testing.assert_close(
        fs_kernel[2:3].float(), fs_ref_seq2, atol=1e-3, rtol=1e-3
    )


@pytest.mark.parametrize("num_seqs", [1, 2, 4])
@torch.inference_mode()
def test_inplace_final_state(num_seqs):
    """Test inplace_final_state=True writes back to initial_state tensor.

    Uses decode mode (T=1) with continuous batching since INPLACE_FINAL_STATE
    requires ssm_state_indices, and 1D indices only work for T=1.
    """
    T = 1
    H, HV, K, V = 2, 2, 16, 16
    total_T = num_seqs * T
    scale = K**-0.5

    set_random_seed(42)

    q = torch.randn(1, total_T, H, K, device=DEVICE, dtype=torch.float32)
    k = F.normalize(
        torch.randn(1, total_T, H, K, device=DEVICE, dtype=torch.float32),
        p=2,
        dim=-1,
    )
    v = torch.randn(1, total_T, HV, V, device=DEVICE, dtype=torch.float32)
    g = F.logsigmoid(torch.randn(1, total_T, HV, device=DEVICE, dtype=torch.float32))
    beta = torch.rand(1, total_T, HV, device=DEVICE, dtype=torch.float32).sigmoid()

    initial_state = (
        torch.randn(num_seqs, HV, V, K, device=DEVICE, dtype=torch.float32) * 0.1
    )
    state_before = initial_state.clone()

    ssm_state_indices = torch.arange(num_seqs, dtype=torch.int32, device=DEVICE)
    cu_seqlens = torch.arange(0, num_seqs + 1, dtype=torch.long, device=DEVICE)

    o_kernel, fs_kernel = fused_recurrent_gated_delta_rule_fwd(
        q=q.contiguous(),
        k=k.contiguous(),
        v=v.contiguous(),
        g=g.contiguous(),
        beta=beta.contiguous(),
        scale=scale,
        initial_state=initial_state,
        inplace_final_state=True,
        cu_seqlens=cu_seqlens,
        ssm_state_indices=ssm_state_indices,
    )

    assert fs_kernel.data_ptr() == initial_state.data_ptr()

    _, fs_list = gated_delta_rule_ref_per_seq(
        q, k, v, g, beta, scale, cu_seqlens, state_before
    )
    for i in range(num_seqs):
        torch.testing.assert_close(
            initial_state[i : i + 1].float(),
            fs_list[i],
            atol=1e-3,
            rtol=1e-3,
        )


@torch.inference_mode()
def test_spec_decoding_with_pad_slot_id():
    """
    Test speculative decoding path (num_accepted_tokens) with PAD_SLOT_ID.

    In spec decoding, ssm_state_indices has shape [N, num_spec+1] and
    num_accepted_tokens indicates how many tokens were accepted per sequence.
    Some indices may be PAD_SLOT_ID = -1 for padded sequences.
    """
    H, HV, K, V = 2, 2, 16, 16
    num_spec = 3
    num_seqs = 3
    total_tokens = num_seqs * (num_spec + 1)
    scale = K**-0.5

    set_random_seed(42)

    q = torch.randn(1, total_tokens, H, K, device=DEVICE, dtype=torch.float32)
    k = F.normalize(
        torch.randn(1, total_tokens, H, K, device=DEVICE, dtype=torch.float32),
        p=2,
        dim=-1,
    )
    v = torch.randn(1, total_tokens, HV, V, device=DEVICE, dtype=torch.float32)
    g = F.logsigmoid(
        torch.randn(1, total_tokens, HV, device=DEVICE, dtype=torch.float32)
    )
    beta = torch.rand(1, total_tokens, HV, device=DEVICE, dtype=torch.float32).sigmoid()

    num_slots = 10
    state_pool = (
        torch.randn(num_slots, HV, V, K, device=DEVICE, dtype=torch.float32) * 0.1
    )

    # First 2 seqs valid, last padded
    ssm_indices = torch.tensor(
        [
            [0, 1, 2, 3],
            [4, 5, 6, 7],
            [PAD_SLOT_ID, PAD_SLOT_ID, PAD_SLOT_ID, PAD_SLOT_ID],
        ],
        dtype=torch.int32,
        device=DEVICE,
    )

    num_accepted_tokens = torch.tensor([3, 2, 1], dtype=torch.int32, device=DEVICE)

    cu_seqlens = torch.tensor([0, 4, 8, 12], dtype=torch.long, device=DEVICE)

    # Should not crash
    o_kernel, fs_kernel = fused_recurrent_gated_delta_rule_fwd(
        q=q.contiguous(),
        k=k.contiguous(),
        v=v.contiguous(),
        g=g.contiguous(),
        beta=beta.contiguous(),
        scale=scale,
        initial_state=state_pool,
        inplace_final_state=True,
        cu_seqlens=cu_seqlens,
        ssm_state_indices=ssm_indices,
        num_accepted_tokens=num_accepted_tokens,
    )


@pytest.mark.parametrize("T", [4, 8])
@torch.inference_mode()
def test_headwise_beta(T):
    """Test with headwise (per-V-dimension) beta tensor."""
    B = 1
    H, HV, K, V = 2, 2, 16, 16
    scale = K**-0.5

    set_random_seed(42)

    q = torch.randn(B, T, H, K, device=DEVICE, dtype=torch.float32)
    k = F.normalize(
        torch.randn(B, T, H, K, device=DEVICE, dtype=torch.float32),
        p=2,
        dim=-1,
    )
    v = torch.randn(B, T, HV, V, device=DEVICE, dtype=torch.float32)
    g = F.logsigmoid(torch.randn(B, T, HV, device=DEVICE, dtype=torch.float32))
    beta = torch.rand(B, T, HV, V, device=DEVICE, dtype=torch.float32).sigmoid()
    initial_state = torch.randn(B, HV, V, K, device=DEVICE, dtype=torch.float32) * 0.1

    o_kernel, fs_kernel = fused_recurrent_gated_delta_rule_fwd(
        q=q.contiguous(),
        k=k.contiguous(),
        v=v.contiguous(),
        g=g.contiguous(),
        beta=beta.contiguous(),
        scale=scale,
        initial_state=initial_state.clone(),
        inplace_final_state=False,
    )

    o_ref, fs_ref = gated_delta_rule_recurrent_ref(
        q, k, v, g, beta, scale, initial_state.clone(), is_beta_headwise=True
    )

    torch.testing.assert_close(o_kernel.float(), o_ref, atol=1e-3, rtol=1e-3)
    torch.testing.assert_close(
        fs_kernel[T - 1 : T].float(), fs_ref, atol=1e-3, rtol=1e-3
    )


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@torch.inference_mode()
def test_half_precision(dtype):
    """Test that the kernel works with half-precision inputs."""
    B, T, H, HV, K, V = 1, 4, 2, 2, 16, 16
    q, k, v, g, beta, initial_state, scale = make_gdn_inputs(
        B, T, H, HV, K, V, dtype=dtype
    )

    o_kernel, fs_kernel = fused_recurrent_gated_delta_rule_fwd(
        q=q.contiguous(),
        k=k.contiguous(),
        v=v.contiguous(),
        g=g.contiguous(),
        beta=beta.contiguous(),
        scale=scale,
        initial_state=initial_state.clone(),
        inplace_final_state=False,
    )

    o_ref, fs_ref = gated_delta_rule_recurrent_ref(
        q.float(),
        k.float(),
        v.float(),
        g.float(),
        beta.float(),
        scale,
        initial_state.clone().float(),
    )

    torch.testing.assert_close(o_kernel.float(), o_ref, atol=5e-2, rtol=5e-2)
    torch.testing.assert_close(
        fs_kernel[T - 1 : T].float(), fs_ref, atol=5e-2, rtol=5e-2
    )


@torch.inference_mode()
def test_pad_slot_id_with_nonzero_tokens():
    """
    Regression test: PAD_SLOT_ID (-1) with sequences that actually have tokens.

    This specifically targets the case where IS_CONTINUOUS_BATCHING=True and
    INPLACE_FINAL_STATE=True, and a sequence with tokens has a PAD_SLOT_ID
    state index. The kernel should return early without crashing.

    Before PR #33326, this would cause an illegal memory access because:
      p_h0 = h0 + (-1) * stride_init_state_token  ->  negative offset
    """
    T = 1
    H, HV, K, V = 2, 2, 16, 16
    N = 4
    scale = K**-0.5

    set_random_seed(42)

    q = torch.randn(1, N * T, H, K, device=DEVICE, dtype=torch.float32)
    k = F.normalize(
        torch.randn(1, N * T, H, K, device=DEVICE, dtype=torch.float32),
        p=2,
        dim=-1,
    )
    v = torch.randn(1, N * T, HV, V, device=DEVICE, dtype=torch.float32)
    g = F.logsigmoid(torch.randn(1, N * T, HV, device=DEVICE, dtype=torch.float32))
    beta = torch.rand(1, N * T, HV, device=DEVICE, dtype=torch.float32).sigmoid()

    num_slots = 4
    state_pool = (
        torch.randn(num_slots, HV, V, K, device=DEVICE, dtype=torch.float32) * 0.1
    )
    state_pool_backup = state_pool.clone()

    # Seqs 0 and 2 valid, seqs 1 and 3 have PAD_SLOT_ID
    ssm_state_indices = torch.tensor(
        [0, PAD_SLOT_ID, 2, PAD_SLOT_ID], dtype=torch.int32, device=DEVICE
    )

    # All sequences have 1 token each
    cu_seqlens = torch.tensor([0, 1, 2, 3, 4], dtype=torch.long, device=DEVICE)

    # Should NOT crash (the original bug from PR #33326)
    o_kernel, fs_kernel = fused_recurrent_gated_delta_rule_fwd(
        q=q.contiguous(),
        k=k.contiguous(),
        v=v.contiguous(),
        g=g.contiguous(),
        beta=beta.contiguous(),
        scale=scale,
        initial_state=state_pool,
        inplace_final_state=True,
        cu_seqlens=cu_seqlens,
        ssm_state_indices=ssm_state_indices,
    )

    # Verify valid sequences
    for i in [0, 2]:
        slot = ssm_state_indices[i].item()
        start = cu_seqlens[i].item()
        end = cu_seqlens[i + 1].item()
        init_i = state_pool_backup[slot : slot + 1].clone()
        _, fs_ref_i = gated_delta_rule_recurrent_ref(
            q[:, start:end],
            k[:, start:end],
            v[:, start:end],
            g[:, start:end],
            beta[:, start:end],
            scale,
            init_i,
        )
        torch.testing.assert_close(
            fs_kernel[slot : slot + 1].float(),
            fs_ref_i,
            atol=1e-3,
            rtol=1e-3,
        )


@torch.inference_mode()
def test_gva_grouped_value_attention():
    """Test grouped value attention where HV > H."""
    B, T, H, HV, K, V = 1, 4, 2, 4, 16, 16  # HV = 2 * H
    q, k, v, g, beta, initial_state, scale = make_gdn_inputs(B, T, H, HV, K, V)

    o_kernel, fs_kernel = fused_recurrent_gated_delta_rule_fwd(
        q=q.contiguous(),
        k=k.contiguous(),
        v=v.contiguous(),
        g=g.contiguous(),
        beta=beta.contiguous(),
        scale=scale,
        initial_state=initial_state.clone(),
        inplace_final_state=False,
    )

    o_ref, fs_ref = gated_delta_rule_recurrent_ref(
        q, k, v, g, beta, scale, initial_state.clone()
    )

    torch.testing.assert_close(o_kernel.float(), o_ref, atol=1e-3, rtol=1e-3)
    torch.testing.assert_close(
        fs_kernel[T - 1 : T].float(), fs_ref, atol=1e-3, rtol=1e-3
    )


@pytest.mark.parametrize(
    "num_seqs,slots",
    [
        (4, [3, 1, 0, 2]),  # non-contiguous mapping
        (3, [5, 0, 3]),  # sparse mapping into larger pool
    ],
)
@torch.inference_mode()
def test_noncontiguous_state_indices(num_seqs, slots):
    """Test that non-contiguous ssm_state_indices work correctly."""
    T = 1
    H, HV, K, V = 2, 2, 16, 16
    total_T = num_seqs * T
    scale = K**-0.5

    set_random_seed(42)

    q = torch.randn(1, total_T, H, K, device=DEVICE, dtype=torch.float32)
    k = F.normalize(
        torch.randn(1, total_T, H, K, device=DEVICE, dtype=torch.float32),
        p=2,
        dim=-1,
    )
    v = torch.randn(1, total_T, HV, V, device=DEVICE, dtype=torch.float32)
    g = F.logsigmoid(torch.randn(1, total_T, HV, device=DEVICE, dtype=torch.float32))
    beta = torch.rand(1, total_T, HV, device=DEVICE, dtype=torch.float32).sigmoid()

    num_slots = max(slots) + 1
    state_pool = (
        torch.randn(num_slots, HV, V, K, device=DEVICE, dtype=torch.float32) * 0.1
    )
    state_backup = state_pool.clone()

    ssm_state_indices = torch.tensor(slots, dtype=torch.int32, device=DEVICE)
    cu_seqlens = torch.arange(0, num_seqs + 1, dtype=torch.long, device=DEVICE) * T

    o_kernel, fs_kernel = fused_recurrent_gated_delta_rule_fwd(
        q=q.contiguous(),
        k=k.contiguous(),
        v=v.contiguous(),
        g=g.contiguous(),
        beta=beta.contiguous(),
        scale=scale,
        initial_state=state_pool,
        inplace_final_state=True,
        cu_seqlens=cu_seqlens,
        ssm_state_indices=ssm_state_indices,
    )

    for i in range(num_seqs):
        slot = slots[i]
        start = cu_seqlens[i].item()
        end = cu_seqlens[i + 1].item()
        init_i = state_backup[slot : slot + 1].clone()
        _, fs_ref_i = gated_delta_rule_recurrent_ref(
            q[:, start:end],
            k[:, start:end],
            v[:, start:end],
            g[:, start:end],
            beta[:, start:end],
            scale,
            init_i,
        )
        torch.testing.assert_close(
            fs_kernel[slot : slot + 1].float(),
            fs_ref_i,
            atol=1e-3,
            rtol=1e-3,
        )


if __name__ == "__main__":
    print("Running smoke tests...")
    test_basic_forward(1, 1, 1, 16, 16, torch.float32)
    print("  basic_forward (T=1): PASSED")
    test_basic_forward(4, 2, 2, 16, 16, torch.float32)
    print("  basic_forward (T=4): PASSED")
    test_multi_seq_decode(2, 2, 2, 16, 16, torch.float32)
    print("  multi_seq_decode: PASSED")
    test_pad_slot_id_regression(2)
    print("  pad_slot_id_regression: PASSED")
    test_pad_slot_id_with_nonzero_tokens()
    print("  pad_slot_id_with_nonzero_tokens: PASSED")
    test_pad_slot_id_all_padded()
    print("  pad_slot_id_all_padded: PASSED")
    test_variable_num_seqs_decode(3)
    print("  variable_num_seqs_decode: PASSED")
    test_zero_length_sequence()
    print("  zero_length_sequence: PASSED")
    test_inplace_final_state(2)
    print("  inplace_final_state: PASSED")
    test_gva_grouped_value_attention()
    print("  gva_grouped_value_attention: PASSED")
    test_half_precision(torch.float16)
    print("  half_precision (fp16): PASSED")
    test_headwise_beta(4)
    print("  headwise_beta: PASSED")
    test_noncontiguous_state_indices(4, [3, 1, 0, 2])
    print("  noncontiguous_state_indices: PASSED")
    test_spec_decoding_with_pad_slot_id()
    print("  spec_decoding_with_pad_slot_id: PASSED")
    print("All smoke tests passed!")
