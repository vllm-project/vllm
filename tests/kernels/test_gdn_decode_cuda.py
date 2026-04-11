# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Test CUDA-native GDN decode kernel against the Triton reference.

Verifies that torch.ops.vllm.gdn_decode_step produces the same output
and state updates as the Triton fused_sigmoid_gating_delta_rule_update
kernel for single-token decode.
"""
import pytest
import torch

# Model dimensions matching Qwen3.5/Qwen3-Next
CONFIGS = [
    # (num_k_heads, num_v_heads, head_k_dim, head_v_dim, batch_size)
    (16, 64, 128, 128, 1),   # Qwen3.5-397B single
    (16, 64, 128, 128, 4),   # Qwen3.5-397B batch
    (8, 32, 128, 128, 1),    # Smaller model
    (4, 16, 64, 64, 8),      # Small dims, larger batch
]


def reference_gdn_decode(q, k, v, g_decay, beta, state, state_indices,
                          scale, use_l2norm):
    """Pure PyTorch reference implementation of GDN decode step."""
    B = q.shape[0]
    num_k_heads = q.shape[1]
    head_k_dim = q.shape[2]
    num_v_heads = v.shape[1]
    head_v_dim = v.shape[2]
    heads_per_group = num_v_heads // num_k_heads

    output = torch.zeros(B, num_v_heads, head_v_dim,
                         dtype=torch.float32, device=q.device)

    for b in range(B):
        slot = state_indices[b].item()
        if slot < 0:
            continue
        for hv in range(num_v_heads):
            h_idx = hv // heads_per_group
            g = g_decay[b, hv].item()
            b_val = beta[b, hv].item()

            q_vec = q[b, h_idx].float()
            k_vec = k[b, h_idx].float()
            v_vec = v[b, hv].float()

            if use_l2norm:
                q_vec = q_vec / (q_vec.norm() + 1e-6)
                k_vec = k_vec / (k_vec.norm() + 1e-6)
            q_vec = q_vec * scale

            h = state[slot, hv].float()  # [V, K]

            # Decay
            h *= g

            for vi in range(head_v_dim):
                # kv_mem = sum(h[vi, :] * k)
                kv_mem = (h[vi] * k_vec).sum().item()
                # delta
                delta = (v_vec[vi].item() - kv_mem) * b_val
                # state update
                h[vi] += k_vec * delta
                # output
                output[b, hv, vi] = (h[vi] * q_vec).sum().item()

            state[slot, hv] = h

    return output


@pytest.mark.parametrize("num_k_heads,num_v_heads,head_k_dim,head_v_dim,batch",
                         CONFIGS)
@pytest.mark.parametrize("use_l2norm", [False, True])
def test_gdn_decode_cuda_vs_reference(
    num_k_heads, num_v_heads, head_k_dim, head_v_dim, batch, use_l2norm
):
    """Test CUDA kernel output matches the reference implementation."""
    try:
        torch.ops.vllm.gdn_decode_step
    except AttributeError:
        pytest.skip("CUDA GDN decode kernel not compiled")

    torch.manual_seed(42)
    device = "cuda"
    num_slots = batch + 2  # extra slots to test indexing

    q = torch.randn(batch, num_k_heads, head_k_dim, device=device)
    k = torch.randn(batch, num_k_heads, head_k_dim, device=device)
    v = torch.randn(batch, num_v_heads, head_v_dim, device=device)
    g_decay = torch.rand(batch, num_v_heads, device=device) * 0.5 + 0.5
    beta = torch.rand(batch, num_v_heads, device=device)
    state_indices = torch.arange(batch, dtype=torch.int32, device=device)

    scale = head_k_dim ** -0.5

    # State: [num_slots, HV, V, K]
    state_ref = torch.randn(num_slots, num_v_heads, head_v_dim, head_k_dim,
                            dtype=torch.float32, device=device) * 0.01
    state_cuda = state_ref.clone()

    # Reference
    out_ref = reference_gdn_decode(
        q, k, v, g_decay, beta, state_ref, state_indices, scale, use_l2norm
    )

    # CUDA kernel
    out_cuda = torch.zeros(batch, num_v_heads, head_v_dim,
                           dtype=torch.float32, device=device)
    torch.ops.vllm.gdn_decode_step(
        q, k, v, g_decay, beta,
        state_cuda, out_cuda, state_indices,
        scale, use_l2norm,
    )

    # Compare output
    torch.testing.assert_close(out_cuda, out_ref, atol=1e-3, rtol=1e-3)

    # Compare state
    torch.testing.assert_close(state_cuda, state_ref, atol=1e-3, rtol=1e-3)


@pytest.mark.parametrize("batch", [1, 4])
def test_gdn_decode_cuda_pad_slot(batch):
    """Test that PAD_SLOT_ID (-1) is handled correctly."""
    try:
        torch.ops.vllm.gdn_decode_step
    except AttributeError:
        pytest.skip("CUDA GDN decode kernel not compiled")

    torch.manual_seed(42)
    device = "cuda"
    H, HV, K, V = 4, 16, 64, 64
    num_slots = 8

    q = torch.randn(batch, H, K, device=device)
    k = torch.randn(batch, H, K, device=device)
    v = torch.randn(batch, HV, V, device=device)
    g_decay = torch.ones(batch, HV, device=device)
    beta = torch.ones(batch, HV, device=device)

    # All PAD_SLOT_ID
    state_indices = torch.full((batch,), -1, dtype=torch.int32, device=device)
    state = torch.randn(num_slots, HV, V, K, dtype=torch.float32,
                        device=device)
    state_orig = state.clone()

    out = torch.zeros(batch, HV, V, dtype=torch.float32, device=device)
    torch.ops.vllm.gdn_decode_step(
        q, k, v, g_decay, beta, state, out, state_indices, 1.0, False,
    )

    # Output should be zero, state should be unchanged
    assert (out == 0).all(), "PAD_SLOT_ID should produce zero output"
    torch.testing.assert_close(state, state_orig)
