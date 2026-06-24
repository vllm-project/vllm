# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Precision tests for vllm's fused_recurrent_gated_delta_rule Triton operator.

Exercises fused_recurrent_gated_delta_rule_fwd_kernel via its Python wrapper.
The kernel runs the gated delta rule recurrence over a sequence, maintaining a
per-head hidden state h of shape (V, K). Per timestep it decays the state by a
scalar gate, applies the delta-rule correction to v, updates the state with the
outer product v^T @ k, and reads out o = h @ (q * scale). Compared against a
naive float32 PyTorch reference.

Source: vllm/model_executor/layers/fla/ops/fused_recurrent.py
"""

import pytest
import torch

from vllm.model_executor.layers.fla.ops.fused_recurrent import (
    fused_recurrent_gated_delta_rule_fwd,
)
from vllm.platforms import current_platform

# fused_recurrent_gated_delta_rule_fwd dispatches a Triton kernel
# that requires a GPU-class backend.
if not (current_platform.is_cuda_alike() or current_platform.is_xpu()):
    pytest.skip(
        "fused_recurrent_gated_delta_rule Triton kernel requires a "
        "CUDA-alike or XPU device",
        allow_module_level=True,
    )

DEVICE = current_platform.device_type


def fused_recurrent_gated_delta_rule_ref(
    q, k, v, g, beta, scale, initial_state, indices,
    use_qk_l2norm=False,
):
    """Naive float32 reference for the gated delta rule recurrence.

    Args:
        q, k: (B, T, H, K) — queries / keys.
        v: (B, T, HV, V) — values (HV >= H for grouped value attention).
        g: (B, T, HV) — per-(head, timestep) scalar log-decay.
        beta: (B, T, HV) — per-(head, timestep) scalar update weight.
        scale: float applied to q before the read-out.
        initial_state: (Nstate, HV, V, K) — hidden state pool.
        indices: (B,) which row of initial_state each sequence uses (> 0).

    Returns:
        o: (B, T, HV, V) — outputs.
        final_state: (Nstate, HV, V, K) — state pool after the last timestep.
    """
    B, T, H, K = q.shape
    HV, V = v.shape[2], v.shape[3]
    rep = HV // H  # value heads per qk head (1 when HV == H)

    o = torch.zeros(B, T, HV, V, dtype=torch.float32, device=q.device)
    final_state = initial_state.float().clone()

    for b in range(B):
        for hv in range(HV):
            h = hv // rep
            sidx = int(indices[b])
            s = initial_state[sidx, hv].float().clone()  # (V, K)

            for t in range(T):
                q_t = q[b, t, h].float()  # (K,)
                k_t = k[b, t, h].float()  # (K,)
                v_t = v[b, t, hv].float()  # (V,)

                if use_qk_l2norm:
                    q_t = q_t / torch.sqrt((q_t * q_t).sum() + 1e-6)
                    k_t = k_t / torch.sqrt((k_t * k_t).sum() + 1e-6)
                q_t = q_t * scale

                s = s * torch.exp(g[b, t, hv].float())  # decay
                v_new = v_t - s @ k_t  # delta-rule correction (V,)
                v_new = v_new * beta[b, t, hv].float()
                s = s + torch.outer(v_new, k_t)  # state update (V, K)
                o[b, t, hv] = s @ q_t

            final_state[sidx, hv] = s

    return o, final_state


def _make_inputs(B, T, H, HV, K, V, dtype=torch.float32, use_initial_state=True):
    # Scales chosen so the recurrent output stays O(1) across all sequence
    # lengths used here. The delta-rule recurrence amplifies the state with T:
    # q,k ~ 0.5 already blows up to ~1e3 by T=16, which makes the comparison
    # dominated by rtol on huge values; q,k ~ 0.25 keeps |o| in [0.5, 1.5] for
    # T up to 32 so the test exercises a numerically healthy regime.
    q = torch.randn(B, T, H, K, device=DEVICE, dtype=dtype) * 0.25
    k = torch.randn(B, T, H, K, device=DEVICE, dtype=dtype) * 0.25
    v = torch.randn(B, T, HV, V, device=DEVICE, dtype=dtype)
    g = torch.randn(B, T, HV, device=DEVICE, dtype=torch.float32) * 0.05
    beta = torch.rand(B, T, HV, device=DEVICE, dtype=torch.float32).sigmoid()

    # Continuous-batching state pool: index 0 is NULL_BLOCK_ID (skipped by the
    # kernel), so sequences map to rows 1..B. ssm_state_indices is (B, T) with a
    # constant row per sequence.
    n_state = B + 1
    if use_initial_state:
        initial_state = torch.randn(
            n_state, HV, V, K, device=DEVICE, dtype=torch.float32
        ) * 0.01
    else:
        initial_state = torch.zeros(
            n_state, HV, V, K, device=DEVICE, dtype=torch.float32
        )
    indices = torch.arange(1, B + 1, device=DEVICE, dtype=torch.int32)
    ssm_state_indices = indices.unsqueeze(1).expand(B, T).contiguous()
    return q, k, v, g, beta, initial_state, indices, ssm_state_indices


# (B, T, H, HV, K, V, use_init) — HV > H exercises grouped value attention.
CONFIGS = [
    (1, 8, 4, 4, 64, 64, True),
    (2, 8, 4, 4, 64, 64, True),
    (1, 16, 4, 4, 64, 64, True),
    (1, 8, 4, 4, 64, 64, False),  # zero initial state
    (1, 8, 2, 4, 64, 64, True),  # GVA: HV = 2 * H
    (1, 8, 4, 4, 128, 128, True),  # K = V = 128
    (2, 32, 4, 8, 64, 64, True),  # GVA + long sequence (deep recurrence)
]


@pytest.mark.parametrize(
    "B,T,H,HV,K,V,use_init",
    CONFIGS,
    ids=[
        f"B{b}_T{t}_H{h}_HV{hv}_K{kk}_V{v}_init{int(ui)}"
        for b, t, h, hv, kk, v, ui in CONFIGS
    ],
)
@torch.inference_mode()
def test_fused_recurrent_gated_delta_rule(B, T, H, HV, K, V, use_init):
    """fused_recurrent_gated_delta_rule_fwd must match the naive reference (fp32)."""
    torch.manual_seed(0)
    scale = K ** -0.5
    q, k, v, g, beta, initial_state, indices, ssm_idx = _make_inputs(
        B, T, H, HV, K, V, use_initial_state=use_init
    )
    state_ref = initial_state.clone()

    o, final_state = fused_recurrent_gated_delta_rule_fwd(
        q, k, v, g, beta, scale=scale, initial_state=initial_state,
        inplace_final_state=True, ssm_state_indices=ssm_idx,
    )
    o_ref, state_ref = fused_recurrent_gated_delta_rule_ref(
        q, k, v, g, beta, scale, state_ref, indices,
    )

    torch.testing.assert_close(o.float(), o_ref, atol=1e-3, rtol=1e-3)
    # inplace_final_state writes the last-timestep state back into the pool.
    for b in range(B):
        idx = int(indices[b])
        torch.testing.assert_close(
            final_state[idx].float(), state_ref[idx], atol=1e-3, rtol=1e-3
        )


@torch.inference_mode()
def test_fused_recurrent_gated_delta_rule_l2norm():
    """use_qk_l2norm_in_kernel must match the reference with q/k normalized."""
    torch.manual_seed(0)
    B, T, H, HV, K, V = 1, 8, 4, 4, 64, 64
    scale = K ** -0.5
    q, k, v, g, beta, initial_state, indices, ssm_idx = _make_inputs(
        B, T, H, HV, K, V
    )
    state_ref = initial_state.clone()

    o, _ = fused_recurrent_gated_delta_rule_fwd(
        q, k, v, g, beta, scale=scale, initial_state=initial_state,
        inplace_final_state=True, ssm_state_indices=ssm_idx,
        use_qk_l2norm_in_kernel=True,
    )
    o_ref, _ = fused_recurrent_gated_delta_rule_ref(
        q, k, v, g, beta, scale, state_ref, indices, use_qk_l2norm=True,
    )

    torch.testing.assert_close(o.float(), o_ref, atol=1e-3, rtol=1e-3)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@torch.inference_mode()
def test_fused_recurrent_gated_delta_rule_low_precision(dtype):
    """bf16/fp16 inputs (cast to fp32 inside the kernel) match the fp32 reference."""
    torch.manual_seed(0)
    B, T, H, HV, K, V = 1, 8, 4, 4, 64, 64
    scale = K ** -0.5
    q, k, v, g, beta, initial_state, indices, ssm_idx = _make_inputs(
        B, T, H, HV, K, V, dtype=dtype
    )
    state_ref = initial_state.clone()

    o, _ = fused_recurrent_gated_delta_rule_fwd(
        q, k, v, g, beta, scale=scale, initial_state=initial_state,
        inplace_final_state=True, ssm_state_indices=ssm_idx,
    )
    o_ref, _ = fused_recurrent_gated_delta_rule_ref(
        q, k, v, g, beta, scale, state_ref, indices,
    )

    torch.testing.assert_close(o.float(), o_ref, atol=5e-3, rtol=5e-3)
