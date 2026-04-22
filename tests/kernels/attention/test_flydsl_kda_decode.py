# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Correctness + availability guard for the FlyDSL KDA decode kernel path
wired up in ``vllm/model_executor/layers/kda.py``.

This test validates that, when enabled via ``VLLM_ROCM_USE_AITER_FLYDSL_KDA``,
the FlyDSL gated delta rule decode kernel
(``aiter.ops.flydsl.linear_attention_kernels.flydsl_gdr_decode``) produces
output numerically close to the FLA triton kernel
(``vllm.model_executor.layers.fla.ops.kda.fused_recurrent_kda``) on the exact
KDA shapes used by Kimi-Linear-48B-A3B.

The test is skipped automatically when:
  * The host is not ROCm.
  * AITER / FlyDSL is not importable (older AITER builds).
  * CUDA/HIP is unavailable.

Why the parametrize grid maps to Kimi-Linear:
  - head_dim = 128 (config.json: ``linear_attn_config.head_dim``)
  - num_heads = 32 total, so per-TP local heads are:
      TP=2 -> 16, TP=4 -> 8, TP=8 -> 4

Correctness tolerance: BF16 recurrent accumulation over the Kimi-Linear
shapes gives empirical max-abs-diff around 3-5e-3; we assert <= 1.5e-2 to
accommodate kernel / compiler drift without masking genuine regressions.
"""

from __future__ import annotations

import pytest
import torch

from vllm.platforms import current_platform


def _flydsl_kernel_or_skip():
    if not current_platform.is_rocm():
        pytest.skip("FlyDSL KDA kernel is ROCm-only")
    if not torch.cuda.is_available():
        pytest.skip("No GPU available for FlyDSL KDA test")
    try:
        from aiter.ops.flydsl.linear_attention_kernels import (  # noqa: E501
            flydsl_gdr_decode,
        )
    except ImportError as e:
        pytest.skip(f"aiter.ops.flydsl.flydsl_gdr_decode unavailable: {e}")
    return flydsl_gdr_decode


@pytest.mark.parametrize(
    "num_heads",
    [
        pytest.param(16, id="tp=2-nh=16"),
        pytest.param(8, id="tp=4-nh=8"),
        pytest.param(4, id="tp=8-nh=4"),
    ],
)
@pytest.mark.parametrize("bs", [1, 4])
@pytest.mark.parametrize("head_dim", [128])
@torch.inference_mode()
def test_flydsl_kda_decode_matches_fla(num_heads: int, bs: int, head_dim: int):
    """Head-to-head correctness: FLA triton vs FlyDSL on Kimi-Linear shapes.

    Runs one decode step (T=1) with matched random inputs and checks the
    kernels agree within BF16 tolerance.
    """
    flydsl_gdr_decode = _flydsl_kernel_or_skip()
    from vllm.model_executor.layers.fla.ops.kda import (
        fused_kda_gate,
        fused_recurrent_kda,
    )

    device = torch.device("cuda")
    dtype = torch.bfloat16
    T = 1  # pure decode

    gen = torch.Generator(device=device).manual_seed(0)
    q = torch.randn(bs, T, num_heads, head_dim, device=device, dtype=dtype, generator=gen) * 0.1
    k = torch.randn(bs, T, num_heads, head_dim, device=device, dtype=dtype, generator=gen) * 0.1
    v = torch.randn(bs, T, num_heads, head_dim, device=device, dtype=dtype, generator=gen) * 0.1

    # Match production shapes from vLLM's ``KimiDeltaAttention``:
    #   * ``g1_raw``  [bs, T, H*D]  = output of ``f_b_proj(f_a_proj(x))``
    #   * ``beta_raw``[bs, T, H]    = output of ``b_proj(x).float()`` (pre-sigmoid)
    #   * ``A_log``   [1, 1, H, 1]  (nn.Parameter)
    #   * ``dt_bias`` [H*D]         (nn.Parameter; per-head-per-dim, not per-head)
    g1_raw = torch.randn(
        bs, T, num_heads * head_dim, device=device, dtype=dtype, generator=gen
    ) * 0.1
    beta_raw = torch.randn(
        bs, T, num_heads, device=device, dtype=torch.float32, generator=gen
    ) * 0.5
    A_log = (
        torch.randn(num_heads, device=device, dtype=torch.float32, generator=gen)
        * -1.0
        - 2.0
    ).view(1, 1, num_heads, 1)
    dt_bias = (
        torch.randn(
            num_heads * head_dim,
            device=device,
            dtype=torch.float32,
            generator=gen,
        )
        * 0.01
    )

    state_shape = (bs, num_heads, head_dim, head_dim)
    state_init = torch.zeros(*state_shape, device=device, dtype=torch.float32)

    # --- FLA path (reference) ----
    # Mirrors what ``KimiDeltaAttention._forward`` does on the decode path
    # *before* this PR moved the gate+sigmoid inside ``_forward``.
    g_fla = fused_kda_gate(g1_raw, A_log, head_dim, g_bias=dt_bias)
    beta_fla = beta_raw.sigmoid()

    q_fla = q.reshape(1, bs * T, num_heads, head_dim)
    k_fla = k.reshape(1, bs * T, num_heads, head_dim)
    v_fla = v.reshape(1, bs * T, num_heads, head_dim)
    g_fla = g_fla.reshape(1, bs * T, num_heads, head_dim)
    beta_fla = beta_fla.reshape(1, bs * T, num_heads)

    cu_seqlens = torch.arange(
        0, bs * T + 1, T, dtype=torch.int32, device=device
    )
    ssm_state_indices = torch.arange(
        bs, dtype=torch.long, device=device
    ).unsqueeze(-1)

    state_fla = state_init.clone()
    out_fla, _ = fused_recurrent_kda(
        q=q_fla,
        k=k_fla,
        v=v_fla,
        g=g_fla,
        beta=beta_fla,
        initial_state=state_fla,
        inplace_final_state=True,
        use_qk_l2norm_in_kernel=True,
        cu_seqlens=cu_seqlens,
        ssm_state_indices=ssm_state_indices,
    )
    out_fla = out_fla.reshape(bs, T, num_heads, head_dim)

    # --- FlyDSL path ----
    state_fly = state_init.clone()
    out_fly = torch.empty(
        bs, T, num_heads, head_dim, device=device, dtype=dtype
    )
    indices = torch.arange(bs, device=device, dtype=torch.int32)
    flydsl_gdr_decode(
        query=q,
        key=k,
        value=v,
        a=g1_raw,
        b=beta_raw,
        dt_bias=dt_bias,
        A_log=A_log.view(-1),
        indices=indices,
        state=state_fly,
        out=out_fly,
        use_qk_l2norm=True,
        need_shuffle_state=False,
    )

    # Correctness check (bf16 recurrent accumulation tolerance).
    diff = (out_fla.float() - out_fly.float()).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    assert max_diff <= 1.5e-2, (
        f"FlyDSL KDA decode diverges from FLA: max_abs_diff={max_diff:.4e} "
        f"(> 1.5e-2), mean_abs_diff={mean_diff:.4e}. "
        f"shape: bs={bs}, num_heads={num_heads}, head_dim={head_dim}"
    )
