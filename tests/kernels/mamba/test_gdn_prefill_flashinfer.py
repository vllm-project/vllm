# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import math

import pytest
import torch
import torch.nn.functional as F

from vllm.platforms import current_platform

if not (current_platform.is_cuda() and current_platform.is_device_capability(90)):
    pytest.skip(
        reason="FlashInfer GDN prefill kernel requires CUDA SM90 (Hopper).",
        allow_module_level=True,
    )

from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import (  # noqa: E402
    fi_chunk_gated_delta_rule,
)

NUM_K_HEADS = 4
NUM_V_HEADS = 8
HEAD_K_DIM = 128
HEAD_V_DIM = 128
CHUNK_SIZE = 64  # matches FLA_CHUNK_SIZE used by vLLM's own kernel warmup


def _build_inputs(qkv_dtype: torch.dtype, seed: int = 0):
    """Build small dummy GDN prefill inputs, mirroring the shapes vLLM's own
    `_warmup_prefill_kernels` constructs for a single T=64 chunk."""
    rng = torch.Generator("cuda").manual_seed(seed)
    T = CHUNK_SIZE

    q = torch.randn(
        1, T, NUM_K_HEADS, HEAD_K_DIM, device="cuda", dtype=qkv_dtype, generator=rng
    )
    k = torch.randn_like(q, generator=rng)
    v = torch.randn(
        1, T, NUM_V_HEADS, HEAD_V_DIM, device="cuda", dtype=qkv_dtype, generator=rng
    )

    # Synthetic gate/beta init, mirroring upstream FLA GatedDeltaNet:
    # https://github.com/fla-org/flash-linear-attention/blob/main/fla/layers/gated_deltanet.py
    a = torch.randn(
        1, T, NUM_V_HEADS, device="cuda", dtype=torch.float32, generator=rng
    )
    b = torch.randn(
        1, T, NUM_V_HEADS, device="cuda", dtype=torch.float32, generator=rng
    )
    A = torch.empty(NUM_V_HEADS, device="cuda", dtype=torch.float32).uniform_(
        0, 16, generator=rng
    )
    A_log = torch.log(A)
    dt = torch.exp(
        torch.rand(NUM_V_HEADS, device="cuda", dtype=torch.float32, generator=rng)
        * (math.log(0.1) - math.log(0.001))
        + math.log(0.001)
    )
    dt = torch.clamp(dt, min=1e-4)
    dt_bias = dt + torch.log(-torch.expm1(-dt))
    g = -A_log.exp().view(1, 1, NUM_V_HEADS) * F.softplus(
        a + dt_bias.view(1, 1, NUM_V_HEADS)
    )
    beta = torch.sigmoid(b)

    # state/g/beta are always fp32 in production (fi_chunk_gated_delta_rule
    # upcasts them internally) -- only q/k/v vary with the model's dtype.
    initial_state = torch.zeros(
        1, NUM_V_HEADS, HEAD_V_DIM, HEAD_K_DIM, device="cuda", dtype=torch.float32
    )
    cu_seqlens = torch.tensor([0, T], device="cuda", dtype=torch.int32)

    return q, k, v, g, beta, initial_state, cu_seqlens


@pytest.mark.parametrize("qkv_dtype", [torch.bfloat16, torch.float16, torch.float32])
def test_gdn_chunk_flashinfer_qkv_dtype_cast(qkv_dtype: torch.dtype):
    """fi_chunk_gated_delta_rule must not raise for any of these q/k/v
    dtypes -- fp32 previously crashed every GDN layer's warmup."""
    q, k, v, g, beta, initial_state, cu_seqlens = _build_inputs(qkv_dtype)

    output, final_state = fi_chunk_gated_delta_rule(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        initial_state=initial_state,
        output_final_state=True,
        cu_seqlens=cu_seqlens,
        use_qk_l2norm_in_kernel=True,
    )

    assert output.shape == (1, CHUNK_SIZE, NUM_V_HEADS, HEAD_V_DIM)
    assert not torch.isnan(output).any()
    assert not torch.isinf(output).any()
    assert final_state is not None
