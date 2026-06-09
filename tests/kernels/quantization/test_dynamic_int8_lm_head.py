# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for DynamicInt8LMHeadMethod (per-channel INT8 lm_head on ROCm)."""

import math

import pytest
import torch

from vllm.model_executor.kernels.linear import (
    MPLinearLayerConfig,
    choose_mp_linear_kernel,
)
from vllm.model_executor.layers.quantization.dynamic_int8_lm_head import (
    DynamicInt8LMHeadMethod,
)
from vllm.scalar_type import scalar_types


def _has_int8_kernel() -> bool:
    """Check if any kernel can handle signed int8 per-channel."""
    try:
        choose_mp_linear_kernel(
            MPLinearLayerConfig(
                full_weight_shape=(128, 256),
                partition_weight_shape=(128, 256),
                weight_type=scalar_types.int8,
                act_type=torch.float16,
                group_size=-1,
                zero_points=False,
                has_g_idx=False,
            )
        )
        return True
    except (ValueError, KeyError):
        return False


DTYPES = [torch.float16, torch.bfloat16]
# (M=vocab, K=hidden) — keep sizes small for fast tests,
# plus one representative real-world shape.
MK_SHAPES = [
    (256, 128),
    (1024, 512),
    (8192, 2560),
]
N_BATCH = [1, 4]
SEEDS = [0]


def _make_layer(M: int, K: int, dtype: torch.dtype) -> torch.nn.Module:
    """Create a minimal layer with a weight parameter, as create_weights does."""
    method = DynamicInt8LMHeadMethod()
    layer = torch.nn.Module()
    method.create_weights(
        layer,
        input_size_per_partition=K,
        output_partition_sizes=[M],
        input_size=K,
        output_size=M,
        params_dtype=dtype,
    )
    return method, layer


@pytest.mark.parametrize("n", N_BATCH)
@pytest.mark.parametrize("m,k", MK_SHAPES)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.skipif(not _has_int8_kernel(), reason="no int8 per-channel kernel")
@torch.inference_mode()
def test_dynamic_int8_lm_head_apply(n, m, k, dtype, seed, default_vllm_config):
    torch.manual_seed(seed)
    method, layer = _make_layer(m, k, dtype)

    # Fill with Xavier-scaled random data to keep magnitudes reasonable.
    xavier = math.sqrt(2 / k)
    layer.weight.data.copy_(
        (torch.rand(m, k, dtype=dtype, device="cpu") * 2 - 1) * xavier
    )
    w_orig = layer.weight.data.clone()

    # Move to GPU, then quantize.
    layer.cuda()
    method.process_weights_after_loading(layer)

    # Verify quantization happened.
    assert layer.weight.dtype == torch.int8, "weight should be INT8 after quantization"
    assert hasattr(layer, "weight_scale"), "weight_scale should be registered"
    assert method._w_orig.dtype == dtype, "_w_orig should keep original dtype"

    # Run apply (exercises wvSplitK_int8 for small N*K).
    x = (torch.rand(n, k, dtype=dtype, device="cuda") * 2 - 1) * xavier
    out = method.apply(layer, x)

    assert out.shape == (n, m)
    assert out.dtype == dtype

    # Reference: FP linear with original weights.
    ref = torch.nn.functional.linear(x, w_orig.to(device="cuda", dtype=dtype))

    # Error budget: each weight has up to ±0.5 * scale quantization error,
    # where scale ≈ xavier / 127 ≈ sqrt(2/K) / 127.  Accumulated over K
    # multiply-adds the error grows as ~scale * sqrt(K).  On top of that,
    # FP16/BF16 accumulation adds ~K * eps * |val| error.  The tolerance
    # below is an empirical upper bound that covers both sources across
    # all tested shapes and dtypes.
    atol = torch.finfo(dtype).eps * math.sqrt(k) * 128
    torch.testing.assert_close(out, ref, atol=atol, rtol=5e-2)


@pytest.mark.parametrize("m,k", MK_SHAPES)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.skipif(not _has_int8_kernel(), reason="no int8 per-channel kernel")
@torch.inference_mode()
def test_dynamic_int8_lm_head_embedding(m, k, dtype, seed, default_vllm_config):
    torch.manual_seed(seed)
    method, layer = _make_layer(m, k, dtype)

    xavier = math.sqrt(2 / k)
    layer.weight.data.copy_(
        (torch.rand(m, k, dtype=dtype, device="cpu") * 2 - 1) * xavier
    )
    w_orig = layer.weight.data.clone()

    layer.cuda()
    method.process_weights_after_loading(layer)

    # Embedding lookup should use original (lossless) weights.
    indices = torch.randint(0, m, (8,), device="cuda")
    emb = method.embedding(layer, indices)

    ref = torch.nn.functional.embedding(indices, w_orig.cuda())
    torch.testing.assert_close(emb, ref, atol=0, rtol=0)
