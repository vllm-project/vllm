# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Tests for INT8 (W8A8) fused-MoE oracle backend selection.

These exercise ``select_int8_moe_backend`` only (no kernels are launched), so
they run on any platform where the Triton INT8 MoE kernel is available — CUDA
(SM >= 7.5) or ROCm — not just gfx950.
"""

import pytest
import torch

from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEParallelConfig,
    RoutingMethodType,
)
from vllm.model_executor.layers.fused_moe.oracle.int8 import (
    Int8MoeBackend,
    select_int8_moe_backend,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    kInt8DynamicTensorSym,
    kInt8DynamicTokenSym,
    kInt8StaticChannelSym,
    kInt8StaticTensorSym,
)
from vllm.platforms import current_platform

# The Triton int8_w8a8 fused-MoE kernel is available on CUDA (Turing+) and on
# ROCm CDNA GPUs. Gate on that rather than on a specific arch.
INT8_MOE_SUPPORTED = (
    current_platform.is_cuda() and current_platform.has_device_capability((7, 5))
) or current_platform.is_rocm()

# XPU supports the per-token scheme only; see TritonExperts._supports_quant_scheme.
INT8_MOE_PER_TOKEN_SUPPORTED = INT8_MOE_SUPPORTED or current_platform.is_xpu()

requires_int8_moe = pytest.mark.skipif(
    not INT8_MOE_SUPPORTED,
    reason="Requires a GPU with Triton INT8 MoE support (CUDA SM>=7.5 or ROCm)",
)

requires_int8_moe_per_token = pytest.mark.skipif(
    not INT8_MOE_PER_TOKEN_SUPPORTED,
    reason="Requires Triton INT8 MoE support (CUDA SM>=7.5, ROCm, or XPU)",
)

DEVICE = current_platform.device_type


def _make_int8_moe_config(moe_backend: str = "auto") -> FusedMoEConfig:
    from vllm.model_executor.layers.fused_moe.activation import MoEActivation

    return FusedMoEConfig(
        num_experts=8,
        experts_per_token=2,
        hidden_dim=256,
        intermediate_size=256,
        num_local_experts=8,
        num_logical_experts=8,
        moe_parallel_config=FusedMoEParallelConfig.make_no_parallel(),
        activation=MoEActivation.SILU,
        in_dtype=torch.bfloat16,
        device=DEVICE,
        routing_method=RoutingMethodType.Renormalize,
        moe_backend=moe_backend,
    )


@requires_int8_moe
@pytest.mark.parametrize(
    "weight_key,activation_key",
    [
        # per-channel weight + dynamic per-token activation
        (kInt8StaticChannelSym, kInt8DynamicTokenSym),
        # per-tensor weight + dynamic per-tensor activation
        (kInt8StaticTensorSym, kInt8DynamicTensorSym),
    ],
)
def test_int8_dynamic_schemes_dispatch_to_triton(weight_key, activation_key):
    """Both dynamic-activation INT8 MoE schemes (per-channel + per-tensor
    weights) select the Triton backend."""
    config = _make_int8_moe_config()
    backend, experts_cls = select_int8_moe_backend(
        config, weight_key=weight_key, activation_key=activation_key
    )
    assert backend == Int8MoeBackend.TRITON
    assert experts_cls is not None


@requires_int8_moe_per_token
def test_int8_per_token_scheme_dispatches_to_triton_on_every_device():
    """The per-token scheme is the one INT8 MoE scheme XPU can also run.

    Its activation quantization is a Triton kernel (per_token_quant_int8),
    unlike the per-tensor schemes, which call ops.scaled_int8_quant -- a custom
    op with no XPU implementation. Kept separate from the parametrized test
    above so XPU exercises exactly the pair it supports.
    """
    config = _make_int8_moe_config()
    backend, experts_cls = select_int8_moe_backend(
        config,
        weight_key=kInt8StaticChannelSym,
        activation_key=kInt8DynamicTokenSym,
    )
    assert backend == Int8MoeBackend.TRITON
    assert experts_cls is not None


@pytest.mark.skipif(
    not current_platform.is_xpu(), reason="XPU-specific scheme restriction"
)
def test_int8_per_tensor_scheme_unsupported_on_xpu():
    """Pins the per-tensor exclusion on XPU: ops.scaled_int8_quant is missing
    there, so the oracle must not offer this scheme rather than fail later."""
    config = _make_int8_moe_config()
    with pytest.raises(NotImplementedError):
        select_int8_moe_backend(
            config,
            weight_key=kInt8StaticTensorSym,
            activation_key=kInt8DynamicTensorSym,
        )


@requires_int8_moe
def test_int8_explicit_moe_backend_triton():
    """An explicit --moe-backend triton selects the Triton INT8 backend."""
    config = _make_int8_moe_config(moe_backend="triton")
    backend, experts_cls = select_int8_moe_backend(
        config,
        weight_key=kInt8StaticChannelSym,
        activation_key=kInt8DynamicTokenSym,
    )
    assert backend == Int8MoeBackend.TRITON
    assert experts_cls is not None


@requires_int8_moe
def test_int8_unsupported_moe_backend_raises():
    """An unsupported --moe-backend for INT8 MoE raises a clear error."""
    config = _make_int8_moe_config(moe_backend="cutlass")
    with pytest.raises(ValueError, match="not supported for Int8 MoE"):
        select_int8_moe_backend(
            config,
            weight_key=kInt8StaticChannelSym,
            activation_key=kInt8DynamicTokenSym,
        )
