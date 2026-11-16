# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for INT8 (W8A8) fused-MoE oracle backend selection.

These exercise backend selection and Quark configuration without launching
MoE kernels. Component selection runs wherever Triton INT8 MoE is available;
it does not imply that the platform's model entry point supports Quark.
"""

from dataclasses import replace

import pytest
import torch

from vllm.config import ParallelConfig, VllmConfig
from vllm.config.kernel import KernelConfig, MoEBackend
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEParallelConfig,
    RoutingMethodType,
)
from vllm.model_executor.layers.fused_moe.oracle.int8 import (
    Int8MoeBackend,
    select_int8_moe_backend,
)
from vllm.model_executor.layers.quantization.quark.quark_moe import (
    QuarkW8A8Int8MoEMethod,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    kInt8DynamicTensorSym,
    kInt8DynamicTokenSym,
    kInt8StaticChannelSym,
    kInt8StaticTensorSym,
)
from vllm.platforms import current_platform
from vllm.utils.import_utils import has_humming

# The Triton int8_w8a8 fused-MoE kernel is available on CUDA (Turing+), on
# ROCm CDNA GPUs and on XPU. Gate on that rather than on a specific arch.
INT8_MOE_SUPPORTED = (
    (current_platform.is_cuda() and current_platform.has_device_capability((7, 5)))
    or current_platform.is_rocm()
    or current_platform.is_xpu()
)

requires_int8_moe = pytest.mark.skipif(
    not INT8_MOE_SUPPORTED,
    reason="Requires Triton INT8 MoE support (CUDA SM>=7.5, ROCm, or XPU)",
)

# So FusedMoEConfig.device names the device the test actually runs on.
DEVICE = current_platform.device_type


def _make_int8_moe_config(
    moe_backend: MoEBackend = "auto", *, is_lora_enabled: bool = False
) -> FusedMoEConfig:
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
        is_lora_enabled=is_lora_enabled,
    )


@requires_int8_moe
@pytest.mark.parametrize(
    "weight_key,activation_key",
    [
        pytest.param(
            kInt8StaticChannelSym,
            kInt8DynamicTokenSym,
            id="per_channel_weight-per_token_act",
        ),
        pytest.param(
            kInt8StaticTensorSym,
            kInt8DynamicTensorSym,
            id="per_tensor_weight-per_tensor_act",
        ),
        pytest.param(
            kInt8StaticChannelSym,
            kInt8StaticTensorSym,
            id="per_channel_weight-static_tensor_act",
        ),
        pytest.param(
            kInt8StaticTensorSym,
            kInt8StaticTensorSym,
            id="per_tensor_weight-static_tensor_act",
        ),
    ],
)
def test_int8_schemes_dispatch_to_triton(weight_key, activation_key):
    """Supported dynamic and static INT8 schemes select Triton."""
    config = _make_int8_moe_config()
    backend, experts_cls = select_int8_moe_backend(
        config, weight_key=weight_key, activation_key=activation_key
    )
    assert backend == Int8MoeBackend.TRITON
    assert experts_cls is not None


@pytest.mark.skipif(not current_platform.is_xpu(), reason="XPU-only behaviour")
def test_scaled_int8_quant_is_available_on_xpu():
    """The per-tensor scheme's activation quantization works on XPU.

    `_int8_quantize` sends per-tensor activations through
    `ops.scaled_int8_quant`, which has an XPU branch. This is what lets the
    per-tensor scheme be enabled here alongside CUDA and ROCm; pin it so a
    future change cannot silently remove that branch.
    """
    from vllm import _custom_ops as ops

    x = torch.randn(4, 16, device=DEVICE, dtype=torch.bfloat16)
    scale = torch.full((1,), 0.05, device=DEVICE, dtype=torch.float32)

    q, returned_scale, azp = ops.scaled_int8_quant(x, scale=scale)

    assert q.dtype == torch.int8
    assert q.shape == x.shape
    assert azp is None
    torch.testing.assert_close(returned_scale, scale)


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


@requires_int8_moe
@pytest.mark.parametrize("weight_key", [kInt8StaticTensorSym, kInt8StaticChannelSym])
@pytest.mark.parametrize("moe_backend", ["auto", "triton"])
def test_static_int8_triton_rejects_lora(weight_key, moe_backend):
    """Quark cannot gain static LoRA support by moving to an internal kernel."""
    config = _make_int8_moe_config(moe_backend, is_lora_enabled=True)
    if moe_backend == "auto":
        error, reason = NotImplementedError, "No Int8 MoE backend supports"
    else:
        error, reason = ValueError, "static INT8 activation quantization with LoRA"
    with pytest.raises(error, match=reason):
        QuarkW8A8Int8MoEMethod(config, weight_key, kInt8StaticTensorSym)


@requires_int8_moe
@pytest.mark.parametrize("moe_backend", ["auto", "triton"])
@pytest.mark.parametrize(
    "weight_key,activation_key",
    [
        (kInt8StaticTensorSym, kInt8DynamicTensorSym),
        (kInt8StaticChannelSym, kInt8DynamicTokenSym),
    ],
)
def test_dynamic_int8_lora_is_not_rejected_by_static_guard(
    moe_backend, weight_key, activation_key
):
    """Check construction support, not nonzero LoRA numerical correctness."""
    from vllm.model_executor.layers.fused_moe.experts.triton_moe import TritonExperts

    config = _make_int8_moe_config(moe_backend, is_lora_enabled=True)
    method = QuarkW8A8Int8MoEMethod(config, weight_key, activation_key)
    assert method.int8_backend == Int8MoeBackend.TRITON
    assert method.experts_cls is TritonExperts
    assert method.experts_cls.supports_lora()
    assert method.activation_quant_key == activation_key


@pytest.mark.skipif(
    not (
        current_platform.is_cuda()
        and current_platform.has_device_capability((7, 5))
        and has_humming()
    ),
    reason="Requires Humming on CUDA SM>=7.5",
)
def test_quark_static_int8_rejects_activation_quantization_fallback(monkeypatch):
    """A weight-only backend must not discard checkpoint activation scales."""
    monkeypatch.delenv("VLLM_HUMMING_INPUT_QUANT_CONFIG", raising=False)
    monkeypatch.setenv("VLLM_HUMMING_MOE_GEMM_TYPE", "indexed")
    monkeypatch.setenv("VLLM_STRICT_QUANT_SCHEME", "0")
    config = _make_int8_moe_config(moe_backend="humming")
    backend, experts_cls = select_int8_moe_backend(
        config, kInt8StaticChannelSym, kInt8StaticTensorSym
    )
    assert backend == Int8MoeBackend.HUMMING
    assert (
        experts_cls.executed_activation_key(kInt8StaticChannelSym, kInt8StaticTensorSym)
        is None
    )
    with pytest.raises(NotImplementedError, match="preserves static INT8"):
        QuarkW8A8Int8MoEMethod(config, kInt8StaticChannelSym, kInt8StaticTensorSym)


@requires_int8_moe
@pytest.mark.parametrize("weight_key", [kInt8StaticTensorSym, kInt8StaticChannelSym])
@pytest.mark.parametrize(
    "all2all_backend,batched",
    [
        ("deepep_high_throughput", False),
        ("deepep_v2", False),
        ("flashinfer_all2allv", False),
        ("deepep_low_latency", True),
        ("nixl_ep", True),
        ("alltoall_batched", True),
    ],
)
def test_quark_static_int8_dispatch_uses_backend_capabilities(
    weight_key, all2all_backend, batched
):
    """Selection follows activation layout, not a static-INT8 backend list."""
    from vllm.model_executor.layers.fused_moe.experts.triton_moe import TritonExperts

    config = _make_int8_moe_config(moe_backend="triton")
    config = replace(
        config,
        num_local_experts=4,
        moe_parallel_config=replace(
            config.moe_parallel_config,
            dp_size=2,
            ep_size=2,
            use_ep=True,
            all2all_backend=all2all_backend,
        ),
    )
    assert config.activation_format.is_batched is batched
    if batched:
        with pytest.raises(ValueError, match="activation format"):
            QuarkW8A8Int8MoEMethod(config, weight_key, kInt8StaticTensorSym)
    else:
        method = QuarkW8A8Int8MoEMethod(config, weight_key, kInt8StaticTensorSym)
        assert method.int8_backend == Int8MoeBackend.TRITON
        assert method.experts_cls is TritonExperts
        assert method.moe is config
        assert config.moe_parallel_config.all2all_backend == all2all_backend


@pytest.mark.parametrize("moe_backend", ["auto", "triton"])
@pytest.mark.parametrize("dp_size", [1, 2])
def test_passthrough_requires_a_backend_that_owns_communication(moe_backend, dp_size):
    """The normal config entry rejects passthrough before Quark selection."""
    with pytest.raises(ValueError, match="passthrough.*only valid"):
        VllmConfig(
            parallel_config=ParallelConfig(
                data_parallel_size=dp_size,
                enable_expert_parallel=True,
                all2all_backend="passthrough",
            ),
            kernel_config=KernelConfig(moe_backend=moe_backend),
        )


@requires_int8_moe
@pytest.mark.parametrize("weight_key", [kInt8StaticTensorSym, kInt8StaticChannelSym])
@pytest.mark.parametrize("dp_size", [1, 2])
def test_quark_static_int8_supported_dispatch_preserves_scale_layout(
    weight_key, dp_size
):
    from vllm.model_executor.layers.fused_moe.experts.triton_moe import TritonExperts

    config = _make_int8_moe_config()
    config.moe_parallel_config = replace(
        config.moe_parallel_config,
        dp_size=dp_size,
        ep_size=dp_size,
        use_ep=dp_size > 1,
        all2all_backend="allgather_reducescatter",
    )
    method = QuarkW8A8Int8MoEMethod(config, weight_key, kInt8StaticTensorSym)
    layer = torch.nn.Module()
    method.create_weights(layer, 8, 256, 256, torch.bfloat16)
    quant_config = method.get_fused_moe_quant_config(layer)

    assert method.int8_backend == Int8MoeBackend.TRITON
    assert method.experts_cls is TritonExperts
    assert not method.is_monolithic
    assert quant_config is not None
    assert quant_config.per_act_token_quant is False
    assert quant_config.per_out_ch_quant is (weight_key == kInt8StaticChannelSym)
