# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for ScaledMM kernel selection logic (CPU-only).

Run `pytest tests/kernels/quantization/test_scaled_mm_kernel_selection.py`.
"""

import inspect
from abc import ABC
from unittest.mock import patch

import pytest
import torch

from vllm.config import KernelConfig, VllmConfig, set_current_vllm_config
from vllm.model_executor.kernels.linear import (
    _POSSIBLE_KERNELS,
    _POSSIBLE_NVFP4_KERNELS,
    AiterInt8ScaledMMLinearKernel,
    CPUInt8ScaledMMLinearKernel,
    HummingFP8ScaledMMLinearKernel,
    Int8ScaledMMLinearKernel,
    Int8ScaledMMLinearLayerConfig,
    ScaledMMLinearKernel,
    TritonInt8ScaledMMLinearKernel,
    _get_linear_backend,
    _resolve_backend_kernels,
    init_fp8_linear_kernel,
    init_int8_linear_kernel,
    register_linear_kernel,
)
from vllm.model_executor.layers.quantization.utils.humming import prioritize_humming
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    kFp8StaticTensorSym,
)
from vllm.platforms import PlatformEnum

pytestmark = pytest.mark.cpu_test


def test_auto_linear_backend_is_unchanged_without_overrides():
    kernels: list[type] = [object]
    quantizations = (
        "fp8_block_w8a8",
        "fp8_w8a8",
        "int8_w8a8",
        "mixed_precision",
        "mxfp8",
        "mxfp4",
        "mxfp6",
        "w8a16_fp8",
        "nvfp4_w4a4",
        "nvfp4_w4a16",
    )
    config = VllmConfig(kernel_config=KernelConfig(linear_backend="auto"))

    with set_current_vllm_config(config):
        for quantization in quantizations:
            assert _get_linear_backend(quantization=quantization) == "auto"
            assert (
                _resolve_backend_kernels(kernels, "test", quantization=quantization)
                is kernels
            )


def test_linear_backend_override_is_quantization_specific():
    config = VllmConfig(
        kernel_config=KernelConfig(
            linear_backend="cutlass",
            linear_backend_per_quant={"nvfp4_w4a16": "humming"},
        )
    )

    with set_current_vllm_config(config):
        assert _get_linear_backend(quantization="nvfp4_w4a16") == "humming"
        assert _get_linear_backend(quantization="nvfp4_w4a4") == "cutlass"
        assert _get_linear_backend(quantization="fp8_w8a8") == "cutlass"


@patch.object(HummingFP8ScaledMMLinearKernel, "is_supported", return_value=(True, None))
@patch("vllm.model_executor.kernels.linear.current_platform")
def test_fp8_linear_backend_override(platform_mock, _):
    platform_mock._enum = PlatformEnum.CUDA
    config = VllmConfig(
        kernel_config=KernelConfig(
            linear_backend="cutlass",
            linear_backend_per_quant={"fp8_w8a8": "humming"},
        )
    )

    with set_current_vllm_config(config):
        kernel = init_fp8_linear_kernel(
            activation_quant_key=kFp8StaticTensorSym,
            weight_quant_key=kFp8StaticTensorSym,
            input_dtype=torch.bfloat16,
            out_dtype=torch.bfloat16,
            weight_shape=(128, 128),
        )

    assert type(kernel) is HummingFP8ScaledMMLinearKernel


def test_is_supported_is_abstract():
    """Test that is_supported() is properly defined as abstract."""
    assert issubclass(ScaledMMLinearKernel, ABC)
    assert hasattr(ScaledMMLinearKernel, "is_supported")


def test_cpu_kernel_implements_is_supported():
    """Test that CPUInt8ScaledMMLinearKernel implements is_supported() method."""
    assert hasattr(CPUInt8ScaledMMLinearKernel, "is_supported"), (
        "CPUInt8ScaledMMLinearKernel missing is_supported() method"
    )
    # Verify it's a classmethod by checking if it can be called with the class
    # and by checking the method type
    assert inspect.ismethod(
        CPUInt8ScaledMMLinearKernel.is_supported
    ) or inspect.isfunction(CPUInt8ScaledMMLinearKernel.is_supported), (
        "CPUInt8ScaledMMLinearKernel.is_supported() should be a classmethod"
    )
    # Verify it can be called as a classmethod
    result, reason = CPUInt8ScaledMMLinearKernel.is_supported()
    assert isinstance(result, bool), "is_supported() should return a bool"
    assert reason is None or isinstance(reason, str), "reason should be str or None"


def test_aiter_kernel_implements_is_supported():
    """Test that AiterInt8ScaledMMLinearKernel implements is_supported() method."""
    assert hasattr(AiterInt8ScaledMMLinearKernel, "is_supported"), (
        "AiterInt8ScaledMMLinearKernel missing is_supported() method"
    )
    # Verify it's a classmethod by checking if it can be called with the class
    # and by checking the method type
    assert inspect.ismethod(
        AiterInt8ScaledMMLinearKernel.is_supported
    ) or inspect.isfunction(AiterInt8ScaledMMLinearKernel.is_supported), (
        "AiterInt8ScaledMMLinearKernel.is_supported() should be a classmethod"
    )
    # Verify it can be called as a classmethod
    # (will return False on CPU, which is expected)
    result, reason = AiterInt8ScaledMMLinearKernel.is_supported()
    assert isinstance(result, bool), "is_supported() should return a bool"
    assert reason is None or isinstance(reason, str), "reason should be str or None"
    # On CPU, it should return False with a reason about requiring ROCm
    # This validates the method works correctly even on non-ROCm platforms


@pytest.mark.parametrize(
    ("is_static_input_scheme", "is_channelwise", "is_fused_module", "expected"),
    [
        # per-tensor activations x per-tensor weights
        (True, False, False, False),
        # per-tensor activations x per-channel weights
        (True, False, True, False),
        (True, True, False, False),
        (True, True, True, False),
        # per-token activations x per-tensor weights
        (False, False, False, False),
        # per-token activations x per-channel weights
        (False, False, True, True),
        (False, True, False, True),
        (False, True, True, True),
    ],
)
def test_aiter_int8_kernel_can_implement_scale_layouts(
    is_static_input_scheme, is_channelwise, is_fused_module, expected
):
    config = Int8ScaledMMLinearLayerConfig(
        is_static_input_scheme=is_static_input_scheme,
        is_channelwise=is_channelwise,
        input_symmetric=True,
        is_fused_module=is_fused_module,
    )
    can_impl, _ = AiterInt8ScaledMMLinearKernel.can_implement(config)
    assert can_impl is expected


def _make_compressed_tensors_per_tensor_int8_scheme():
    from compressed_tensors.quantization import QuantizationStrategy

    from vllm.model_executor.layers.quantization.compressed_tensors.schemes import (
        CompressedTensorsW8A8Int8,
    )

    return CompressedTensorsW8A8Int8(
        strategy=QuantizationStrategy.TENSOR,
        is_static_input_scheme=False,
        input_symmetric=True,
    )


def _make_quark_per_tensor_int8_scheme():
    from vllm.model_executor.layers.quantization.quark.schemes.quark_w8a8_int8 import (
        QuarkW8A8Int8,
    )
    from vllm.model_executor.layers.quantization.utils.quant_utils import (
        kInt8DynamicTokenSym,
        kInt8StaticTensorSym,
    )

    return QuarkW8A8Int8(kInt8StaticTensorSym, kInt8DynamicTokenSym)


@pytest.mark.parametrize(
    "make_scheme",
    [
        _make_compressed_tensors_per_tensor_int8_scheme,
        _make_quark_per_tensor_int8_scheme,
    ],
    ids=["compressed_tensors", "quark"],
)
@pytest.mark.parametrize("output_partition_sizes", [[16, 32], [48]], ids=str)
@patch.object(TritonInt8ScaledMMLinearKernel, "is_supported", return_value=(True, None))
@patch.object(AiterInt8ScaledMMLinearKernel, "is_supported", return_value=(True, None))
@patch(
    "vllm.model_executor.parameter.get_tensor_model_parallel_world_size", return_value=1
)
@patch("vllm.model_executor.parameter.get_tensor_model_parallel_rank", return_value=0)
@patch("vllm.model_executor.kernels.linear.current_platform")
def test_aiter_int8_per_tensor_weight_selection_and_loading(
    platform_mock,
    _tp_rank,
    _tp_size,
    _aiter_supported,
    _triton_supported,
    output_partition_sizes,
    make_scheme,
):
    """Per-tensor weights x per-token activations through the scheme's own
    create_weights/process_weights_after_loading: AITER is selected only for
    fused modules, whose shard scales must become per-channel after loading."""
    platform_mock._enum = PlatformEnum.ROCM
    platform_mock.get_device_capability.return_value = None
    scheme = make_scheme()
    layer = torch.nn.Module()
    scheme.create_weights(
        layer,
        output_partition_sizes=output_partition_sizes,
        input_size_per_partition=32,
        params_dtype=torch.bfloat16,
        weight_loader=lambda *args, **kwargs: None,
    )

    is_fused = len(output_partition_sizes) > 1
    expected_kernel = (
        AiterInt8ScaledMMLinearKernel if is_fused else TritonInt8ScaledMMLinearKernel
    )
    assert type(scheme.kernel) is expected_kernel
    if not is_fused:
        return

    n = sum(output_partition_sizes)
    shard_scales = torch.tensor([0.5, 2.0])
    layer.weight.data.copy_(torch.ones(n, 32, dtype=torch.int8))
    layer.weight_scale.data.copy_(shard_scales)
    scheme.process_weights_after_loading(layer)

    # AiterInt8ScaledMMLinearKernel.apply_weights expects [K, N] weights and
    # one scale per output channel.
    assert layer.weight.shape == (32, n)
    assert layer.weight_scale.numel() == layer.weight.shape[1]
    expected = torch.repeat_interleave(
        shard_scales, torch.tensor(output_partition_sizes)
    )
    assert torch.equal(layer.weight_scale.flatten(), expected)


def test_cpu_kernel_accepts_all_configs():
    """Test that CPUInt8ScaledMMLinearKernel accepts all config combinations."""
    configs = [
        Int8ScaledMMLinearLayerConfig(
            is_channelwise=False,
            is_static_input_scheme=True,
            input_symmetric=True,
        ),
        Int8ScaledMMLinearLayerConfig(
            is_channelwise=True,
            is_static_input_scheme=False,
            input_symmetric=False,
        ),
    ]

    for config in configs:
        can_impl, reason = CPUInt8ScaledMMLinearKernel.can_implement(config)
        assert can_impl, (
            f"CPUInt8ScaledMMLinearKernel should accept config {config}: {reason}"
        )


class OOTInt8ScaledMMLinearKernel(Int8ScaledMMLinearKernel):
    @classmethod
    def is_supported(
        cls, compute_capability: int | None = None
    ) -> tuple[bool, str | None]:
        return True, None

    @classmethod
    def can_implement(cls, c: Int8ScaledMMLinearLayerConfig) -> tuple[bool, str | None]:
        return True, None

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        pass

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        pass


@patch("vllm.model_executor.kernels.linear.current_platform")
def test_register_oot_linear_kernel(platform_mock):
    """Test that the linear kernel registration works correctly."""
    platform_mock._enum = PlatformEnum.OOT
    register_linear_kernel(OOTInt8ScaledMMLinearKernel, PlatformEnum.OOT, "int8")

    kernel = init_int8_linear_kernel(True, True, True, "module")

    assert isinstance(kernel, OOTInt8ScaledMMLinearKernel), (
        "init_int8_linear_kernel should return an instance of the registered kernel"
    )


@pytest.mark.parametrize(
    "possible_kernels", [_POSSIBLE_KERNELS, _POSSIBLE_NVFP4_KERNELS]
)
def test_humming_priority_only_reorders_humming_and_marlin(possible_kernels):
    """Preferring Humming must not demote Marlin below unrelated kernels."""
    kernels = possible_kernels[PlatformEnum.CUDA]
    original = list(kernels)

    assert prioritize_humming(kernels, compute_capability=80) == original

    names = [k.__name__ for k in prioritize_humming(kernels, compute_capability=90)]
    humming = next(i for i, name in enumerate(names) if "Humming" in name)
    assert "Marlin" in names[humming + 1]
    del names[humming]
    assert names == [k.__name__ for k in original if "Humming" not in k.__name__]
    assert kernels == original
