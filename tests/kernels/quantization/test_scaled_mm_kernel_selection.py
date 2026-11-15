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
    _POSSIBLE_FP8_BLOCK_KERNELS,
    _POSSIBLE_KERNELS,
    _POSSIBLE_NVFP4_KERNELS,
    AiterInt8ScaledMMLinearKernel,
    CPUInt8ScaledMMLinearKernel,
    Fp8BlockScaledMMLinearKernel,
    FP8ScaledMMLinearLayerConfig,
    HummingFP8ScaledMMLinearKernel,
    Int8ScaledMMLinearKernel,
    Int8ScaledMMLinearLayerConfig,
    MarlinFP8ScaledMMLinearKernel,
    ScaledMMLinearKernel,
    TritonInt8ScaledMMLinearKernel,
    _get_linear_backend,
    _prefer_humming_for_fp8_block,
    _resolve_backend_kernels,
    choose_scaled_mm_linear_kernel,
    init_fp8_linear_kernel,
    init_int8_linear_kernel,
    init_wfp8_a16_linear_kernel,
    register_linear_kernel,
)
from vllm.model_executor.layers.quantization.utils.humming import prioritize_humming
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    GroupShape,
    create_fp8_quant_key,
    kFp8Dynamic128Sym,
    kFp8DynamicTensorSym,
    kFp8Static128BlockSym,
    kFp8StaticTensorSym,
)
from vllm.platforms import PlatformEnum, current_platform
from vllm.platforms.interface import DeviceCapability

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


@pytest.mark.parametrize("strict", [False, True])
@patch.object(HummingFP8ScaledMMLinearKernel, "is_supported", return_value=(True, None))
@patch("vllm.model_executor.kernels.linear.current_platform")
def test_fp8_linear_backend_override(platform_mock, _, monkeypatch, strict):
    """Humming FP8 linear is weight-only, so it runs W8A8 layers as W8A16
    unless VLLM_STRICT_QUANT_SCHEME forbids that."""
    monkeypatch.setenv("VLLM_STRICT_QUANT_SCHEME", str(int(strict)))
    platform_mock._enum = PlatformEnum.CUDA
    config = VllmConfig(
        kernel_config=KernelConfig(
            linear_backend="cutlass",
            linear_backend_per_quant={"fp8_w8a8": "humming"},
        )
    )

    def init_kernel():
        return init_fp8_linear_kernel(
            activation_quant_key=kFp8StaticTensorSym,
            weight_quant_key=kFp8StaticTensorSym,
            input_dtype=torch.bfloat16,
            out_dtype=torch.bfloat16,
            weight_shape=(128, 128),
        )

    with set_current_vllm_config(config):
        if strict:
            with pytest.raises(ValueError, match="VLLM_STRICT_QUANT_SCHEME"):
                init_kernel()
        else:
            assert type(init_kernel()) is HummingFP8ScaledMMLinearKernel


@pytest.mark.parametrize(
    "group_shape,weight_shape,expected",
    [
        (GroupShape(128, 128), (512, 512), True),
        (GroupShape(128, 128), (576, 512), False),
        (GroupShape(128, 128), (512, 576), False),
        (GroupShape(1, 128), (576, 512), True),
        (GroupShape(1, 128), (512, 576), False),
        (GroupShape.PER_CHANNEL, (576, 576), True),
        (GroupShape.PER_TENSOR, (576, 576), True),
    ],
)
def test_humming_fp8_rejects_partial_scale_groups(group_shape, weight_shape, expected):
    """Humming sizes group and block scales by floor division, so it cannot load
    the scale a checkpoint stores for a partial trailing group or block."""
    config = FP8ScaledMMLinearLayerConfig(
        weight_quant_key=create_fp8_quant_key(static=True, group_shape=group_shape),
        activation_quant_key=kFp8DynamicTensorSym,
        weight_shape=weight_shape,
        input_dtype=torch.bfloat16,
        out_dtype=torch.bfloat16,
    )

    assert HummingFP8ScaledMMLinearKernel.can_implement(config)[0] == expected


@patch.object(MarlinFP8ScaledMMLinearKernel, "is_supported", return_value=(True, None))
@patch.object(HummingFP8ScaledMMLinearKernel, "is_supported", return_value=(True, None))
@patch("vllm.model_executor.kernels.linear.current_platform")
def test_wfp8_a16_falls_back_to_marlin_for_partial_blocks(platform_mock, _, __):
    platform_mock._enum = PlatformEnum.CUDA

    with set_current_vllm_config(VllmConfig()):
        kernel = init_wfp8_a16_linear_kernel(
            weight_quant_key=kFp8Static128BlockSym,
            activation_quant_key=kFp8DynamicTensorSym,
            weight_shape=(576, 512),
            input_dtype=torch.bfloat16,
            out_dtype=torch.bfloat16,
        )

    assert type(kernel) is MarlinFP8ScaledMMLinearKernel


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


@pytest.mark.skipif(
    not hasattr(torch.ops._C, "int8_scaled_mm_with_quant"),
    reason="int8_scaled_mm_with_quant is not built",
)
def test_cpu_sgl_int8_linear_accepts_3d_input():
    """The SGL int8 path must accept B x T x D activations (Whisper encoder)."""
    N, K = 64, 128
    layer = torch.nn.Module()
    layer.weight = torch.nn.Parameter(
        torch.randint(-127, 128, (N, K), dtype=torch.int8), requires_grad=False
    )
    layer.weight_scale = torch.nn.Parameter(
        torch.rand(N, 1) * 0.01 + 1e-3, requires_grad=False
    )
    layer.bias = torch.nn.Parameter(
        torch.randn(N, dtype=torch.bfloat16), requires_grad=False
    )
    config = Int8ScaledMMLinearLayerConfig(
        is_channelwise=True,
        is_static_input_scheme=False,
        input_symmetric=True,
    )
    kernel = CPUInt8ScaledMMLinearKernel(
        config,
        ("weight", "weight_scale", "input_scale", "input_zero_point", "azp_adj"),
    )
    kernel.process_weights_for_sgl(layer)
    kernel.linear_method = kernel._apply_weights_sgl

    x = torch.randn(2, 3, K, dtype=torch.bfloat16)
    out = kernel.apply_weights(layer, x, layer.bias)
    ref = kernel.apply_weights(layer, x.reshape(-1, K), layer.bias)

    assert out.shape == (2, 3, N)
    torch.testing.assert_close(out, ref.reshape(2, 3, N), rtol=0, atol=0)


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


class OOTFp8BlockScaledMMLinearKernel(Fp8BlockScaledMMLinearKernel):
    @classmethod
    def is_supported(
        cls, compute_capability: int | None = None
    ) -> tuple[bool, str | None]:
        return True, None

    @classmethod
    def can_implement(cls, c: FP8ScaledMMLinearLayerConfig) -> tuple[bool, str | None]:
        return True, None

    def apply_block_scaled_mm(
        self,
        A: torch.Tensor,
        B: torch.Tensor,
        As: torch.Tensor,
        Bs: torch.Tensor,
    ) -> torch.Tensor:
        pass


@patch("vllm.model_executor.kernels.linear.current_platform")
def test_register_oot_fp8_block_linear_kernel(platform_mock, default_vllm_config):
    """Test that block-scaled FP8 kernel registration works correctly."""
    platform_mock._enum = PlatformEnum.OOT
    register_linear_kernel(
        OOTFp8BlockScaledMMLinearKernel, PlatformEnum.OOT, "fp8_block"
    )

    kernel = init_fp8_linear_kernel(
        activation_quant_key=kFp8Dynamic128Sym,
        weight_quant_key=kFp8Static128BlockSym,
        input_dtype=torch.bfloat16,
        out_dtype=torch.bfloat16,
        weight_shape=(128, 128),
    )

    assert isinstance(kernel, OOTFp8BlockScaledMMLinearKernel), (
        "init_fp8_linear_kernel should return an instance of the registered kernel"
    )


def test_humming_priority_fp8_block_sm86():
    """On SM86, Humming moves directly ahead of Marlin for block-scaled FP8. No
    other kernel moves, and other devices, other FP8 schemes and a single forced
    backend are unchanged."""
    block = "fp8_block_w8a8"
    kernels = _POSSIBLE_FP8_BLOCK_KERNELS[PlatformEnum.CUDA]
    original = list(kernels)
    humming, marlin = HummingFP8ScaledMMLinearKernel, MarlinFP8ScaledMMLinearKernel

    for capability in (75, 80, 89, 90):
        assert _prefer_humming_for_fp8_block(kernels, block, capability) == original
    assert _prefer_humming_for_fp8_block(kernels, "fp8_w8a8", 86) == original
    for forced in ([marlin], [humming]):
        assert _prefer_humming_for_fp8_block(forced, block, 86) == forced

    reordered = _prefer_humming_for_fp8_block(kernels, block, 86)
    assert reordered[reordered.index(humming) + 1] is marlin
    assert [k for k in reordered if k is not humming] == [
        k for k in original if k is not humming
    ]
    assert kernels == original


@pytest.mark.parametrize(
    "capability,linear_backend,weight_shape,expected_kernel",
    [
        (DeviceCapability(8, 6), "auto", (128, 128), HummingFP8ScaledMMLinearKernel),
        (DeviceCapability(8, 6), "marlin", (128, 128), MarlinFP8ScaledMMLinearKernel),
        (DeviceCapability(8, 0), "auto", (128, 128), MarlinFP8ScaledMMLinearKernel),
        # Humming cannot load a partial scale block, so Marlin keeps the layer.
        (DeviceCapability(8, 6), "auto", (576, 512), MarlinFP8ScaledMMLinearKernel),
    ],
)
def test_choose_fp8_block_kernel_sm86(
    capability, linear_backend, weight_shape, expected_kernel, monkeypatch
):
    """FP8 layers pass no compute capability, so the current device decides."""
    monkeypatch.setenv("VLLM_STRICT_QUANT_SCHEME", "0")
    config = FP8ScaledMMLinearLayerConfig(
        weight_quant_key=kFp8Static128BlockSym,
        activation_quant_key=kFp8Dynamic128Sym,
        input_dtype=torch.bfloat16,
        out_dtype=torch.bfloat16,
        weight_shape=weight_shape,
    )
    vllm_config = VllmConfig(kernel_config=KernelConfig(linear_backend=linear_backend))
    kernels = [MarlinFP8ScaledMMLinearKernel, HummingFP8ScaledMMLinearKernel]

    with (
        patch.dict(_POSSIBLE_FP8_BLOCK_KERNELS, {current_platform._enum: kernels}),
        patch.object(current_platform, "is_cuda", return_value=True),
        patch.object(
            current_platform, "get_device_capability", return_value=capability
        ),
        patch.object(
            MarlinFP8ScaledMMLinearKernel, "is_supported", return_value=(True, None)
        ),
        patch.object(
            HummingFP8ScaledMMLinearKernel, "is_supported", return_value=(True, None)
        ),
        set_current_vllm_config(vllm_config),
    ):
        chosen = choose_scaled_mm_linear_kernel(
            config,
            _POSSIBLE_FP8_BLOCK_KERNELS,
            quantization="fp8_block_w8a8",
            activation_quant_key=kFp8Dynamic128Sym,
        )

    assert chosen is expected_kernel


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
