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
    Fp8BlockScaledMMLinearKernel,
    FP8ScaledMMLinearLayerConfig,
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
    kFp8Dynamic128Sym,
    kFp8Static128BlockSym,
    kFp8StaticTensorSym,
)
from vllm.platforms import PlatformEnum
from vllm.utils.deep_gemm import is_deep_gemm_fp8_linear_requested

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


@pytest.mark.parametrize(
    "scheme_kind",
    ["fbgemm_fp8", "modelopt_pc_pt", "online_reload"],
    ids=["fbgemm", "modelopt", "online_reload"],
)
@pytest.mark.parametrize(
    "target_kernel_name",
    [
        "AiterHipbMMPerTokenFp8ScaledMMLinearKernel",
        "AiterPreshuffledPerTokenFp8ScaledMMLinearKernel",
        "AiterPerTokenFp8ScaledMMLinearKernel",
    ],
    ids=["hipb_mm", "preshuffled", "per_token"],
)
def test_fbgemm_and_modelopt_fp8_select_aiter_ptpc_kernels_and_canonicalize_scales(
    scheme_kind: str,
    target_kernel_name: str,
    default_vllm_config,
    monkeypatch: pytest.MonkeyPatch,
):
    """FBGEMMFp8, ModelOpt PcPt, and Online fp8_per_channel direct reload must
    emit kFp8StaticChannelSym so all three AITER PTPC kernels select cleanly,
    canonicalize 1-D (N,) and 2-D scales, preserve 3-D activation shapes, and
    keep addresses across direct reloads.
    """
    from types import SimpleNamespace

    import vllm._aiter_ops as aiter_ops_mod
    from tests.single_rank_group import SingleRankGroup, override_parallel_groups
    from vllm._aiter_ops import rocm_aiter_ops
    from vllm.config import ModelConfig
    from vllm.config.quantization import QuantizationConfigArgs
    from vllm.model_executor.kernels.linear.scaled_mm import aiter as aiter_mod
    from vllm.model_executor.layers.fusion.quant_activation import get_input_quant_key
    from vllm.model_executor.layers.linear import ReplicatedLinear
    from vllm.model_executor.layers.quantization import fbgemm_fp8 as fbgemm_mod
    from vllm.model_executor.layers.quantization import modelopt as modelopt_mod
    from vllm.model_executor.layers.quantization.online.base import (
        OnlineQuantizationConfig,
    )
    from vllm.model_executor.layers.quantization.utils.quant_utils import (
        kFp8DynamicTokenSym,
        kFp8StaticChannelSym,
    )
    from vllm.model_executor.model_loader.reload.direct import (
        begin_direct_reload,
        finish_direct_reload,
        record_checkpoint_layout,
    )
    from vllm.model_executor.model_loader.utils import process_weights_after_loading
    from vllm.platforms import current_platform
    from vllm.utils.torch_utils import set_default_torch_dtype

    aiter_kernels = (
        aiter_mod.AiterHipbMMPerTokenFp8ScaledMMLinearKernel,
        aiter_mod.AiterPreshuffledPerTokenFp8ScaledMMLinearKernel,
        aiter_mod.AiterPerTokenFp8ScaledMMLinearKernel,
    )
    target_kernel = getattr(aiter_mod, target_kernel_name)

    monkeypatch.setattr(current_platform, "_enum", PlatformEnum.ROCM)
    monkeypatch.setattr(current_platform, "is_rocm", lambda: True)
    monkeypatch.setattr(current_platform, "is_fp8_fnuz", lambda: False)
    monkeypatch.setattr(current_platform, "fp8_dtype", lambda: torch.float8_e4m3fn)
    monkeypatch.setattr(
        current_platform, "has_device_capability", lambda cap, device_id=0: True
    )
    monkeypatch.setattr(
        current_platform, "get_device_capability", lambda device_id=0: (9, 4)
    )
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda _d=None: SimpleNamespace(gcnArchName="gfx942"),
    )
    monkeypatch.setattr(aiter_ops_mod, "is_aiter_found_and_supported", lambda: True)

    monkeypatch.setenv("VLLM_ROCM_USE_AITER", "1")
    monkeypatch.setenv("VLLM_ROCM_USE_AITER_LINEAR", "1")
    monkeypatch.setenv(
        "VLLM_ROCM_USE_AITER_LINEAR_HIPBMM",
        "1" if target_kernel_name.startswith("AiterHipbMM") else "0",
    )
    rocm_aiter_ops.refresh_env_variables()
    try:
        for cls in aiter_kernels:
            supported = cls is target_kernel
            monkeypatch.setattr(
                cls,
                "is_supported",
                classmethod(
                    lambda _c, cc=None, ok=supported: (
                        ok,
                        None if ok else "disabled in test",
                    )
                ),
            )
        monkeypatch.setattr(
            rocm_aiter_ops,
            "is_shuffled_per_token_w8a8_gemm_tuned",
            lambda n, k, dt: (
                target_kernel
                is aiter_mod.AiterPreshuffledPerTokenFp8ScaledMMLinearKernel
            ),
        )
        monkeypatch.setattr(
            rocm_aiter_ops,
            "is_per_token_w8a8_gemm_tuned",
            lambda n, k, dt: (
                target_kernel is aiter_mod.AiterPerTokenFp8ScaledMMLinearKernel
            ),
        )

        def _fallback_shuffle(tensor: torch.Tensor, layout=(16, 16)):
            r, c = tensor.shape
            br, bc = layout
            return (
                tensor.reshape(r // br, br, c // bc, bc)
                .permute(0, 2, 1, 3)
                .contiguous()
                .view_as(tensor)
            )

        monkeypatch.setattr(rocm_aiter_ops, "shuffle_weight", _fallback_shuffle)

        default_vllm_config.model_config = ModelConfig(dtype="bfloat16")
        if scheme_kind == "fbgemm_fp8":
            quant_config = fbgemm_mod.FBGEMMFp8Config(
                ignore_list=[], input_scale_ub=1200.0
            )
        elif scheme_kind == "modelopt_pc_pt":
            quant_config = modelopt_mod.ModelOptFp8Config(
                quant_method="FP8_PER_CHANNEL_PER_TOKEN",
                is_checkpoint_fp8_serialized=True,
                kv_cache_quant_method=None,
                exclude_modules=[],
            )
        else:
            assert scheme_kind == "online_reload"
            monkeypatch.setenv("VLLM_RELOAD_MODE", "direct")
            quant_config = OnlineQuantizationConfig(
                QuantizationConfigArgs(linear="fp8_per_channel")
            )

        def _build_and_load(seed: int):
            n_out, k_in = 64, 32
            with (
                override_parallel_groups(SingleRankGroup()),
                set_default_torch_dtype(torch.bfloat16),
            ):
                layer = ReplicatedLinear(
                    k_in,
                    n_out,
                    bias=False,
                    params_dtype=torch.bfloat16,
                    quant_config=quant_config,
                    prefix="proj",
                )
            method = layer.quant_method
            if scheme_kind == "fbgemm_fp8":
                assert isinstance(method, fbgemm_mod.FBGEMMFp8LinearMethod)
                selected = method.fp8_linear
                assert selected.config.weight_quant_key == kFp8StaticChannelSym
                assert layer.input_scale_ub.item() == 1200.0
                assert get_input_quant_key(layer) is None
                scale_shape = (n_out, 1)
            else:
                assert isinstance(method, modelopt_mod.ModelOptLinearMethod)
                assert method.spec.weight == kFp8StaticChannelSym
                selected = method.kernel
                assert get_input_quant_key(layer) == kFp8DynamicTokenSym
                scale_shape = (n_out,)
                if scheme_kind == "online_reload":
                    assert getattr(layer, "quantizes_on_load", False) is True
            assert type(selected) is target_kernel

            record_checkpoint_layout(layer)
            torch.manual_seed(seed)
            if scheme_kind == "online_reload":
                w_bf16 = torch.randn(n_out, k_in, dtype=torch.bfloat16) * 0.5 + (
                    seed * 0.25
                )
                layer.weight.weight_loader(layer.weight, w_bf16)
                assert layer.weight_scale.shape == (n_out,)
                w_ckpt = layer.weight.detach().clone()
                ws_ckpt = layer.weight_scale.detach().clone()
            else:
                w_ckpt = (torch.randn(n_out, k_in) * 0.5).to(torch.float8_e4m3fn)
                ws_ckpt = torch.linspace(0.1, 1.0, n_out, dtype=torch.float32).reshape(
                    scale_shape
                ) + (seed * 0.25)
                layer.weight.weight_loader(layer.weight, w_ckpt)
                layer.weight_scale.weight_loader(layer.weight_scale, ws_ckpt)
            process_weights_after_loading(
                layer, default_vllm_config.model_config, torch.device("cpu")
            )
            return layer, w_ckpt, ws_ckpt

        layer, _, ws_cold = _build_and_load(seed=0)
        expected_scale_shape = (
            (1, 64)
            if target_kernel is aiter_mod.AiterHipbMMPerTokenFp8ScaledMMLinearKernel
            else (64, 1)
        )
        assert layer.weight_scale.shape == expected_scale_shape
        assert layer.weight_scale.is_contiguous()
        assert torch.equal(layer.weight_scale.flatten(), ws_cold.flatten())
        if target_kernel is aiter_mod.AiterHipbMMPerTokenFp8ScaledMMLinearKernel:
            assert layer.weight.shape == (32, 64)
            assert not layer.weight.is_contiguous()

        kernel = (
            layer.quant_method.fp8_linear
            if scheme_kind == "fbgemm_fp8"
            else layer.quant_method.kernel
        )
        seen_bs_shapes: list[tuple[int, ...]] = []
        monkeypatch.setattr(
            kernel,
            "quant_fp8",
            lambda x_2d, _xs, _ub: (
                x_2d.to(torch.float8_e4m3fn),
                torch.ones((x_2d.shape[0], 1), dtype=torch.float32),
            ),
        )

        def _make_fake_gemm(n_dim: int):
            def _gemm(a, b, _as, bs, _bias, dt):
                seen_bs_shapes.append(tuple(bs.shape))
                return torch.zeros((a.shape[0], b.shape[n_dim]), dtype=dt)

            return _gemm

        monkeypatch.setattr(rocm_aiter_ops, "hipb_mm_fp8", _make_fake_gemm(1))
        for op_name in ("preshuffled_per_token_w8a8_gemm", "w8a8_gemm"):
            monkeypatch.setattr(rocm_aiter_ops, op_name, _make_fake_gemm(0))
        x_3d = torch.ones((2, 4, 32), dtype=torch.bfloat16)
        out_3d, _ = layer(x_3d)
        out_apply_3d = layer.quant_method.apply(layer, x_3d)
        assert out_3d.shape == (2, 4, 64)
        assert out_apply_3d.shape == (2, 4, 64)
        assert seen_bs_shapes == [expected_scale_shape, expected_scale_shape]

        cold_w_ptr = layer.weight.data_ptr()
        cold_ws_ptr = layer.weight_scale.data_ptr()

        begin_direct_reload(layer)
        torch.manual_seed(1)
        if scheme_kind == "online_reload":
            w_bf16_new = torch.randn(64, 32, dtype=torch.bfloat16) * 0.5 + 0.25
            layer.weight.weight_loader(layer.weight, w_bf16_new)
        else:
            w_new = (torch.randn(64, 32) * 0.5).to(torch.float8_e4m3fn)
            ws_new = (
                torch.linspace(0.1, 1.0, 64, dtype=torch.float32).reshape(ws_cold.shape)
                + 0.25
            )
            layer.weight.weight_loader(layer.weight, w_new)
            layer.weight_scale.weight_loader(layer.weight_scale, ws_new)
        finish_direct_reload(
            layer, default_vllm_config.model_config, torch.device("cpu")
        )

        fresh_layer, _, _ = _build_and_load(seed=1)
        assert layer.weight.data_ptr() == cold_w_ptr
        assert layer.weight_scale.data_ptr() == cold_ws_ptr
        assert layer.weight_scale.shape == expected_scale_shape
        assert layer.weight_scale.is_contiguous()
        assert torch.equal(
            layer.weight.contiguous().view(torch.uint8),
            fresh_layer.weight.contiguous().view(torch.uint8),
        )
        assert torch.equal(layer.weight_scale, fresh_layer.weight_scale)
    finally:
        monkeypatch.undo()
        rocm_aiter_ops.refresh_env_variables()


def test_fbgemm_and_modelopt_fp8_cross_backend_can_implement(
    monkeypatch: pytest.MonkeyPatch,
):
    """kFp8StaticChannelSym x kFp8DynamicTokenSym keeps expected can_implement
    contracts across all FP8 backends.
    """
    from dataclasses import replace

    from vllm.model_executor.kernels.linear.scaled_mm import xpu as xpu_mod
    from vllm.model_executor.kernels.linear.scaled_mm.cpu import (
        CPUFp8ChannelWiseScaledMMLinearKernel,
    )
    from vllm.model_executor.kernels.linear.scaled_mm.cutlass import (
        CutlassFP8ScaledMMLinearKernel,
    )
    from vllm.model_executor.kernels.linear.scaled_mm.flashinfer import (
        FlashInferFP8ScaledMMLinearKernel,
    )
    from vllm.model_executor.kernels.linear.scaled_mm.pytorch import (
        ChannelWiseTorchFP8ScaledMMLinearKernel,
        RowWiseTorchFP8ScaledMMLinearKernel,
    )
    from vllm.model_executor.kernels.linear.scaled_mm.triton import (
        TritonFP8ScaledMMLinearKernel,
    )
    from vllm.model_executor.kernels.linear.scaled_mm.xpu import (
        XPUW8A8FP8LinearKernel,
        XPUW8A16FP8LinearKernel,
    )
    from vllm.model_executor.layers.quantization.utils.quant_utils import (
        kFp8DynamicTokenSym,
        kFp8StaticChannelSym,
    )

    cfg_bf16 = FP8ScaledMMLinearLayerConfig(
        weight_quant_key=kFp8StaticChannelSym,
        activation_quant_key=kFp8DynamicTokenSym,
        weight_shape=(64, 64),
        input_dtype=torch.bfloat16,
        out_dtype=torch.bfloat16,
    )
    cfg_fp16 = replace(cfg_bf16, out_dtype=torch.float16)
    w_key_xpu = replace(kFp8StaticChannelSym, dtype=torch.float8_e4m3fn)
    a_key_xpu = replace(kFp8DynamicTokenSym, dtype=torch.float8_e4m3fn)
    cfg_xpu = replace(
        cfg_bf16,
        weight_quant_key=w_key_xpu,
        activation_quant_key=a_key_xpu,
    )
    monkeypatch.setattr(xpu_mod, "kFp8StaticChannelSym", w_key_xpu)
    monkeypatch.setattr(
        XPUW8A8FP8LinearKernel,
        "_SUPPORTED_WEIGHT_QUANT_KEYS",
        {
            w_key_xpu,
            replace(xpu_mod.kFp8StaticTensorSym, dtype=torch.float8_e4m3fn),
        },
    )
    monkeypatch.setattr(
        XPUW8A8FP8LinearKernel,
        "_SUPPORTED_ACT_QUANT_KEYS",
        {
            a_key_xpu,
            replace(xpu_mod.kFp8DynamicTensorSym, dtype=torch.float8_e4m3fn),
        },
    )

    assert CutlassFP8ScaledMMLinearKernel.can_implement(cfg_bf16)[0]
    assert not FlashInferFP8ScaledMMLinearKernel.can_implement(cfg_bf16)[0]
    assert TritonFP8ScaledMMLinearKernel.can_implement(cfg_bf16)[0]
    assert RowWiseTorchFP8ScaledMMLinearKernel.can_implement(cfg_bf16)[0]
    assert not RowWiseTorchFP8ScaledMMLinearKernel.can_implement(cfg_fp16)[0]
    assert ChannelWiseTorchFP8ScaledMMLinearKernel.can_implement(cfg_fp16)[0]
    assert CPUFp8ChannelWiseScaledMMLinearKernel.can_implement(cfg_bf16)[0]
    assert XPUW8A8FP8LinearKernel.can_implement(cfg_xpu)[0]
    assert XPUW8A16FP8LinearKernel.can_implement(cfg_xpu)[0]
    assert HummingFP8ScaledMMLinearKernel.can_implement(cfg_bf16)[0]
    assert cfg_bf16.weight_quant_key.scale.group_shape.is_per_channel()


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


@pytest.mark.parametrize(
    "kernel_config,requested",
    [
        (KernelConfig(), False),
        (KernelConfig(linear_backend="cutlass"), False),
        (KernelConfig(linear_backend="deep_gemm"), True),
        (
            KernelConfig(linear_backend_per_quant={"fp8_block_w8a8": "deep_gemm"}),
            True,
        ),
        (KernelConfig(linear_backend_per_quant={"fp8_w8a8": "deep_gemm"}), False),
        (
            KernelConfig(
                linear_backend="deep_gemm",
                linear_backend_per_quant={"fp8_block_w8a8": "cutlass"},
            ),
            False,
        ),
    ],
)
def test_deep_gemm_fp8_linear_requested(kernel_config, requested):
    assert is_deep_gemm_fp8_linear_requested(kernel_config) is requested
