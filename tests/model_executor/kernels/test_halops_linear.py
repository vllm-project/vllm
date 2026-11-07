# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""HALOps MXFP8 linear backend: registration, backend selection, shape gating,
weight preparation and the custom-op boundary.

The `halops` wheel is only published for Blackwell, so all but the
weight-preparation test stand in for the package in `sys.modules` and run on
CPU.
"""

import sys
import types

import pytest
import torch

from vllm.config import VllmConfig, set_current_vllm_config
from vllm.config.kernel import KernelConfig
from vllm.model_executor.kernels.linear import (
    _LINEAR_BACKEND_KERNEL_MAP,
    _POSSIBLE_MXFP8_KERNELS,
    HALOpsMxfp8LinearKernel,
    Mxfp8LinearLayerConfig,
    TorchMxFp8LinearKernel,
    _resolve_backend_kernels,
)
from vllm.model_executor.kernels.linear.mxfp8 import halops as halops_mod
from vllm.model_executor.layers.quantization.utils.mxfp8_utils import (
    MXFP8_BLOCK_SIZE,
    swizzle_mxfp8_scale,
)
from vllm.platforms import PlatformEnum, current_platform

pytestmark = pytest.mark.cpu_test


def test_halops_is_registered_last_in_the_cuda_mxfp8_priority_list() -> None:
    """`auto` keeps preferring every other CUDA MXFP8 kernel."""
    assert HALOpsMxfp8LinearKernel in _LINEAR_BACKEND_KERNEL_MAP["halops"]
    assert _POSSIBLE_MXFP8_KERNELS[PlatformEnum.CUDA][-1] is HALOpsMxfp8LinearKernel


def test_linear_backend_halops_selects_the_halops_kernel() -> None:
    """`--linear-backend halops` narrows MXFP8 selection to HALOps."""
    config = VllmConfig(kernel_config=KernelConfig(linear_backend="halops"))

    with set_current_vllm_config(config):
        selected = _resolve_backend_kernels(
            _POSSIBLE_MXFP8_KERNELS[PlatformEnum.CUDA],
            "MXFP8",
            quantization="mxfp8",
        )

    assert selected == [HALOpsMxfp8LinearKernel]


def test_linear_backend_halops_falls_back_for_other_layer_types() -> None:
    """A layer type HALOps does not implement keeps automatic selection."""
    config = VllmConfig(kernel_config=KernelConfig(linear_backend="halops"))

    with set_current_vllm_config(config):
        selected = _resolve_backend_kernels(
            [TorchMxFp8LinearKernel], "MXFP8", quantization="mxfp8"
        )

    assert selected == [TorchMxFp8LinearKernel]


@pytest.mark.parametrize(
    ("config", "reason"),
    [
        (
            Mxfp8LinearLayerConfig(weight_shape=(128, 128), bmm_batch_size=2),
            "HALOps dense_gemm does not implement grouped BMM.",
        ),
        (
            Mxfp8LinearLayerConfig(weight_shape=(128, 64)),
            "HALOps requires K divisible by 128, got 64.",
        ),
        (
            Mxfp8LinearLayerConfig(weight_shape=(64, 128)),
            "HALOps requires N divisible by 128, got 64.",
        ),
    ],
)
def test_halops_rejects_unsupported_layer_configs(config, reason) -> None:
    assert HALOpsMxfp8LinearKernel.can_implement(config) == (False, reason)


def test_halops_accepts_128_aligned_matmul() -> None:
    assert HALOpsMxfp8LinearKernel.can_implement(
        Mxfp8LinearLayerConfig(weight_shape=(256, 512))
    ) == (True, None)


def test_missing_halops_package_names_the_install_command(monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "halops", None)

    with pytest.raises(ImportError, match=r'uv pip install "vllm\[halops\]" -f'):
        halops_mod._import_halops()


requires_halops_sm100 = pytest.mark.skipif(
    not HALOpsMxfp8LinearKernel.is_supported()[0],
    reason="needs a Blackwell (SM100) GPU with the halops wheel installed",
)


def _make_layer(n: int, k: int) -> torch.nn.Module:
    layer = torch.nn.Module()
    weight = torch.randn(n, 2 * k).to(torch.float8_e4m3fn)[:, ::2]
    assert not weight.is_contiguous()
    layer.weight = torch.nn.Parameter(weight, requires_grad=False)
    layer.weight_scale = torch.nn.Parameter(
        torch.randint(0, 256, (n, k // MXFP8_BLOCK_SIZE), dtype=torch.uint8),
        requires_grad=False,
    )
    return layer


@requires_halops_sm100
def test_halops_weight_preparation_swizzles_scales_and_keeps_loaders() -> None:
    """Loaded weights must stay reloadable by the weight cache daemon."""
    n, k = 128, 256
    layer = _make_layer(n, k)
    expected_weight = layer.weight.detach().contiguous()
    expected_scale = swizzle_mxfp8_scale(
        layer.weight_scale.detach().clone(), M=n, K=k
    ).view(n, k // MXFP8_BLOCK_SIZE)
    weight_loader, scale_loader = object(), object()
    layer.weight.weight_loader = weight_loader
    layer.weight_scale.weight_loader = scale_loader
    kernel = HALOpsMxfp8LinearKernel(Mxfp8LinearLayerConfig(weight_shape=(n, k)))

    kernel.process_weights_after_loading(layer)

    assert layer.weight.is_contiguous()
    assert torch.equal(
        layer.weight.view(torch.uint8), expected_weight.view(torch.uint8)
    )
    assert layer.weight_scale.shape == (n, k // MXFP8_BLOCK_SIZE)
    assert layer.weight_scale.dtype == torch.uint8
    torch.testing.assert_close(layer.weight_scale, expected_scale)
    assert layer.weight.weight_loader is weight_loader
    assert layer.weight_scale.weight_loader is scale_loader
    assert layer.weight_block_size == [1, MXFP8_BLOCK_SIZE]


class _FakeHalops(types.ModuleType):
    """Stand-in for the external `halops` package; records the kernel's calls."""

    def __init__(self) -> None:
        super().__init__("halops")
        self.quant_calls: list[tuple[torch.Tensor, str, bool]] = []
        self.gemm_calls: list[dict] = []

    def group_wise_quant(self, x: torch.Tensor, *, mode: str, use_swizzle: bool):
        self.quant_calls.append((x, mode, use_swizzle))
        m, k = x.shape
        m_sf = (m + 127) // 128 * 128
        return (
            torch.empty((m, k), dtype=torch.float8_e4m3fn, device=x.device),
            torch.zeros((m_sf, k // 32), dtype=torch.uint8, device=x.device),
        )

    def dense_gemm(self, **kwargs) -> None:
        self.gemm_calls.append(kwargs)
        kwargs["output"].fill_(3.0)


@pytest.fixture
def fake_halops(monkeypatch) -> _FakeHalops:
    fake = _FakeHalops()
    monkeypatch.setitem(sys.modules, "halops", fake)
    return fake


def _weights(n: int = 128, k: int = 64):
    return (
        torch.empty((n, k), dtype=torch.float8_e4m3fn),
        torch.zeros((n, k // MXFP8_BLOCK_SIZE), dtype=torch.uint8),
    )


def test_halops_linear_quantizes_activations_and_passes_weight_scales(
    fake_halops,
) -> None:
    """The output keeps the real M; only the SfAtom scales are padded."""
    weight, weight_scale = _weights()
    x = torch.randn((2, 3, 64), dtype=torch.bfloat16)
    bias = torch.ones(128, dtype=torch.bfloat16)

    out = halops_mod._halops_mxfp8_linear(x, weight, weight_scale, bias)

    assert len(fake_halops.quant_calls) == 1
    (gemm,) = fake_halops.gemm_calls
    assert gemm["output"].shape == (6, 128)
    assert gemm["bias"] is bias
    assert gemm["sfb"] is weight_scale
    assert gemm["alpha"] is None
    assert gemm["sm_count"] == -1
    assert gemm["mode"] == "w8a8-mxfp8"
    assert out.shape == (2, 3, 128)
    assert out.dtype is torch.bfloat16
    torch.testing.assert_close(out, torch.full_like(out, 3.0))


def test_halops_linear_without_bias_passes_none(fake_halops) -> None:
    weight, weight_scale = _weights()

    halops_mod._halops_mxfp8_linear(
        torch.randn((1, 64), dtype=torch.bfloat16), weight, weight_scale, None
    )

    assert fake_halops.gemm_calls[0]["bias"] is None


def test_halops_linear_zero_tokens_skips_the_kernel(fake_halops) -> None:
    weight, weight_scale = _weights()

    out = halops_mod._halops_mxfp8_linear(
        torch.empty((0, 64), dtype=torch.bfloat16), weight, weight_scale, None
    )

    assert out.shape == (0, 128)
    assert fake_halops.quant_calls == []
    assert fake_halops.gemm_calls == []


def test_halops_linear_quantizes_a_contiguous_copy(fake_halops) -> None:
    weight, weight_scale = _weights()
    storage = torch.randn((4, 128), dtype=torch.bfloat16)
    x = storage[:, ::2]
    assert not x.is_contiguous()

    out = halops_mod._halops_mxfp8_linear(x, weight, weight_scale, None)

    q_input, mode, use_swizzle = fake_halops.quant_calls[0]
    assert q_input.is_contiguous()
    assert torch.equal(q_input, x)
    assert mode == "w8a8-mxfp8"
    assert use_swizzle
    assert out.dtype is torch.bfloat16


def test_halops_linear_upcasts_fp32_activations(fake_halops) -> None:
    weight, weight_scale = _weights()
    x = torch.randn((4, 64), dtype=torch.float32)

    out = halops_mod._halops_mxfp8_linear(x, weight, weight_scale, None)

    q_input = fake_halops.quant_calls[0][0]
    assert q_input.dtype is torch.bfloat16
    assert q_input.shape == (4, 64)
    assert out.dtype is torch.float32


def test_halops_custom_op_dispatches_to_the_kernel(fake_halops) -> None:
    """The op is registered on the platform dispatch key, not `Composite`."""
    if not current_platform.is_cuda():
        pytest.skip("the custom op is registered for the CUDA dispatch key")
    weight, weight_scale = (t.cuda() for t in _weights())
    x = torch.randn((4, 64), dtype=torch.bfloat16, device="cuda")

    out = torch.ops.vllm.halops_mxfp8_linear(x, weight, weight_scale, None)

    assert out.shape == (4, 128)
    assert len(fake_halops.gemm_calls) == 1


def test_halops_custom_op_fake_impl_matches_the_real_shape() -> None:
    weight, weight_scale = _weights()
    x = torch.empty((2, 3, 64), dtype=torch.bfloat16)

    out = halops_mod._halops_mxfp8_linear_fake(x, weight, weight_scale, None)

    assert out.shape == (2, 3, 128)
    assert out.dtype is torch.bfloat16
    assert out.device == x.device


def test_halops_custom_op_is_registered_with_a_fake_impl() -> None:
    from torch._subclasses.fake_tensor import FakeTensorMode

    schema = str(torch.ops.vllm.halops_mxfp8_linear.default._schema)
    assert "halops_mxfp8_linear" in schema
    assert "Tensor? bias" in schema

    with FakeTensorMode():
        weight, weight_scale = _weights()
        x = torch.empty((2, 3, 64), dtype=torch.bfloat16)
        out = torch.ops.vllm.halops_mxfp8_linear(x, weight, weight_scale, None)

    assert out.shape == (2, 3, 128)
    assert out.dtype is torch.bfloat16
