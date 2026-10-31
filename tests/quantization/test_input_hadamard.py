# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for `InputHadamardLinearMethod`, the format-agnostic block Hadamard
rotation of linear-layer inputs."""

import pytest
import torch

from vllm.config.model import ModelConfig
from vllm.model_executor.kernels.linear.mxfp4.aiter import AiterMxfp4LinearKernel
from vllm.model_executor.kernels.linear.mxfp4.base import MxFp4LinearLayerConfig
from vllm.model_executor.kernels.linear.mxfp4.emulation import (
    EmulationMxfp4LinearKernel,
)
from vllm.model_executor.layers.linear import LinearMethodBase
from vllm.model_executor.layers.quantization.input_hadamard import (
    InputHadamardLinearMethod,
    block_hadamard_rotate,
    hadamard_matrix,
)
from vllm.model_executor.layers.quantization.online.mxfp4 import (
    Mxfp4OnlineLinearMethod,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import kMxfp4Dynamic
from vllm.model_executor.model_loader.reload.layerwise import (
    finalize_layerwise_reload,
    initialize_layerwise_reload,
    record_metadata_for_reloading,
)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.utils import replace_parameter, set_weight_attrs
from vllm.platforms import current_platform
from vllm.utils.torch_utils import set_random_seed


@pytest.mark.parametrize("size", [16, 32, 64])
def test_block_hadamard_rotate_preserves_products(size: int):
    h = hadamard_matrix(size, torch.device("cpu"))
    torch.testing.assert_close(h @ h.T, torch.eye(size))
    assert torch.equal(h.abs(), torch.full_like(h, size**-0.5))

    x = torch.randn(8, 256, dtype=torch.float64)
    w = torch.randn(16, 256, dtype=torch.float64)
    torch.testing.assert_close(
        block_hadamard_rotate(x, h).double() @ block_hadamard_rotate(w, h).double().T,
        x @ w.T,
        rtol=1e-5,
        atol=1e-5,
    )


class _DenseOnlineMethod(LinearMethodBase):
    """Stand-in for an online method: plain fp32 GEMM that, like the online
    methods, flags the layer once it has processed the weight."""

    def create_weights(
        self,
        layer,
        input_size_per_partition,
        output_partition_sizes,
        input_size,
        output_size,
        params_dtype,
        **extra_weight_attrs,
    ):
        weight = torch.nn.Parameter(
            torch.zeros(sum(output_partition_sizes), input_size_per_partition),
            requires_grad=False,
        )
        set_weight_attrs(weight, extra_weight_attrs)
        layer.register_parameter("weight", weight)

    def process_weights_after_loading(self, layer):
        if getattr(layer, "_already_called_process_weights_after_loading", False):
            return
        layer.processed_weight = layer.weight.detach().clone()
        layer._already_called_process_weights_after_loading = True

    def apply(self, layer, x, bias=None):
        return x @ layer.weight.T


def _make_layer(
    rotate_weight: bool, input_size: int = 128, output_size: int = 16, size: int = 32
) -> tuple[torch.nn.Module, InputHadamardLinearMethod]:
    layer = torch.nn.Module()
    layer.weight_loader = "v1"
    method = InputHadamardLinearMethod(_DenseOnlineMethod(), size, rotate_weight)
    method.create_weights(
        layer,
        input_size,
        [output_size],
        input_size,
        output_size,
        torch.float32,
        weight_loader="v2",
    )
    return layer, method


def test_rotates_weight_once_and_input_on_apply():
    layer, method = _make_layer(rotate_weight=True)
    assert layer.weight.weight_loader == "v1", "wrapped v1 methods keep the v1 loader"
    assert not hasattr(layer, "input_rotation")
    set_random_seed(0)
    w = torch.randn(16, 128)
    x = torch.randn(4, 128)
    layer.weight.data.copy_(w)

    method.process_weights_after_loading(layer)
    method.process_weights_after_loading(layer)  # e.g. after layerwise loading

    h = hadamard_matrix(32, w.device)
    torch.testing.assert_close(layer.processed_weight, block_hadamard_rotate(w, h))
    torch.testing.assert_close(method.apply(layer, x), x @ w.T, rtol=1e-4, atol=1e-4)


def test_checkpoint_rotation_is_checked_and_dropped():
    layer, method = _make_layer(rotate_weight=False)
    set_random_seed(0)
    w = torch.randn(16, 128)
    x = torch.randn(4, 128)
    h = hadamard_matrix(32, w.device)
    layer.weight.data.copy_(block_hadamard_rotate(w, h))
    signs = (h * 32**0.5).round().to(torch.int8)
    layer.input_rotation.weight_loader(layer.input_rotation, signs)

    method.process_weights_after_loading(layer)

    assert not hasattr(layer, "input_rotation")
    torch.testing.assert_close(method.apply(layer, x), x @ w.T, rtol=1e-4, atol=1e-4)


def test_rejects_non_sylvester_and_missing_rotation():
    layer, method = _make_layer(rotate_weight=False)
    flipped = (hadamard_matrix(32, torch.device("cpu")) * 32**0.5).round()
    flipped[0] = -flipped[0]
    with pytest.raises(NotImplementedError, match="Sylvester"):
        layer.input_rotation.weight_loader(layer.input_rotation, flipped.to(torch.int8))
    with pytest.raises(ValueError, match="input_rotation"):
        method.process_weights_after_loading(layer)


@pytest.mark.parametrize("rotate_weight", [True, False])
def test_layerwise_reload(rotate_weight: bool, default_vllm_config):
    """Reloading reruns the weight rotation, or the input_rotation check and
    drop, on the new weights, and keeps the processed tensors in place."""
    default_vllm_config.model_config = ModelConfig()
    layer = torch.nn.Module()
    layer.weight_loader = default_weight_loader
    method = InputHadamardLinearMethod(_DenseOnlineMethod(), 32, rotate_weight)
    method.create_weights(
        layer, 128, [16], 128, 16, torch.float32, weight_loader=default_weight_loader
    )
    layer.quant_method = method
    model = torch.nn.Sequential(layer)
    record_metadata_for_reloading(model)
    h = hadamard_matrix(32, torch.device("cpu"))
    signs = (h * 32**0.5).round().to(torch.int8)

    def load(w: torch.Tensor) -> None:
        layer.weight.weight_loader(
            layer.weight, w if rotate_weight else block_hadamard_rotate(w, h)
        )
        if not rotate_weight:
            layer.input_rotation.weight_loader(layer.input_rotation, signs)

    set_random_seed(0)
    load(torch.randn(16, 128))
    method.process_weights_after_loading(layer)
    weight = layer.weight
    w = torch.randn(16, 128)
    x = torch.randn(4, 128)

    initialize_layerwise_reload(model)
    load(w)
    finalize_layerwise_reload(model, default_vllm_config.model_config)

    assert layer.weight is weight
    assert not hasattr(layer, "input_rotation")
    torch.testing.assert_close(layer.weight, block_hadamard_rotate(w, h))
    torch.testing.assert_close(method.apply(layer, x), x @ w.T, rtol=1e-4, atol=1e-4)


class _FusingMethod(_DenseOnlineMethod):
    """Takes the rotation over and records the input `apply` receives."""

    def fuse_input_hadamard(self, layer, size):
        return True

    def apply(self, layer, x, bias=None):
        layer.applied_input = x
        return super().apply(layer, x, bias)


def test_fused_rotation_is_skipped_by_the_wrapper():
    layer = torch.nn.Module()
    layer.weight_loader = "v1"
    method = InputHadamardLinearMethod(_FusingMethod(), 32, rotate_weight=True)
    method.create_weights(layer, 128, [16], 128, 16, torch.float32)
    set_random_seed(0)
    w = torch.randn(16, 128)
    x = torch.randn(4, 128)
    layer.weight.data.copy_(w)

    method.process_weights_after_loading(layer)
    method.apply(layer, x)

    assert layer.input_hadamard_fused
    assert not hasattr(layer, "input_hadamard")
    h = hadamard_matrix(32, w.device)
    torch.testing.assert_close(layer.processed_weight, block_hadamard_rotate(w, h))
    assert torch.equal(layer.applied_input, x), "the wrapper must not rotate twice"


def test_rejects_indivisible_input_size():
    with pytest.raises(ValueError, match="divisible"):
        _make_layer(rotate_weight=True, input_size=48)


def _online_mxfp4(
    weight: torch.Tensor,
    x: torch.Tensor,
    kernel_cls: type,
    hadamard_size: int | None,
) -> tuple[torch.nn.Module, torch.Tensor]:
    output_size, input_size = weight.shape
    inner = Mxfp4OnlineLinearMethod()
    inner.kernel = kernel_cls(
        MxFp4LinearLayerConfig(activation_quant_key=kMxfp4Dynamic)
    )
    if isinstance(inner.kernel, AiterMxfp4LinearKernel):
        # The ASM GEMM writes bf16 only; serving sets the default dtype to bf16.
        inner.kernel.out_dtype = torch.bfloat16
    method: LinearMethodBase = inner
    if hadamard_size is not None:
        method = InputHadamardLinearMethod(inner, hadamard_size, rotate_weight=True)
    layer = torch.nn.Module()
    layer.weight_loader = default_weight_loader
    method.create_weights(
        layer=layer,
        input_size_per_partition=input_size,
        output_partition_sizes=[output_size],
        input_size=input_size,
        output_size=output_size,
        params_dtype=torch.bfloat16,
        weight_loader=default_weight_loader,
    )
    replace_parameter(layer, "weight", weight.clone())
    method.process_weights_after_loading(layer)
    return layer, method.apply(layer, x)


@pytest.mark.skipif(
    not current_platform.supports_mx(), reason="MXFP4 kernels need MX support."
)
# AITER picks its ASM GEMM for (N, K) it has tuned configs for, such as
# (3072, 1536), and its Triton GEMM otherwise.
@pytest.mark.parametrize(
    "kernel_cls, input_size, output_size, fused",
    [
        (EmulationMxfp4LinearKernel, 2048, 256, False),
        (AiterMxfp4LinearKernel, 2048, 256, False),
        (AiterMxfp4LinearKernel, 1536, 3072, True),
    ],
    ids=["emulation", "aiter-triton", "aiter-asm"],
)
@pytest.mark.parametrize("num_tokens", [16, 256])
def test_online_mxfp4_hadamard(
    kernel_cls,
    input_size,
    output_size,
    fused,
    num_tokens,
    default_vllm_config,
    dist_init,
):
    """The rotated layer computes the same GEMM with less MXFP4 error, and
    matches the unrotated layer run on explicitly rotated operands. Only
    AITER's ASM path rotates inside its activation quant."""
    supported, reason = kernel_cls.is_supported()
    if not supported:
        pytest.skip(reason)
    default_vllm_config.model_config = ModelConfig()
    set_random_seed(0)
    with torch.device(current_platform.device_type):
        x = torch.randn(num_tokens, input_size, dtype=torch.bfloat16)
        x[:, torch.randperm(input_size)[:16]] *= 50.0
        weight = torch.randn(output_size, input_size, dtype=torch.bfloat16) * 0.02
        weight[:, torch.randperm(input_size)[:16]] *= 10.0
        want = x.float() @ weight.float().T

        _, plain = _online_mxfp4(weight, x, kernel_cls, None)
        layer, rotated = _online_mxfp4(weight, x, kernel_cls, 32)
        h = hadamard_matrix(32, x.device)
        _, explicit = _online_mxfp4(
            block_hadamard_rotate(weight, h),
            block_hadamard_rotate(x, h),
            kernel_cls,
            None,
        )

    assert layer.input_hadamard_fused == fused

    def rel_err(y: torch.Tensor) -> float:
        return ((y.float() - want).norm() / want.norm()).item()

    # How much the rotation helps depends on the shape and outlier layout:
    # 15% to 67% across these cases.
    assert rel_err(rotated) < 0.9 * rel_err(plain), (rel_err(rotated), rel_err(plain))
    assert ((rotated.float() - explicit.float()).norm() / want.norm()).item() < 1e-2
