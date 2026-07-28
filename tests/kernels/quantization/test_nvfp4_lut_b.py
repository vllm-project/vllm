# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.config.kernel import KernelConfig
from vllm.model_executor.kernels.linear import _LINEAR_BACKEND_KERNEL_MAP
from vllm.model_executor.kernels.linear.nvfp4.base import NvFp4LinearLayerConfig
from vllm.model_executor.kernels.linear.nvfp4.lut_b import (
    LUT_B_PACKED_TILE_BYTES,
    LutBNvFp4LinearKernel,
    dequantize_lut_b,
    dequantize_nvfp4_weight,
    pack_lut_b_indices,
    quantize_lut_b,
    quantize_lut_b_calibration_free,
    unpack_lut_b_indices,
)
from vllm.model_executor.layers.quantization import modelopt
from vllm.model_executor.layers.quantization.compressed_tensors.schemes import (
    compressed_tensors_w4a4_nvfp4 as ct_nvfp4,
)
from vllm.model_executor.layers.quantization.quark.schemes import quark_nvfp4


def _make_nvfp4_layer(global_scales: torch.Tensor) -> torch.nn.Module:
    layer = torch.nn.Module()
    layer.weight = torch.nn.Parameter(
        torch.full((16, 32), 0x42, dtype=torch.uint8),
        requires_grad=False,
    )
    layer.weight_scale = torch.nn.Parameter(
        torch.ones((16, 4), dtype=torch.float8_e4m3fn),
        requires_grad=False,
    )
    layer.weight_global_scale = torch.nn.Parameter(
        global_scales,
        requires_grad=False,
    )
    layer.logical_widths = [8, 8]
    layer.output_size_per_partition = 16
    return layer


def test_lut_b_index_pack_round_trip() -> None:
    indices = torch.randint(0, 8, (7, 512), dtype=torch.uint8)

    packed = pack_lut_b_indices(indices)
    reconstructed = unpack_lut_b_indices(packed)

    assert packed.shape == (7, LUT_B_PACKED_TILE_BYTES)
    torch.testing.assert_close(reconstructed, indices)


def test_lut_b_tile_layout_has_3_125_bits_per_weight() -> None:
    weight = torch.empty(16, 128, dtype=torch.float32)
    expected_values = ((0.5, 1.0), (2.0, 4.0))
    for n_tile in range(2):
        for k_tile in range(2):
            weight[
                n_tile * 8 : (n_tile + 1) * 8,
                k_tile * 64 : (k_tile + 1) * 64,
            ] = expected_values[n_tile][k_tile]

    packed, codebooks = quantize_lut_b(weight)
    reconstructed = dequantize_lut_b(
        packed,
        codebooks,
        out_dtype=weight.dtype,
    )

    assert packed.shape == (2, 2, 192)
    assert codebooks.shape == (2, 2, 8)
    assert codebooks.dtype == torch.float8_e4m3fn
    assert packed.numel() + codebooks.numel() == 4 * 200
    torch.testing.assert_close(reconstructed, weight)


def test_lut_b_multistart_does_not_increase_reconstruction_error() -> None:
    torch.manual_seed(4)
    weight = torch.randn(8, 64, dtype=torch.float32)
    baseline_packed, baseline_codebooks = quantize_lut_b(weight)
    multistart_packed, multistart_codebooks, _, _, _ = quantize_lut_b_calibration_free(
        weight, algorithm="multistart"
    )

    baseline = dequantize_lut_b(
        baseline_packed,
        baseline_codebooks,
        out_dtype=torch.float32,
    )
    multistart = dequantize_lut_b(
        multistart_packed,
        multistart_codebooks,
        out_dtype=torch.float32,
    )
    assert torch.mean((multistart - weight).square()) <= torch.mean(
        (baseline - weight).square()
    )


def test_lut_b_sparse_residual_reduces_reconstruction_error() -> None:
    torch.manual_seed(5)
    weight = torch.randn(8, 64, dtype=torch.float32)
    packed, codebooks, scale, position, residual = quantize_lut_b_calibration_free(
        weight, algorithm="residual_1"
    )
    baseline = dequantize_lut_b(
        packed,
        codebooks,
        out_dtype=torch.float32,
        output_scale=scale,
    )
    corrected = dequantize_lut_b(
        packed,
        codebooks,
        out_dtype=torch.float32,
        output_scale=scale,
        residual_position=position,
        residual_value=residual,
    )

    assert position is not None and position.shape == (1, 1)
    assert residual is not None and residual.shape == (1, 1)
    assert torch.mean((corrected - weight).square()) < torch.mean(
        (baseline - weight).square()
    )


def test_lut_b_multiple_sparse_residuals() -> None:
    torch.manual_seed(6)
    weight = torch.randn(8, 64, dtype=torch.float32)
    packed, codebooks, scale, position, residual = quantize_lut_b_calibration_free(
        weight, algorithm="scaled_residual_4"
    )
    corrected = dequantize_lut_b(
        packed,
        codebooks,
        out_dtype=torch.float32,
        output_scale=scale,
        residual_position=position,
        residual_value=residual,
    )

    assert position is not None and position.shape == (1, 1, 4)
    assert residual is not None and residual.shape == (1, 1, 4)
    assert torch.isfinite(corrected).all()


def test_nvfp4_decode_preserves_fused_partition_scales() -> None:
    layer = _make_nvfp4_layer(torch.tensor([0.5, 0.25], dtype=torch.float32))

    weight = dequantize_nvfp4_weight(
        layer.weight,
        layer.weight_scale,
        layer.weight_global_scale,
        layer.logical_widths,
        out_dtype=torch.float32,
    )

    expected = torch.tensor([0.5, 1.0], dtype=torch.float32).repeat(16, 32)
    expected[8:] *= 0.5
    torch.testing.assert_close(weight, expected)


def test_lut_b_backend_repacks_and_uses_reference_linear() -> None:
    layer = _make_nvfp4_layer(torch.tensor([0.5, 0.25], dtype=torch.float32))
    source_weight = dequantize_nvfp4_weight(
        layer.weight,
        layer.weight_scale,
        layer.weight_global_scale,
        layer.logical_widths,
        out_dtype=torch.float32,
    )
    kernel = LutBNvFp4LinearKernel(NvFp4LinearLayerConfig())

    kernel.process_weights_after_loading(layer)

    assert layer.weight.shape == (2, 1, 192)
    assert layer.weight_codebook.shape == (2, 1, 8)
    assert not hasattr(layer, "weight_scale")
    assert not hasattr(layer, "weight_global_scale")
    reconstructed = dequantize_lut_b(
        layer.weight,
        layer.weight_codebook,
        out_dtype=torch.float32,
    )
    torch.testing.assert_close(reconstructed, source_weight)

    x = torch.randn(3, 64, dtype=torch.float32)
    bias = torch.randn(16, dtype=torch.float32)
    actual = kernel.apply_weights(layer, x, bias)
    expected = torch.nn.functional.linear(x, reconstructed, bias)
    torch.testing.assert_close(actual, expected)


def test_compressed_tensors_passes_distinct_scales_to_lut_b() -> None:
    layer = _make_nvfp4_layer(torch.tensor([2.0, 4.0], dtype=torch.float32))
    layer.weight_packed = layer.weight
    del layer.weight
    layer.input_global_scale = torch.nn.Parameter(
        torch.ones(2, dtype=torch.float32),
        requires_grad=False,
    )
    scheme = ct_nvfp4.CompressedTensorsW4A4Fp4.__new__(
        ct_nvfp4.CompressedTensorsW4A4Fp4
    )
    scheme.kernel = LutBNvFp4LinearKernel(NvFp4LinearLayerConfig())

    scheme.process_weights_after_loading(layer)

    reconstructed = dequantize_lut_b(
        layer.weight,
        layer.weight_codebook,
        out_dtype=torch.float32,
    )
    expected = torch.tensor([0.5, 1.0], dtype=torch.float32).repeat(16, 32)
    expected[8:] *= 0.5
    torch.testing.assert_close(reconstructed, expected)


@pytest.mark.parametrize(
    ("method_cls", "input_scale_name"),
    [
        (modelopt.ModelOptNvFp4LinearMethod, "input_scale"),
        (quark_nvfp4.QuarkNVFP4, "input_scale_2"),
    ],
)
def test_nvfp4_loaders_pass_distinct_scales_to_lut_b(
    method_cls: type,
    input_scale_name: str,
) -> None:
    layer = _make_nvfp4_layer(torch.tensor([0.5, 0.25], dtype=torch.float32))
    del layer.weight_global_scale
    layer.weight_scale_2 = torch.nn.Parameter(
        torch.tensor([0.5, 0.25], dtype=torch.float32),
        requires_grad=False,
    )
    setattr(
        layer,
        input_scale_name,
        torch.nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=False),
    )
    method = method_cls.__new__(method_cls)  # type: ignore[call-overload]
    method.kernel = LutBNvFp4LinearKernel(NvFp4LinearLayerConfig())

    method.process_weights_after_loading(layer)

    reconstructed = dequantize_lut_b(
        layer.weight,
        layer.weight_codebook,
        out_dtype=torch.float32,
    )
    expected = torch.tensor([0.5, 1.0], dtype=torch.float32).repeat(16, 32)
    expected[8:] *= 0.5
    torch.testing.assert_close(reconstructed, expected)


def test_lut_b_linear_backend_registration() -> None:
    assert _LINEAR_BACKEND_KERNEL_MAP["lut_b"] == {LutBNvFp4LinearKernel}
    assert KernelConfig(linear_backend="lut-b").linear_backend == "lut_b"
