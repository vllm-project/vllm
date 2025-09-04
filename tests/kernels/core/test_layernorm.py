# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from tests.kernels.quant_utils import FP8_DTYPE
from tests.kernels.utils import create_strided_tensor, opcheck
from vllm.model_executor.layers.layernorm import PolyNorm, RMSNorm
from vllm.platforms import current_platform

# Tuple of (size, strides)
TENSOR_SIZES_STRIDES = [
    # Normal tensor layouts
    ((7, 8), (8, 1)),
    ((7, 768), (768, 1)),
    ((7, 769), (769, 1)),
    ((7, 5120), (5120, 1)),
    ((7, 5125), (5125, 1)),
    ((7, 5126), (5126, 1)),
    ((7, 8192), (8192, 1)),
    ((7, 8199), (8199, 1)),
    ((83, 8), (8, 1)),
    ((83, 768), (768, 1)),
    ((83, 769), (769, 1)),
    ((83, 5120), (5120, 1)),
    ((83, 5125), (5125, 1)),
    ((83, 5126), (5126, 1)),
    ((83, 8192), (8192, 1)),
    ((83, 8199), (8199, 1)),
    ((4096, 8), (8, 1)),
    ((4096, 768), (768, 1)),
    ((4096, 769), (769, 1)),
    ((4096, 5120), (5120, 1)),
    ((4096, 5125), (5125, 1)),
    ((4096, 5126), (5126, 1)),
    ((4096, 8192), (8192, 1)),
    ((4096, 8199), (8199, 1)),

    # Strided tensor layouts
    ((7, 8), (15, 1)),
    ((7, 768), (1536, 1)),
    ((7, 769), (769, 1)),
    ((7, 5120), (5121, 1)),
    ((7, 5125), (5128, 1)),
    ((7, 5126), (8192, 1)),
    ((7, 8192), (16384, 1)),
    ((7, 8199), (16789, 1)),
    ((83, 8), (23, 1)),
    ((83, 768), (2304, 1)),
    ((83, 769), (2555, 1)),
    ((83, 5120), (5122, 1)),
    ((83, 5125), (5127, 1)),
    ((83, 5126), (8194, 1)),
    ((83, 8192), (10000, 1)),
    ((83, 8199), (10001, 1)),
    ((4096, 8), (9, 1)),
    ((4096, 768), (769, 1)),
    ((4096, 769), (770, 1)),
    ((4096, 5120), (5122, 1)),
    ((4096, 5125), (5130, 1)),
    ((4096, 5126), (5130, 1)),
    ((4096, 8192), (9000, 1)),
    ((4096, 8199), (9001, 1)),

    # Multiple num_tokens dimensions
    ((7, 8, 8), (64, 8, 1)),
    ((7, 8, 8), (128, 16, 1)),
    ((12, 7, 768), (5607, 801, 1)),
    ((2, 3, 6, 769), (13860, 4620, 770, 1)),

    # Unsupported: last stride is not 1
    ((7, 8), (1, 8)),
    ((7, 768), (1, 8)),
    ((2, 3, 6, 769), (2, 3076, 9228, 4)),

    # Unsupported: multiple non-contiguous num_tokens dimensions
    ((7, 8, 8), (8, 64, 1)),
    ((7, 8, 8), (160, 16, 1)),
    ((12, 7, 768), (8010, 801, 1)),
    ((2, 3, 6, 769), (2310, 770, 9240, 1)),
]
# Whether to add residual to the input.
ADD_RESIDUAL = [False, True]
# Input and residual dtype.
DTYPES = [torch.half, torch.bfloat16, torch.float]
# Tuple of (quant_dtype, quant_scale)
QUANT_DTYPE_AND_SCALE = [
    (None, None),
    (FP8_DTYPE, 1.0),
    (FP8_DTYPE, 0.01),
    (FP8_DTYPE, 10.0),
]
SEEDS = [0]
CUDA_DEVICES = [
    f"cuda:{i}" for i in range(1 if torch.cuda.device_count() == 1 else 2)
]


@pytest.mark.parametrize("tensor_sizes_strides", TENSOR_SIZES_STRIDES)
@pytest.mark.parametrize("add_residual", ADD_RESIDUAL)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("quant_dtype_and_scale", QUANT_DTYPE_AND_SCALE)
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("device", CUDA_DEVICES)
@torch.inference_mode()
def test_rms_norm(
    tensor_sizes_strides: tuple[tuple[int, ...], tuple[int, ...]],
    add_residual: bool,
    dtype: torch.dtype,
    quant_dtype_and_scale: tuple[torch.dtype, float],
    seed: int,
    device: str,
) -> None:

    # Validate tensor_sizes_strides.
    tensor_size, tensor_stride = tensor_sizes_strides
    assert len(tensor_size) == len(tensor_stride), \
        f"Invalid tensor_sizes_strides: {tensor_sizes_strides}"
    hidden_size = tensor_size[-1]

    # Validate quant_dtype.
    quant_dtype, quant_scale = quant_dtype_and_scale
    assert quant_dtype is None or quant_dtype is FP8_DTYPE, \
        f"Invalid quant_dtype: {quant_dtype}"

    # Set up the inputs.
    current_platform.seed_everything(seed)
    torch.set_default_device(device)
    layer = RMSNorm(hidden_size).to(dtype=dtype)
    layer.weight.data.normal_(mean=1.0, std=0.1)
    weight = layer.weight.data
    scale = 1 / (2 * hidden_size)
    x = create_strided_tensor(tensor_size, tensor_stride, dtype).normal_()
    x *= scale
    residual = torch.randn_like(x).contiguous() * scale \
        if add_residual else None

    # NOTE(woosuk): The reference implementation should be executed first
    # because the custom kernel is in-place.
    ref_out = layer.forward_native(x, residual)
    if add_residual:
        ref_out, ref_residual = ref_out[0], ref_out[1]
    if quant_dtype is FP8_DTYPE:
        # static_scaled_fp8_quant only supports contiguous input.
        ref_out = ref_out.contiguous()
        ref_out_quant = torch.empty_like(ref_out, dtype=FP8_DTYPE)
        quant_scale_t = torch.tensor(quant_scale, dtype=torch.float32)
        torch.ops._C.static_scaled_fp8_quant(ref_out_quant, ref_out,
                                             quant_scale_t)
        ref_out = ref_out_quant

    # Check whether the custom kernel supports the given tensor layout.
    is_supported = True
    # If last stride is not 1, not supported.
    if tensor_stride[-1] != 1:
        is_supported = False
    # If multiple num_tokens dimensions with non-contiguous strides, not
    # supported.
    for i in range(len(tensor_size) - 2):
        if tensor_stride[i] != tensor_size[i + 1] * tensor_stride[i + 1]:
            is_supported = False
            break

    # Run the custom kernel and make sure it raises an error if the tensor
    # layout is not supported.
    try:
        if quant_dtype is FP8_DTYPE:
            out = torch.empty_like(x, dtype=FP8_DTYPE).contiguous()
            if add_residual:
                torch.ops._C.fused_add_rms_norm_static_fp8_quant(
                    out, x, residual, weight, quant_scale_t, 1e-6)
            else:
                torch.ops._C.rms_norm_static_fp8_quant(out, x, weight,
                                                       quant_scale_t, 1e-6)
        else:
            out = layer.forward_cuda(x, residual)
            if add_residual:
                out, residual = out[0], out[1]
    except RuntimeError as e:
        if not is_supported:
            return
        raise e

    if not is_supported:
        raise RuntimeError("Custom kernel does not support the given tensor "
                           "layout but did not raise an error.")

    # NOTE(woosuk): LayerNorm operators (including RMS) typically have larger
    # numerical errors than other operators because they involve reductions.
    # Therefore, we use a larger tolerance.
    rtol_out = 2e-1 if quant_dtype is FP8_DTYPE else 1e-2
    torch.testing.assert_close(out.to(dtype=torch.float32),
                               ref_out.to(dtype=torch.float32),
                               atol=1e-2,
                               rtol=rtol_out)
    if add_residual:
        torch.testing.assert_close(residual,
                                   ref_residual,
                                   atol=1e-2,
                                   rtol=1e-2)

    # Op checks
    if quant_dtype is FP8_DTYPE:
        if add_residual:
            opcheck(torch.ops._C.fused_add_rms_norm_static_fp8_quant,
                    (out, x, residual, weight, quant_scale_t,
                     layer.variance_epsilon))
        else:
            opcheck(torch.ops._C.rms_norm_static_fp8_quant,
                    (out, x, weight, quant_scale_t, layer.variance_epsilon))
    else:
        if add_residual:
            opcheck(torch.ops._C.fused_add_rms_norm,
                    (x, residual, weight, layer.variance_epsilon))
        else:
            opcheck(torch.ops._C.rms_norm,
                    (out, x, weight, layer.variance_epsilon))


@pytest.mark.parametrize("tensor_sizes_strides", TENSOR_SIZES_STRIDES)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("device", CUDA_DEVICES)
@torch.inference_mode()
def test_poly_norm(
    tensor_sizes_strides: tuple[tuple[int, ...], tuple[int, ...]],
    dtype: torch.dtype,
    seed: int,
    device: str,
) -> None:
    # Validate tensor_sizes_strides.
    tensor_size, tensor_stride = tensor_sizes_strides
    assert len(tensor_size) == len(tensor_stride), \
        f"Invalid tensor_sizes_strides: {tensor_sizes_strides}"
    hidden_size = tensor_size[-1]

    current_platform.seed_everything(seed)
    torch.set_default_device(device)
    layer = PolyNorm().to(dtype=dtype)
    layer.weight.data.normal_(mean=1.0, std=0.1)
    layer.bias.data.normal_(mean=1.0, std=0.1)
    scale = 1 / (2 * hidden_size)
    x = create_strided_tensor(tensor_size, tensor_stride, dtype).normal_()
    x *= scale

    ref_out = layer.forward_native(x)
    out = layer(x)
    torch.testing.assert_close(out, ref_out, atol=1e-2, rtol=1e-2)

    # PolyNorm does not support strided tensor layouts, so we make sure that it
    # raises an error.
    is_supported = x.is_contiguous()
    try:
        opcheck(torch.ops._C.poly_norm,
                (out, x, layer.weight.data, layer.bias.data,
                 layer.variance_epsilon))
    except Exception as e:
        if not is_supported:
            return
        raise e

    if not is_supported:
        raise RuntimeError("Custom kernel does not support the strided tensor "
                           "but did not raise an error.")
