# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

import vllm._custom_ops as ops
from tests.kernels.quant_utils import (
    FP8_DTYPE,
    ref_dynamic_per_tensor_fp8_quant,
    ref_dynamic_per_token_quant,
)
from tests.kernels.utils import opcheck
from vllm.model_executor.layers.quantization.utils.fp8_utils import (
    dequant_fp8_weight,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    GroupShape,
    scaled_quantize,
)
from vllm.platforms import current_platform
from vllm.utils.torch_utils import set_random_seed

DTYPES = [torch.bfloat16, torch.float]
HIDDEN_SIZES = [17, 1024, 1025, 1026, 5137, 8193]
NUM_TOKENS = [1, 7, 4096]
SCALE_UBS = [True, False]
SEEDS = [0]


def opcheck_fp8_quant(
    output,
    input,
    scale=None,
    scale_ub=None,
    use_per_token_if_dynamic=False,
    group_shape=None,
):
    if scale is not None:
        opcheck(
            torch.ops._C.static_scaled_fp8_quant,
            (output, input, scale, group_shape),
        )
    elif use_per_token_if_dynamic:
        scale = torch.empty(
            (input.shape[0], 1), device=input.device, dtype=torch.float32
        )
        opcheck(
            torch.ops._C.dynamic_per_token_scaled_fp8_quant,
            (output, input, scale, scale_ub),
        )
    else:
        scale = torch.empty(
            (input.numel() // input.shape[-1], 1),
            device=input.device,
            dtype=torch.float32,
        )
        opcheck(torch.ops._C.dynamic_scaled_fp8_quant, (output, input, scale))


@pytest.mark.parametrize("num_tokens", NUM_TOKENS)
@pytest.mark.parametrize("hidden_size", HIDDEN_SIZES)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("do_scale_ub", SCALE_UBS)
@pytest.mark.parametrize("seed", SEEDS)
@torch.inference_mode()
def test_dynamic_per_token_fp8_quant(
    num_tokens: int, hidden_size: int, dtype: torch.dtype, do_scale_ub: bool, seed: int
) -> None:
    set_random_seed(seed)

    x = (
        torch.rand(num_tokens, hidden_size, dtype=dtype, device="cuda") + 1e-6
    )  # avoid nans

    scale_ub = (
        torch.mean(x).to(dtype=torch.float32, device="cuda") if do_scale_ub else None
    )
    ref_out, ref_scales = ref_dynamic_per_token_quant(x, FP8_DTYPE, scale_ub)
    ops_out, ops_scales = ops.scaled_fp8_quant(
        x, scale_ub=scale_ub, use_per_token_if_dynamic=True
    )

    torch.testing.assert_close(ref_scales, ops_scales)
    torch.testing.assert_close(
        ref_out.to(dtype=torch.float32), ops_out.to(dtype=torch.float32)
    )

    opcheck_fp8_quant(ops_out, x, None, scale_ub, use_per_token_if_dynamic=True)


@pytest.mark.parametrize("num_tokens", NUM_TOKENS)
@pytest.mark.parametrize("hidden_size", HIDDEN_SIZES)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("seed", SEEDS)
@torch.inference_mode()
def test_dynamic_per_tensor_fp8_quant(
    num_tokens: int, hidden_size: int, dtype: torch.dtype, seed: int
) -> None:
    set_random_seed(seed)

    x = torch.rand(num_tokens, hidden_size, dtype=dtype, device="cuda")

    ref_out, ref_scale = ref_dynamic_per_tensor_fp8_quant(x)
    ops_out, ops_scale = ops.scaled_fp8_quant(x)

    torch.testing.assert_close(ref_scale, ops_scale)
    torch.testing.assert_close(
        ref_out.to(dtype=torch.float32), ops_out.to(dtype=torch.float32)
    )

    opcheck_fp8_quant(ops_out, x)


@pytest.mark.parametrize("pad", [8, 16])
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("seed", SEEDS)
@torch.inference_mode()
def test_dynamic_per_tensor_fp8_quant_strided_input(
    pad: int, dtype: torch.dtype, seed: int
) -> None:
    """Non-contiguous rows take the strided abs-max kernel; contiguous rows take
    the flat one. Both must produce exactly the same scale and bytes.

    pad=8 (16 B for bf16) leaves rows misaligned for 32 B vector loads, pad=16
    keeps them aligned but non-contiguous.
    """
    set_random_seed(seed)
    num_tokens, hidden_size = 64, 4096

    full = torch.rand(num_tokens, hidden_size + pad, dtype=dtype, device="cuda")
    x = full[:, :hidden_size]
    assert not x.is_contiguous()

    ops_out, ops_scale = ops.scaled_fp8_quant(x)
    flat_out, flat_scale = ops.scaled_fp8_quant(x.contiguous())
    assert torch.equal(ops_scale, flat_scale)
    assert torch.equal(ops_out.view(torch.uint8), flat_out.view(torch.uint8))

    ref_out, ref_scale = ref_dynamic_per_tensor_fp8_quant(x)
    assert torch.equal(ref_scale, ops_scale)
    assert torch.equal(ref_out.view(torch.uint8), ops_out.view(torch.uint8))


# Regression test for a case with large activations where an int32 index cannot
# represent the number of elements.
@torch.inference_mode()
@pytest.mark.parametrize("seed", SEEDS)
def test_fp8_quant_large(seed: int) -> None:
    set_random_seed(seed)

    num_tokens = 1024000  # Mistral-Nemo's max_position_embeddings
    hidden_size = 1152  # Smallest hidden_size to reproduce the error
    dtype = torch.bfloat16

    x = torch.rand(num_tokens, hidden_size, dtype=dtype, device="cuda")
    ref_out, scale = ref_dynamic_per_tensor_fp8_quant(x)
    ops_out, _ = ops.scaled_fp8_quant(x, scale)

    # Minimize memory footprint in this test by freeing x and upconverting
    # the outputs in place. (torch.allclose does not support fp8)
    del x
    ref_out = ref_out.to(dtype=dtype)
    ops_out = ops_out.to(dtype=dtype)

    torch.testing.assert_close(ref_out, ops_out)


# Test static FP8 quantization with 2D group scales
GROUP_SHAPES_2D = [
    (-1, -1),  # Per-tensor
    (-1, 1),  # Per-channel
    (1, -1),  # Per-token
    (-1, 128),  # Per-head quantization
    (1, 128),  # DeepSeek-style per-token-per-group (group_m=1, group_n=128)
    (128, 128),  # DeepSeek-style block quantization
    (1, 64),  # Smaller group size
    (1, 16),  # Small group (scalar path in kernel)
    (4, 256),  # Non-trivial both dimensions
]
# Use sizes divisible by all group shapes
NUM_TOKENS_GROUP = [128, 512]
HIDDEN_SIZES_GROUP = [256, 1024, 2048]


@pytest.mark.parametrize("num_tokens", NUM_TOKENS_GROUP)
@pytest.mark.parametrize("hidden_size", HIDDEN_SIZES_GROUP)
@pytest.mark.parametrize("group_shape", GROUP_SHAPES_2D)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("seed", SEEDS)
@torch.inference_mode()
def test_static_fp8_quant_group_2d(
    num_tokens: int,
    hidden_size: int,
    group_shape: tuple[int, int],
    dtype: torch.dtype,
    seed: int,
) -> None:
    """Test static FP8 quantization with 2D group scales using scaled_quantize."""
    # Normalize group_shape (-1 means full extent)
    norm_group_m = num_tokens if group_shape[0] == -1 else group_shape[0]
    norm_group_n = hidden_size if group_shape[1] == -1 else group_shape[1]

    # Skip if sizes are not divisible by group shape
    if num_tokens % norm_group_m != 0 or hidden_size % norm_group_n != 0:
        pytest.skip(
            f"Skipping: ({num_tokens}, {hidden_size}) not divisible by "
            f"group_shape ({group_shape[0]}, {group_shape[1]})"
        )

    set_random_seed(seed)

    x = torch.rand(num_tokens, hidden_size, dtype=dtype, device="cuda")
    ref_out, scale = scaled_quantize(
        x,
        GroupShape(*group_shape),
        current_platform.fp8_dtype(),
        compute_dtype=torch.float32,
    )
    ops_out, ops_scale = ops.scaled_fp8_quant(x, scale=scale, group_shape=group_shape)

    torch.testing.assert_close(scale, ops_scale)
    torch.testing.assert_close(ref_out.float(), ops_out.float(), rtol=1.2e-1, atol=1e-5)

    opcheck_fp8_quant(ops_out, x, scale=scale)


@pytest.mark.parametrize("num_tokens", NUM_TOKENS_GROUP)
@pytest.mark.parametrize("hidden_size", HIDDEN_SIZES_GROUP)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("group_shape", [(1, -1), (-1, 1)])  # per-token, per-channel
@torch.inference_mode()
def test_static_fp8_quant_1d_scale(
    num_tokens: int,
    hidden_size: int,
    dtype: torch.dtype,
    seed: int,
    group_shape: tuple[int, int],
) -> None:
    """Test static FP8 quantization with 1D scale (per-token or per-channel)."""
    set_random_seed(seed)

    x = torch.rand(num_tokens, hidden_size, dtype=dtype, device="cuda")
    ref_out, scale_2d = scaled_quantize(
        x, GroupShape(*group_shape), FP8_DTYPE, compute_dtype=torch.float32
    )

    # Flatten scale to 1D for testing 1D scale path
    scale_1d = scale_2d.flatten()
    ops_out, ops_scale = ops.scaled_fp8_quant(
        x, scale=scale_1d, group_shape=group_shape
    )

    torch.testing.assert_close(scale_1d, ops_scale)
    torch.testing.assert_close(ref_out.float(), ops_out.float(), rtol=0.12, atol=0.0)

    opcheck_fp8_quant(ops_out, x, scale=scale_1d, group_shape=group_shape)


@pytest.mark.skipif(not current_platform.supports_fp8(), reason="FP8 not supported.")
@pytest.mark.parametrize("shape", [(16, 16), (257, 513), (1024, 4096)])
@pytest.mark.parametrize("weight_layout", ["transposed", "row_major", "padded"])
@pytest.mark.parametrize(
    "scale_layout",
    ["scalar_0d", "scalar_1d", "per_row_1d", "per_row_2d", "ptpc", "ptpc_fp16"],
)
def test_dequant_fp8_weight_bitwise(
    shape: tuple[int, int],
    weight_layout: str,
    scale_layout: str,
) -> None:
    """Verify dequant_fp8_weight matches the float32 reference expression bit
    for bit (values and layout) on inputs where a bf16-rounded or misapplied
    scale changes the result.
    """
    K, N = shape
    out_dtype = torch.float16 if scale_layout == "ptpc_fp16" else torch.bfloat16

    # Every finite FP8 code (NaN codes -> 0).
    codes = torch.arange(256, dtype=torch.uint8, device="cuda")
    if current_platform.is_fp8_fnuz():
        codes[codes == 0x80] = 0
    else:
        codes[(codes & 0x7F) == 0x7F] = 0
    w_flat = codes.repeat((K * N + 255) // 256)[: K * N]
    if weight_layout == "transposed":
        # [K, N] with stride (1, K), as after process_weights_after_loading.
        weight = w_flat.view(FP8_DTYPE).reshape(N, K).t()
    elif weight_layout == "row_major":
        weight = w_flat.view(FP8_DTYPE).reshape(K, N)
    else:
        # [K, N] with stride (1, K + 256), as _maybe_pad_fp8_weight leaves it on ROCm.
        padded = torch.zeros(N, K + 256, dtype=torch.uint8, device="cuda")
        padded[:, :K] = w_flat.reshape(N, K)
        weight = padded[:, :K].view(FP8_DTYPE).t()
        assert weight.stride() == (1, K + 256)

    num_scales = {
        "per_row_1d": K,
        "per_row_2d": K,
        "ptpc": N,
        "ptpc_fp16": N,
    }.get(scale_layout, 1)
    gen = torch.Generator().manual_seed(0)
    values = torch.empty(num_scales).uniform_(2**-12, 2**-4, generator=gen)
    # A checkpoint scale whose bf16 rounding changes 122 of the 254 finite e4m3fn codes.
    values[0] = torch.tensor(0x3A9B4C70, dtype=torch.int32).view(torch.float32)
    values = values.cuda()
    if scale_layout == "scalar_0d":
        scale = scale_bcast = values[0]
    elif scale_layout == "scalar_1d":
        scale = scale_bcast = values
    elif scale_layout == "per_row_1d":
        scale, scale_bcast = values, values.unsqueeze(1)
    elif scale_layout == "per_row_2d":
        scale = scale_bcast = values.unsqueeze(1)
    else:  # ptpc, ptpc_fp16: the transposed [N, 1] scale passed at the call site.
        scale = scale_bcast = values.reshape(N, 1).t()

    ref = (weight.to(torch.float32) * scale_bcast.to(torch.float32)).to(out_dtype)
    got = dequant_fp8_weight(weight, scale, out_dtype=out_dtype)

    assert got.shape == ref.shape
    assert got.dtype == out_dtype
    assert got.stride() == ref.stride()
    assert torch.equal(got.view(torch.uint16), ref.view(torch.uint16))

    if out_dtype == torch.bfloat16:
        bf16_scale = weight.to(torch.bfloat16) * scale_bcast.to(torch.bfloat16)
        assert not torch.equal(bf16_scale.view(torch.uint16), ref.view(torch.uint16))
    if num_scales > 1:
        first_only = (weight.to(torch.float32) * values[0].to(torch.float32)).to(
            out_dtype
        )
        assert not torch.equal(first_only.view(torch.uint16), ref.view(torch.uint16))

    if shape == (16, 16):
        got_cpu = dequant_fp8_weight(weight.cpu(), scale.cpu(), out_dtype=out_dtype)
        assert torch.equal(got_cpu.view(torch.uint16), ref.cpu().view(torch.uint16))


@pytest.mark.skipif(not current_platform.supports_fp8(), reason="FP8 not supported.")
@pytest.mark.parametrize("scale_layout", ["scalar", "per_row", "ptpc"])
def test_dequant_fp8_weight_eager_equals_compiled(scale_layout: str) -> None:
    """Eager (Triton kernel) and torch.compile (Inductor-fused expression)
    produce identical bits across scalar, per-row, and PTPC scale layouts.
    """
    K, N = 64, 128
    codes = torch.arange(256, dtype=torch.uint8, device="cuda")
    if current_platform.is_fp8_fnuz():
        codes[codes == 0x80] = 0
    else:
        codes[(codes & 0x7F) == 0x7F] = 0
    weight = (
        codes.repeat((K * N + 255) // 256)[: K * N].view(FP8_DTYPE).reshape(N, K).t()
    )

    num_scales = {"per_row": K, "ptpc": N}.get(scale_layout, 1)
    gen = torch.Generator().manual_seed(0)
    values = torch.empty(num_scales).uniform_(2**-12, 2**-4, generator=gen)
    values[0] = torch.tensor(0x3A9B4C70, dtype=torch.int32).view(torch.float32)
    values = values.cuda()
    if scale_layout == "scalar":
        scale = values[0]
    elif scale_layout == "per_row":
        scale = values
    else:
        scale = values.reshape(N, 1).t()

    compiled_fn = torch.compile(dequant_fp8_weight, fullgraph=True)
    eager_out = dequant_fp8_weight(weight, scale, out_dtype=torch.bfloat16)
    compiled_out = compiled_fn(weight, scale, out_dtype=torch.bfloat16)
    assert torch.equal(eager_out.view(torch.uint16), compiled_out.view(torch.uint16))
