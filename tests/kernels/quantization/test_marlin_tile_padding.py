# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for Marlin thread-tile padding of TP-sharded weight shapes.

Run `pytest tests/kernels/quantization/test_marlin_tile_padding.py`.
"""

from types import SimpleNamespace

import pytest
import torch

from vllm import _custom_ops as ops
from vllm.model_executor.layers.quantization.utils.marlin_utils import (
    GPTQ_MARLIN_TILE,
    apply_gptq_marlin_linear,
    marlin_make_empty_g_idx,
    marlin_make_workspace_new,
    marlin_moe_padded_intermediate,
    marlin_pad_qweight,
    marlin_pad_scales,
    marlin_padded_nk,
    marlin_permute_scales,
    marlin_repacked_nk,
    marlin_zero_points,
)
from vllm.model_executor.layers.quantization.utils.marlin_utils_fp4 import (
    apply_fp4_marlin_linear,
    is_fp4_marlin_supported,
    prepare_fp4_layer_for_marlin,
)
from vllm.model_executor.layers.quantization.utils.marlin_utils_fp8 import (
    apply_fp8_marlin_linear,
    apply_mxfp8_marlin_linear,
    is_fp8_marlin_supported,
    prepare_fp8_layer_for_marlin,
    prepare_mxfp8_layer_for_marlin,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    gptq_pack,
    gptq_quantize_weights,
    quantize_weights,
)
from vllm.platforms import current_platform
from vllm.scalar_type import scalar_types

# (size_n, size_k) rank-local shapes that violate Marlin tile alignment,
# e.g. produced by TP-sharding dims that are valid at TP=1.
ODD_SHAPES = [
    (200, 288),  # N padded
    (256, 208),  # K padded
    (200, 208),  # both padded
    (4640, 512),  # Nemotron-Super-120B q_proj shard at TP=4
]
ALIGNED_SHAPES = [(64, 128), (128, 64), (256, 256), (4608, 4096)]


def _is_tile_aligned(size_n: int, size_k: int) -> bool:
    return (size_n % 64 == 0 and size_k % 128 == 0) or (
        size_n % 128 == 0 and size_k % 64 == 0
    )


@pytest.mark.parametrize("shape", ODD_SHAPES + ALIGNED_SHAPES)
@pytest.mark.parametrize("group_size", [-1, 16, 32, 64, 128])
def test_marlin_padded_nk(shape, group_size):
    size_n, size_k = shape
    padded_n, padded_k = marlin_padded_nk(size_n, size_k, group_size)

    assert padded_n >= size_n and padded_k >= size_k
    assert _is_tile_aligned(padded_n, padded_k)
    if group_size > 0:
        assert padded_k % group_size == 0

    # Aligned shapes must pass through unchanged (zero hot-path cost).
    if _is_tile_aligned(size_n, size_k) and (
        group_size <= 0 or size_k % group_size == 0
    ):
        assert (padded_n, padded_k) == (size_n, size_k)

    # Minimal: no valid shape with a smaller padded area exists.
    area = padded_n * padded_k
    for cand_n in range(size_n, padded_n + 1):
        for cand_k in range(size_k, padded_k + 1):
            if (
                _is_tile_aligned(cand_n, cand_k)
                and (group_size <= 0 or cand_k % group_size == 0)
                and cand_n * cand_k < area
            ):
                pytest.fail(f"({cand_n}, {cand_k}) beats ({padded_n}, {padded_k})")

    # Apply-time derivation from the repacked-tensor shape must round-trip.
    for num_bits in (4, 8):
        pack_factor = 32 // num_bits
        repacked_shape = (
            padded_k // GPTQ_MARLIN_TILE,
            padded_n * GPTQ_MARLIN_TILE // pack_factor,
        )
        repacked = torch.empty(repacked_shape, device="meta")
        assert marlin_repacked_nk(repacked, num_bits) == (padded_n, padded_k)


def test_marlin_pad_helpers_shapes():
    size_n, size_k, group_size = 200, 208, 16
    padded_n, padded_k = marlin_padded_nk(size_n, size_k, group_size)

    qweight = torch.zeros(size_k // 8, size_n, dtype=torch.int32)
    padded = marlin_pad_qweight(qweight, size_n, size_k, padded_n, padded_k)
    assert padded.shape == (padded_k // 8, padded_n)

    scales = torch.ones(size_k // group_size, size_n)
    padded = marlin_pad_scales(scales, size_n, size_k, padded_n, padded_k, group_size)
    assert padded.shape == (padded_k // group_size, padded_n)
    assert padded[:, size_n:].abs().sum() == 0

    channelwise = torch.ones(1, size_n)
    padded = marlin_pad_scales(channelwise, size_n, size_k, padded_n, padded_k, -1)
    assert padded.shape == (1, padded_n)


# Rank-local MoE intermediate sizes. group<=0 / 32 with a non-multiple-of-64
# size is where tile padding triggers; 64/128 are already tile-aligned.
MOE_INTERMEDIATE_SIZES = [64, 96, 100, 176, 192, 256, 2816]


@pytest.mark.parametrize("intermediate", MOE_INTERMEDIATE_SIZES)
@pytest.mark.parametrize("group_size", [-1, 32, 64, 128])
def test_marlin_moe_padded_intermediate(intermediate, group_size):
    # The MoE gate only admits shapes where the group does not straddle the
    # boundary, i.e. group divides the intermediate size.
    if group_size > 0 and intermediate % group_size != 0:
        pytest.skip("group straddles the boundary; rejected by the MoE gate")

    padded = marlin_moe_padded_intermediate(intermediate, group_size)
    assert padded >= intermediate
    # Valid MoE thread tile: gate-up n = 2*intermediate % 128, down k % 64.
    assert (2 * padded) % 128 == 0
    assert padded % 64 == 0
    if group_size > 0:
        assert padded % group_size == 0

    # Minimal: no smaller valid intermediate exists.
    for cand in range(intermediate, padded):
        if (
            (2 * cand) % 128 == 0
            and cand % 64 == 0
            and (group_size <= 0 or cand % group_size == 0)
        ):
            pytest.fail(f"{cand} beats {padded}")

    # Already-tile-aligned sizes pass through unchanged (zero hot-path cost).
    if intermediate % 64 == 0:
        assert padded == intermediate


def test_marlin_moe_pad_helpers_shapes():
    from vllm.model_executor.layers.fused_moe.oracle.int_wna16 import (
        _pad_rows,
        _pad_w13_bias,
        _pad_w13_shard_cols,
    )

    E, rows, N, padded_N = 2, 8, 96, 128

    # w13 stores the two gate/up shards along the last dim; padding each shard
    # must preserve the loaded values and zero the padded columns.
    w13 = torch.arange(E * rows * 2 * N).reshape(E, rows, 2 * N).float()
    padded = _pad_w13_shard_cols(w13, N, padded_N)
    assert padded.shape == (E, rows, 2 * padded_N)
    shards = padded.view(E, rows, 2, padded_N)
    orig = w13.view(E, rows, 2, N)
    assert torch.equal(shards[..., :N], orig)
    assert shards[..., N:].abs().sum() == 0

    # w2 stores the intermediate dim in the rows.
    w2 = torch.ones(E, N // 32, 16)
    padded = _pad_rows(w2, padded_N // 32)
    assert padded.shape == (E, padded_N // 32, 16)
    assert padded[:, N // 32 :, :].abs().sum() == 0

    bias = torch.arange(E * 2 * N).reshape(E, 2 * N).float()
    padded = _pad_w13_bias(bias, N, padded_N)
    assert padded.shape == (E, 2 * padded_N)
    bias_shards = padded.view(E, 2, padded_N)
    assert torch.equal(bias_shards[..., :N], bias.view(E, 2, N))
    assert bias_shards[..., N:].abs().sum() == 0


def _gpu_marlin_unsupported() -> bool:
    return not (
        current_platform.is_cuda() and current_platform.has_device_capability(80)
    )


@pytest.mark.skipif(
    _gpu_marlin_unsupported() or not is_fp8_marlin_supported(),
    reason="FP8 Marlin is not supported on this GPU type.",
)
@pytest.mark.parametrize("shape", ODD_SHAPES)
@pytest.mark.parametrize("use_bias", [False, True])
def test_fp8_marlin_padded_round_trip(shape, use_bias):
    size_n, size_k = shape
    dtype = torch.float16
    layer = torch.nn.Module()
    layer.output_size_per_partition = size_n
    layer.input_size_per_partition = size_k
    layer.orig_dtype = dtype

    weight = torch.randn(size_k, size_n, dtype=dtype, device="cuda") / size_k**0.5
    scale = weight.abs().max() / 448
    weight_fp8 = (weight / scale).to(torch.float8_e4m3fn)
    layer.weight = torch.nn.Parameter(weight_fp8, requires_grad=False)
    layer.weight_scale = torch.nn.Parameter(
        scale.to(torch.float32), requires_grad=False
    )
    bias = None
    if use_bias:
        bias = torch.randn(size_n, dtype=dtype, device="cuda")
        layer.bias = torch.nn.Parameter(bias.clone(), requires_grad=False)

    prepare_fp8_layer_for_marlin(layer, size_k_first=True)

    x = torch.randn(8, size_k, dtype=dtype, device="cuda")
    output = apply_fp8_marlin_linear(
        input=x,
        weight=layer.weight,
        weight_scale=layer.weight_scale,
        workspace=layer.workspace,
        size_n=size_n,
        size_k=size_k,
        bias=layer.bias if use_bias else None,
    )
    ref = x @ (weight_fp8.to(dtype) * scale.to(dtype))
    if use_bias:
        ref = ref + bias

    assert output.shape == (8, size_n)
    torch.testing.assert_close(output, ref, rtol=2e-2, atol=2e-2)


def _dequant_fp4(packed: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """Dequantize packed e2m1 nibbles (N, K // 2) -> (N, K) in dtype."""
    lo = (packed & 0b10000000) | ((packed & 0b01110000) >> 2)
    lo = lo.view(torch.float8_e4m3fn).to(dtype) * (2**6)
    hi_bits = packed << 4
    hi = (hi_bits & 0b10000000) | ((hi_bits & 0b01110000) >> 2)
    hi = hi.view(torch.float8_e4m3fn).to(dtype) * (2**6)
    return torch.cat([hi.unsqueeze(2), lo.unsqueeze(2)], 2).view(packed.size(0), -1)


@pytest.mark.skipif(
    _gpu_marlin_unsupported() or not is_fp4_marlin_supported(),
    reason="FP4 Marlin is not supported on this GPU type.",
)
@pytest.mark.parametrize("shape", ODD_SHAPES)
def test_nvfp4_marlin_padded_round_trip(shape):
    size_n, size_k = shape
    group_size = 16
    dtype = torch.float16
    layer = torch.nn.Module()
    layer.output_size_per_partition = size_n
    layer.input_size_per_partition = size_k
    layer.params_dtype = dtype

    packed = torch.randint(
        0, 256, (size_n, size_k // 2), dtype=torch.uint8, device="cuda"
    )
    scales = (torch.rand(size_n, size_k // group_size, device="cuda") + 0.25).to(
        torch.float8_e4m3fn
    )
    global_scale = torch.tensor([0.002], dtype=torch.float32, device="cuda")

    ref_weight = (
        _dequant_fp4(packed, dtype)
        * scales.to(dtype).repeat_interleave(group_size, 1)
        * global_scale.to(dtype)
    )

    layer.weight = torch.nn.Parameter(packed, requires_grad=False)
    layer.weight_scale = torch.nn.Parameter(scales, requires_grad=False)
    layer.weight_global_scale = torch.nn.Parameter(global_scale, requires_grad=False)

    prepare_fp4_layer_for_marlin(layer)

    x = torch.randn(8, size_k, dtype=dtype, device="cuda") / size_k**0.5
    output = apply_fp4_marlin_linear(
        input=x,
        weight=layer.weight,
        weight_scale=layer.weight_scale,
        weight_global_scale=layer.weight_global_scale,
        workspace=layer.workspace,
        size_n=size_n,
        size_k=size_k,
    )
    ref = x @ ref_weight.T

    assert output.shape == (8, size_n)
    torch.testing.assert_close(output, ref, rtol=2e-2, atol=2e-2)


@pytest.mark.skipif(
    _gpu_marlin_unsupported(),
    reason="Marlin is not supported on this GPU type.",
)
@pytest.mark.parametrize("shape", ODD_SHAPES)
@pytest.mark.parametrize("group_size", [-1, 128])
def test_gptq_marlin_padded_round_trip(shape, group_size):
    """Pad-then-repack a GPTQ int4 weight the way MarlinLinearKernel does and
    check the GEMM against the dequantized reference.

    Symmetric int4's quantized zero decodes to -8, so this exercises the
    zero-padded-scales cancellation, not just zero weights.
    """
    size_n, size_k = shape
    if group_size > 0 and size_k % group_size != 0:
        pytest.skip("group must divide the rank-local K (not fixable by padding)")
    dtype = torch.float16
    quant_type = scalar_types.uint4b8
    device = torch.device("cuda")

    weight = torch.randn(size_k, size_n, dtype=dtype, device=device) / size_k**0.5
    w_ref, q_w, s, _, _ = gptq_quantize_weights(
        weight, quant_type, group_size, act_order=False
    )
    qweight = gptq_pack(q_w, quant_type.size_bits, size_k, size_n)

    padded_n, padded_k = marlin_padded_nk(size_n, size_k, group_size)
    qweight = marlin_pad_qweight(qweight, size_n, size_k, padded_n, padded_k)
    marlin_qweight = ops.gptq_marlin_repack(
        b_q_weight=qweight,
        perm=torch.empty(0, dtype=torch.int, device=device),
        size_k=padded_k,
        size_n=padded_n,
        num_bits=quant_type.size_bits,
    )
    s = marlin_pad_scales(s, size_n, size_k, padded_n, padded_k, group_size)
    marlin_s = marlin_permute_scales(
        s, size_k=padded_k, size_n=padded_n, group_size=group_size
    )

    x = torch.randn(8, size_k, dtype=dtype, device=device)
    output = apply_gptq_marlin_linear(
        input=x,
        weight=marlin_qweight,
        weight_scale=marlin_s,
        weight_zp=marlin_make_empty_g_idx(device),
        g_idx=marlin_make_empty_g_idx(device),
        g_idx_sort_indices=marlin_make_empty_g_idx(device),
        workspace=marlin_make_workspace_new(device),
        wtype=quant_type,
        output_size_per_partition=size_n,
        input_size_per_partition=size_k,
        is_k_full=True,
    )
    ref = x @ w_ref

    assert output.shape == (8, size_n)
    torch.testing.assert_close(output, ref, rtol=2e-2, atol=2e-2)


@pytest.mark.skipif(
    _gpu_marlin_unsupported() or not is_fp8_marlin_supported(),
    reason="FP8 Marlin is not supported on this GPU type.",
)
@pytest.mark.parametrize("shape", [(200, 512), (4640, 512)])
def test_fp8_block_marlin_padded_round_trip(shape):
    """Block-quantized FP8 (e.g. Nemotron NVFP4 checkpoints' FP8 layers):
    group_size=128 exercises the lcm K-alignment in marlin_padded_nk and the
    weight_scale_inv group-wise scale padding."""
    size_n, size_k = shape
    block = 128
    dtype = torch.float16
    layer = torch.nn.Module()
    layer.output_size_per_partition = size_n
    layer.input_size_per_partition = size_k
    layer.orig_dtype = dtype
    layer.weight_block_size = [block, block]

    weight = torch.randn(size_n, size_k, dtype=dtype, device="cuda") / size_k**0.5
    n_blocks, k_blocks = (size_n + block - 1) // block, size_k // block
    padded = torch.zeros(n_blocks * block, size_k, dtype=dtype, device="cuda")
    padded[:size_n] = weight
    scales = padded.view(n_blocks, block, k_blocks, block).abs().amax(dim=(1, 3)) / 448
    scales_expanded = scales.repeat_interleave(block, 0)[:size_n].repeat_interleave(
        block, 1
    )
    weight_fp8 = (weight / scales_expanded).to(torch.float8_e4m3fn)

    layer.weight = torch.nn.Parameter(weight_fp8, requires_grad=False)
    layer.weight_scale_inv = torch.nn.Parameter(
        scales.to(torch.float32), requires_grad=False
    )

    prepare_fp8_layer_for_marlin(layer, size_k_first=False)

    x = torch.randn(8, size_k, dtype=dtype, device="cuda")
    output = apply_fp8_marlin_linear(
        input=x,
        weight=layer.weight,
        weight_scale=layer.weight_scale_inv,
        workspace=layer.workspace,
        size_n=size_n,
        size_k=size_k,
        bias=None,
    )
    ref = x @ (weight_fp8.to(dtype) * scales_expanded.to(dtype)).T

    assert output.shape == (8, size_n)
    torch.testing.assert_close(output, ref, rtol=2e-2, atol=2e-2)


@pytest.mark.skipif(
    _gpu_marlin_unsupported() or not is_fp8_marlin_supported(),
    reason="FP8 Marlin is not supported on this GPU type.",
)
@pytest.mark.parametrize("shape", [(200, 288), (4640, 512)])
def test_mxfp8_marlin_padded_round_trip(shape):
    """MXFP8 exercises the e8m0 scale path, where padded 0.0 scales clamp to
    2^-127 instead of zero and must still contribute nothing."""
    size_n, size_k = shape
    group_size = 32
    # The e8m0-scale Marlin kernels are only instantiated for bf16 activations.
    dtype = torch.bfloat16
    layer = torch.nn.Module()
    layer.output_size_per_partition = size_n
    layer.input_size_per_partition = size_k

    weight_fp8 = (torch.randn(size_n, size_k, dtype=dtype, device="cuda") / 4).to(
        torch.float8_e4m3fn
    )
    # e8m0 exponents around 1.0 (127): scales in [2^-6, 2^0]
    scales = torch.randint(
        121, 128, (size_n, size_k // group_size), dtype=torch.uint8, device="cuda"
    )
    ref_weight = weight_fp8.to(dtype) * (
        2.0 ** (scales.to(dtype) - 127)
    ).repeat_interleave(group_size, 1)

    layer.weight = torch.nn.Parameter(weight_fp8, requires_grad=False)
    layer.weight_scale = torch.nn.Parameter(scales, requires_grad=False)

    prepare_mxfp8_layer_for_marlin(layer)

    x = torch.randn(8, size_k, dtype=dtype, device="cuda") / size_k**0.5
    output = apply_mxfp8_marlin_linear(
        input=x,
        weight=layer.weight,
        weight_scale=layer.weight_scale,
        workspace=layer.workspace,
        size_n=size_n,
        size_k=size_k,
    )
    ref = x @ ref_weight.T

    assert output.shape == (8, size_n)
    torch.testing.assert_close(output, ref, rtol=2e-2, atol=2e-2)


@pytest.mark.skipif(
    _gpu_marlin_unsupported(),
    reason="Marlin is not supported on this GPU type.",
)
@pytest.mark.parametrize("shape", [(200, 512), (4640, 512)])
def test_awq_zp_marlin_padded_round_trip(shape):
    """AWQ-style uint4 with runtime zero-points, padded the way
    MarlinLinearKernel does: padded columns rely on (q=0 - zp=0) * scale=0."""
    size_n, size_k = shape
    group_size = 128
    dtype = torch.float16
    quant_type = scalar_types.uint4
    device = torch.device("cuda")

    weight = torch.randn(size_k, size_n, dtype=dtype, device=device) / size_k**0.5
    w_ref, q_w, s, zp = quantize_weights(
        weight, quant_type, group_size, zero_points=True
    )
    qweight = gptq_pack(q_w, quant_type.size_bits, size_k, size_n)

    padded_n, padded_k = marlin_padded_nk(size_n, size_k, group_size)
    qweight = marlin_pad_qweight(qweight, size_n, size_k, padded_n, padded_k)
    marlin_qweight = ops.gptq_marlin_repack(
        b_q_weight=qweight,
        perm=torch.empty(0, dtype=torch.int, device=device),
        size_k=padded_k,
        size_n=padded_n,
        num_bits=quant_type.size_bits,
    )
    s = marlin_pad_scales(s, size_n, size_k, padded_n, padded_k, group_size)
    marlin_s = marlin_permute_scales(
        s, size_k=padded_k, size_n=padded_n, group_size=group_size
    )
    zp = marlin_pad_scales(zp, size_n, size_k, padded_n, padded_k, group_size)
    marlin_zp = marlin_zero_points(
        zp,
        size_k=padded_k // group_size,
        size_n=padded_n,
        num_bits=quant_type.size_bits,
    )

    x = torch.randn(8, size_k, dtype=dtype, device=device)
    output = apply_gptq_marlin_linear(
        input=x,
        weight=marlin_qweight,
        weight_scale=marlin_s,
        weight_zp=marlin_zp,
        g_idx=marlin_make_empty_g_idx(device),
        g_idx_sort_indices=marlin_make_empty_g_idx(device),
        workspace=marlin_make_workspace_new(device),
        wtype=quant_type,
        output_size_per_partition=size_n,
        input_size_per_partition=size_k,
        is_k_full=True,
    )
    ref = x @ w_ref

    assert output.shape == (8, size_n)
    torch.testing.assert_close(output, ref, rtol=2e-2, atol=2e-2)


class _FakeLinear:
    def __init__(self, size_n, size_k, input_size=None):
        self.output_size_per_partition = size_n
        self.input_size_per_partition = size_k
        self.output_size = size_n
        self.input_size = input_size if input_size is not None else size_k


def test_check_marlin_supports_layer_allow_tile_padding():
    from vllm.model_executor.layers.quantization.utils.marlin_utils import (
        check_marlin_supports_layer,
    )

    # Tile-misaligned but group-aligned: rejected strictly, allowed w/ padding
    layer = _FakeLinear(4640, 512, input_size=2048)
    assert not check_marlin_supports_layer(layer, 128)
    assert check_marlin_supports_layer(layer, 128, allow_tile_padding=True)
    assert check_marlin_supports_layer(layer, -1, allow_tile_padding=True)

    # A group straddling the TP shard cannot be fixed by padding
    layer = _FakeLinear(4608, 4672, input_size=18688)
    assert not check_marlin_supports_layer(layer, 128, allow_tile_padding=True)


@pytest.mark.skipif(
    _gpu_marlin_unsupported(),
    reason="Marlin is not supported on this GPU type.",
)
@pytest.mark.parametrize("group_size", [-1, 32])
@pytest.mark.parametrize("shape", [(96, 256, 8), (160, 512, 4)])
def test_gptq_marlin_moe_padded_round_trip(shape, group_size):
    """Pad a tile-misaligned MoE intermediate the way the WNA16 Marlin MoE prep
    does, run the real repack + fused_marlin_moe, and check against the
    dequantized reference. Symmetric int4's quantized zero decodes to -8, so the
    padded region only stays out of the output via the zero-padded scales.
    """
    from tests.kernels.utils import torch_experts
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.model_executor.layers.fused_moe import fused_topk
    from vllm.model_executor.layers.fused_moe.experts.marlin_moe import (
        fused_marlin_moe,
    )
    from vllm.model_executor.layers.fused_moe.oracle.int_wna16 import (
        _pad_rows,
        _pad_w13_shard_cols,
    )
    from vllm.model_executor.layers.quantization.utils.marlin_utils import (
        marlin_moe_padded_intermediate,
        marlin_moe_permute_scales,
    )

    n, k, e = shape
    topk, m = 2, 33
    padded_n = marlin_moe_padded_intermediate(n, group_size)
    assert padded_n != n, "test should exercise padding"

    dtype = torch.float16
    device = torch.device("cuda")
    quant_type = scalar_types.uint4b8
    bits = quant_type.size_bits
    pack = 32 // bits

    a = torch.randn((m, k), device=device, dtype=dtype) / 10
    w1 = torch.randn((e, 2 * n, k), device=device, dtype=dtype) / k**0.5
    w2 = torch.randn((e, k, n), device=device, dtype=dtype) / n**0.5

    def quant(w, size_k, size_n):
        # w is (size_n, size_k); gptq expects (size_k, size_n).
        ref, q_w, s, _, _ = gptq_quantize_weights(
            w.T, quant_type, group_size, act_order=False
        )
        return ref, gptq_pack(q_w, bits, size_k, size_n), s

    w13_qw, w13_s, w13_ref = [], [], []
    w2_qw, w2_s, w2_ref = [], [], []
    for i in range(e):
        ref, qw, s = quant(w1[i], k, 2 * n)
        w13_ref.append(ref.T)  # (2n, k)
        w13_qw.append(qw)
        w13_s.append(s)
        ref, qw, s = quant(w2[i], n, k)
        w2_ref.append(ref.T)  # (k, n)
        w2_qw.append(qw)
        w2_s.append(s)

    w13_qweight = torch.stack(w13_qw)
    w2_qweight = torch.stack(w2_qw)
    w13_scales = torch.stack(w13_s)
    w2_scales = torch.stack(w2_s)
    w1_ref = torch.stack(w13_ref)  # (e, 2n, k)
    w2_ref = torch.stack(w2_ref)  # (e, k, n)

    # Pad the intermediate via the production helpers.
    w13_qweight = _pad_w13_shard_cols(w13_qweight, n, padded_n)
    w2_qweight = _pad_rows(w2_qweight, padded_n // pack)
    w13_scales = _pad_w13_shard_cols(w13_scales, n, padded_n)
    if group_size > 0:
        w2_scales = _pad_rows(w2_scales, padded_n // group_size)

    sort_idx = torch.empty((e, 0), dtype=torch.int32, device=device)
    marlin_w13 = ops.gptq_marlin_moe_repack(
        w13_qweight, sort_idx, w13_qweight.shape[1] * pack, w13_qweight.shape[2], bits
    )
    marlin_w2 = ops.gptq_marlin_moe_repack(
        w2_qweight, sort_idx, w2_qweight.shape[1] * pack, w2_qweight.shape[2], bits
    )
    group_or_pack = group_size if group_size != -1 else pack
    marlin_w13_s = marlin_moe_permute_scales(
        s=w13_scales, size_k=n, size_n=w13_scales.shape[2], group_size=group_size
    )
    marlin_w2_s = marlin_moe_permute_scales(
        s=w2_scales,
        size_k=w2_scales.shape[1] * group_or_pack,
        size_n=w2_scales.shape[2],
        group_size=group_size,
    )

    score = torch.randn((m, e), device=device, dtype=dtype)
    topk_weights, topk_ids, _ = fused_topk(a, score, topk, False)

    marlin_out = fused_marlin_moe(
        a,
        marlin_w13,
        marlin_w2,
        None,
        None,
        marlin_w13_s,
        marlin_w2_s,
        topk_weights,
        topk_ids,
        quant_type_id=quant_type.id,
        global_num_experts=e,
        is_k_full=True,
    )
    with set_current_vllm_config(VllmConfig()):
        ref = torch_experts(
            a,
            w1_ref,
            w2_ref,
            topk_weight=topk_weights,
            topk_ids=topk_ids,
            global_num_experts=e,
        )

    torch.testing.assert_close(marlin_out, ref, atol=5e-2, rtol=0)


@pytest.mark.skipif(
    current_platform.is_rocm(),
    reason="MoE Marlin is not selected on ROCm.",
)
def test_check_moe_marlin_supports_layer_padding():
    from vllm.model_executor.layers.quantization.utils.marlin_utils import (
        check_moe_marlin_supports_layer,
    )

    def make_layer(hidden, intermediate):
        layer = SimpleNamespace()
        layer.hidden_size = hidden
        layer.apply_router_weight_on_input = False
        layer.moe_config = SimpleNamespace(
            intermediate_size_per_partition_unpadded=intermediate
        )
        return layer

    # group=32 with intermediate % 64 != 0: rejected strictly, accepted w/ padding
    layer = make_layer(4096, 96)
    assert not check_moe_marlin_supports_layer(layer, 32)
    assert check_moe_marlin_supports_layer(layer, 32, allow_tile_padding=True)
    # channelwise misaligned intermediate is paddable
    assert check_moe_marlin_supports_layer(layer, -1, allow_tile_padding=True)

    # A group straddling the boundary cannot be fixed by padding
    layer = make_layer(4096, 176)
    assert not check_moe_marlin_supports_layer(layer, 128, allow_tile_padding=True)

    # hidden_size is the MoE I/O extent and is never padded
    layer = make_layer(4090, 128)
    assert not check_moe_marlin_supports_layer(layer, 64, allow_tile_padding=True)


@pytest.mark.skipif(
    _gpu_marlin_unsupported() or not is_fp8_marlin_supported(),
    reason="FP8 Marlin is not supported on this GPU type.",
)
@pytest.mark.parametrize("quant", ["channel", "tensor"])
@pytest.mark.parametrize("shape", [(96, 256, 8), (160, 512, 4)])
def test_fp8_marlin_moe_padded_round_trip(shape, quant):
    """FP8 weight-only MoE: pad a tile-misaligned intermediate and check the
    real prepare + fused_marlin_moe against the dequantized reference."""
    from tests.kernels.utils import torch_experts
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.model_executor.layers.fused_moe import fused_topk
    from vllm.model_executor.layers.fused_moe.experts.marlin_moe import (
        fused_marlin_moe,
    )
    from vllm.model_executor.layers.quantization.utils.marlin_utils import (
        marlin_moe_intermediate_size,
        marlin_moe_padded_intermediate,
    )
    from vllm.model_executor.layers.quantization.utils.marlin_utils_fp8 import (
        prepare_fp8_moe_layer_for_marlin,
    )

    n, k, e = shape
    topk, m = 2, 33
    fp8 = torch.float8_e4m3fn
    dtype = torch.bfloat16
    device = torch.device("cuda")
    padded_n = marlin_moe_padded_intermediate(n, -1)
    assert padded_n != n

    def q(w):  # (out, in) -> fp8 weight, scale, dequant reference
        dim = None if quant == "tensor" else 1
        s = (w.abs().amax(dim, keepdim=dim is not None) / 448.0).clamp(min=1e-8)
        wq = (w / s).clamp(-448, 448).to(fp8)
        ref = wq.to(dtype) * s.to(dtype)
        s = s.reshape(1) if quant == "tensor" else s.squeeze(1)
        return wq, s, ref

    a = torch.randn((m, k), device=device, dtype=dtype) / 10
    w1 = torch.randn((e, 2 * n, k), device=device, dtype=dtype) / k**0.5
    w2 = torch.randn((e, k, n), device=device, dtype=dtype) / n**0.5
    w13_q, w13_s, w1_ref = zip(*(q(w1[i]) for i in range(e)))
    w2_q, w2_s, w2_ref = zip(*(q(w2[i]) for i in range(e)))

    w13_weight, w2_weight = torch.stack(w13_q), torch.stack(w2_q)
    layer = SimpleNamespace(
        num_experts=e,
        hidden_size=k,
        intermediate_size_per_partition=n,
        orig_dtype=dtype,
        w13_weight=w13_weight,
    )
    pw13, pw2, ps13, ps2 = prepare_fp8_moe_layer_for_marlin(
        layer, w13_weight, w2_weight, torch.stack(w13_s), torch.stack(w2_s)
    )
    assert marlin_moe_intermediate_size(pw13, pw2) == padded_n

    score = torch.randn((m, e), device=device, dtype=dtype)
    topk_weights, topk_ids, _ = fused_topk(a, score, topk, False)
    out = fused_marlin_moe(
        a,
        pw13,
        pw2,
        None,
        None,
        ps13,
        ps2,
        topk_weights,
        topk_ids,
        quant_type_id=scalar_types.float8_e4m3fn.id,
        global_num_experts=e,
        is_k_full=True,
        workspace=layer.workspace,
    )
    with set_current_vllm_config(VllmConfig()):
        ref = torch_experts(
            a,
            torch.stack(w1_ref),
            torch.stack(w2_ref),
            topk_weight=topk_weights,
            topk_ids=topk_ids,
            global_num_experts=e,
        )
    torch.testing.assert_close(out, ref, atol=8e-2, rtol=0)


@pytest.mark.skipif(
    _gpu_marlin_unsupported() or not is_fp8_marlin_supported(),
    reason="FP8 Marlin is not supported on this GPU type.",
)
@pytest.mark.parametrize("shape", [(96, 256, 8), (160, 512, 4)])
def test_mxfp8_marlin_moe_padded_round_trip(shape):
    """MXFP8 weight-only MoE round-trip at a tile-misaligned intermediate, with
    unit e8m0 scales so the reference is the exact fp8 dequant."""
    from tests.kernels.utils import torch_experts
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.model_executor.layers.fused_moe import fused_topk
    from vllm.model_executor.layers.fused_moe.experts.marlin_moe import (
        fused_marlin_moe,
    )
    from vllm.model_executor.layers.quantization.utils.marlin_utils import (
        marlin_moe_intermediate_size,
        marlin_moe_padded_intermediate,
    )
    from vllm.model_executor.layers.quantization.utils.marlin_utils_fp8 import (
        prepare_mxfp8_moe_layer_for_marlin,
    )

    n, k, e = shape
    topk, m, gs, e8m0_one = 2, 33, 32, 127
    fp8 = torch.float8_e4m3fn
    dtype = torch.bfloat16
    device = torch.device("cuda")
    padded_n = marlin_moe_padded_intermediate(n, gs)
    assert padded_n != n

    a = torch.randn((m, k), device=device, dtype=dtype) / 10
    w13_weight = torch.randn((e, 2 * n, k), device=device, dtype=dtype) / k**0.5
    w2_weight = torch.randn((e, k, n), device=device, dtype=dtype) / n**0.5
    w13_weight = w13_weight.clamp(-448, 448).to(fp8)
    w2_weight = w2_weight.clamp(-448, 448).to(fp8)
    w13_scale = torch.full(
        (e, 2 * n, k // gs), e8m0_one, dtype=torch.uint8, device=device
    )
    w2_scale = torch.full((e, k, n // gs), e8m0_one, dtype=torch.uint8, device=device)

    layer = SimpleNamespace(
        num_experts=e, hidden_size=k, intermediate_size_per_partition=n
    )
    with set_current_vllm_config(VllmConfig()):
        pw13, pw2, ps13, ps2 = prepare_mxfp8_moe_layer_for_marlin(
            layer, w13_weight, w2_weight, w13_scale, w2_scale
        )
    assert marlin_moe_intermediate_size(pw13, pw2) == padded_n

    score = torch.randn((m, e), device=device, dtype=dtype)
    topk_weights, topk_ids, _ = fused_topk(a, score, topk, False)
    out = fused_marlin_moe(
        a,
        pw13,
        pw2,
        None,
        None,
        ps13,
        ps2,
        topk_weights,
        topk_ids,
        quant_type_id=scalar_types.float8_e4m3fn.id,
        global_num_experts=e,
        is_k_full=True,
        workspace=layer.workspace,
    )
    with set_current_vllm_config(VllmConfig()):
        ref = torch_experts(
            a,
            w13_weight.to(dtype),
            w2_weight.to(dtype),
            topk_weight=topk_weights,
            topk_ids=topk_ids,
            global_num_experts=e,
        )
    torch.testing.assert_close(out, ref, atol=8e-2, rtol=0)
