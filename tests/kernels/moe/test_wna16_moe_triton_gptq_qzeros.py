# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Regression coverage for the WNA16 MoE qzeros reshape in
`convert_to_wna16_moe_kernel_format`'s TRITON backend branch.

Both AutoGPTQ and compressed-tensors checkpoints persist their packed weights
K-first int32; the TRITON branch transposes w13/w2 and scales to the N-first
uint8 layout `fused_moe_kernel_gptq_awq` expects. This test locks in that the
zero-points get the matching transform -- and that its result actually maps
zero-point k to output channel k, not just that the shape/dtype line up.

The compressed-tensors case is run asymmetric (`symmetric=False`), where the
zero-point values are load-bearing: a byte/nibble permutation bug there
silently offsets every dequantized weight rather than crashing.

Pure tensor-shape/value test, no GPU or Triton kernel launch involved.
"""

import pytest
import torch
from compressed_tensors.quantization import (
    QuantizationArgs,
    QuantizationStrategy,
    QuantizationType,
)

from vllm.model_executor.layers.fused_moe.oracle.int_wna16 import (
    WNA16MoEBackend,
    convert_to_wna16_moe_kernel_format,
)
from vllm.model_executor.layers.quantization.auto_gptq import AutoGPTQConfig

E = 4  # num experts
K = 512  # hidden size
N = 256  # intermediate size (per gate/up half of w13; also w2's output dim)
GROUP_SIZE = 64
PACK = 8  # int4 values packed per int32


def _autogptq_config() -> AutoGPTQConfig:
    return AutoGPTQConfig(
        weight_bits=4,
        group_size=GROUP_SIZE,
        desc_act=False,
        is_sym=True,  # AutoGPTQConfig only supports symmetric at the top
        # level, but AutoGPTQ checkpoints still ship a qzeros tensor
        # regardless of symmetry.
        lm_head_quantized=False,
        dynamic={},
        full_config={},
    )


def _compressed_tensors_asym_config() -> QuantizationArgs:
    """compressed-tensors pack-quantized WNA16, asymmetric: the qzeros values
    carry real per-group offsets, so the reshape must preserve them exactly."""
    return QuantizationArgs(
        num_bits=4,
        type=QuantizationType.INT,
        symmetric=False,
        strategy=QuantizationStrategy.GROUP,
        group_size=GROUP_SIZE,
    )


QUANT_CONFIGS = [
    pytest.param(_autogptq_config, id="autogptq-sym"),
    pytest.param(_compressed_tensors_asym_config, id="compressed-tensors-asym"),
]


def _make_k_first_weights(seed: int = 0):
    """K-first packed layout, as produced by AutoGPTQ / compressed-tensors
    checkpoint loaders before any backend-specific post-processing."""
    g = torch.Generator().manual_seed(seed)
    hi = 2**31 - 1
    w13 = torch.randint(0, hi, (E, K // PACK, 2 * N), dtype=torch.int32, generator=g)
    w2 = torch.randint(0, hi, (E, N // PACK, K), dtype=torch.int32, generator=g)
    w13_scale = torch.rand(
        (E, K // GROUP_SIZE, 2 * N), dtype=torch.float16, generator=g
    )
    w2_scale = torch.rand((E, N // GROUP_SIZE, K), dtype=torch.float16, generator=g)
    w13_qzeros = torch.randint(
        0, hi, (E, K // GROUP_SIZE, 2 * N // PACK), dtype=torch.int32, generator=g
    )
    w2_qzeros = torch.randint(
        0, hi, (E, N // GROUP_SIZE, K // PACK), dtype=torch.int32, generator=g
    )
    return w13, w2, w13_scale, w2_scale, w13_qzeros, w2_qzeros


def _ref_qzeros_kernel_layout(qz_k_first: torch.Tensor) -> torch.Tensor:
    """Independent reference for the K-first int32 -> N-first uint8 qzeros
    transform, written at the zero-point-value level rather than as a byte
    shuffle.

    Input  ``(E, K // gs, N // PACK)`` int32: element ``[e, kg, p]`` packs the
    int4 zero-points for output channels ``[p * PACK, p * PACK + PACK)`` at
    K-group ``kg``, LSB-first.

    Output ``(E, N // 2, K // gs)`` uint8: ``fused_moe_kernel_gptq_awq`` reads
    ``out[e, bn // 2, kg]`` as the byte whose low nibble is the zero-point for
    output channel ``bn`` and whose high nibble is channel ``bn + 1``.
    """
    n_experts, n_kgroups, n_packed = qz_k_first.shape
    n_channels = n_packed * PACK

    bytes_u8 = qz_k_first.view(torch.uint8).reshape(n_experts, n_kgroups, n_packed, 4)
    low = bytes_u8 & 0xF
    high = (bytes_u8 >> 4) & 0xF
    # interleave low/high nibble per byte -> per-channel zero-points
    zp = torch.stack([low, high], dim=-1).reshape(n_experts, n_kgroups, n_channels)

    pairs = zp.reshape(n_experts, n_kgroups, n_channels // 2, 2)
    packed = (pairs[..., 0] | (pairs[..., 1] << 4)).to(torch.uint8)
    return packed.transpose(1, 2).contiguous()  # (E, N // 2, K // gs)


def _run(quant_config, *, with_qzeros: bool):
    w13, w2, w13_scale, w2_scale, w13_qzeros, w2_qzeros = _make_k_first_weights()
    result = convert_to_wna16_moe_kernel_format(
        backend=WNA16MoEBackend.TRITON,
        layer=torch.nn.Module(),
        quant_config=quant_config,
        input_dtype=torch.float16,
        w13=w13,
        w2=w2,
        w13_scale=w13_scale,
        w2_scale=w2_scale,
        w13_qzeros=w13_qzeros if with_qzeros else None,
        w2_qzeros=w2_qzeros if with_qzeros else None,
    )
    assert result is not None
    (_, _, _, _, _, _, _, _, w13_qz_out, w2_qz_out, *_rest) = result
    return (w13_qzeros, w2_qzeros), (w13_qz_out, w2_qz_out)


@pytest.mark.parametrize("make_config", QUANT_CONFIGS)
def test_triton_qzeros_reshaped_to_kernel_expected_layout(make_config):
    (w13_qzeros, w2_qzeros), (w13_qz_out, w2_qz_out) = _run(
        make_config(), with_qzeros=True
    )

    # fused_moe_kernel_gptq_awq expects N-first uint8, 2 packed int4 zero
    # points per byte: (E, N // 2, K // group_size).
    assert w13_qz_out.shape == (E, (2 * N) // 2, K // GROUP_SIZE)
    assert w2_qz_out.shape == (E, K // 2, N // GROUP_SIZE)
    assert w13_qz_out.dtype == torch.uint8
    assert w2_qz_out.dtype == torch.uint8

    # ...and the transform maps zero-point k to output channel k, verified
    # against an independently constructed expected tensor.
    torch.testing.assert_close(
        w13_qz_out, _ref_qzeros_kernel_layout(w13_qzeros), rtol=0, atol=0
    )
    torch.testing.assert_close(
        w2_qz_out, _ref_qzeros_kernel_layout(w2_qzeros), rtol=0, atol=0
    )


@pytest.mark.parametrize("make_config", QUANT_CONFIGS)
def test_triton_qzeros_reshape_is_none_safe(make_config):
    """A checkpoint that ships no zero-point tensor at all must not be forced
    through the reshape path (it would crash on `None`)."""
    _, (w13_qz_out, w2_qz_out) = _run(make_config(), with_qzeros=False)
    assert w13_qz_out is None
    assert w2_qz_out is None
