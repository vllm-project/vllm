# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Regression coverage for the AutoGPTQ/compressed-tensors qzeros reshape in
`convert_to_wna16_moe_kernel_format`'s TRITON backend branch.

Locks in that w13/w2 zero-points get transposed and reshaped from the
K-first AutoGPTQ layout into the N-first, packed-uint8 layout
`fused_moe_kernel_gptq_awq` expects -- the weights and scales already get
this transform; this test ensures the zero-points do too.

Pure tensor-shape/dtype test, no GPU or Triton kernel launch involved.
"""

import torch

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


def _make_quant_config() -> AutoGPTQConfig:
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


def _make_k_first_weights():
    """K-first packed layout, as produced by AutoGPTQ/compressed-tensors
    checkpoint loaders before any backend-specific post-processing."""
    w13 = torch.randint(0, 2**31 - 1, (E, K // PACK, 2 * N), dtype=torch.int32)
    w2 = torch.randint(0, 2**31 - 1, (E, N // PACK, K), dtype=torch.int32)
    w13_scale = torch.rand((E, K // GROUP_SIZE, 2 * N), dtype=torch.float16)
    w2_scale = torch.rand((E, N // GROUP_SIZE, K), dtype=torch.float16)
    w13_qzeros = torch.randint(
        0, 2**31 - 1, (E, K // GROUP_SIZE, 2 * N // PACK), dtype=torch.int32
    )
    w2_qzeros = torch.randint(
        0, 2**31 - 1, (E, N // GROUP_SIZE, K // PACK), dtype=torch.int32
    )
    return w13, w2, w13_scale, w2_scale, w13_qzeros, w2_qzeros


def test_gptq_triton_qzeros_reshaped_to_kernel_expected_layout():
    w13, w2, w13_scale, w2_scale, w13_qzeros, w2_qzeros = _make_k_first_weights()

    result = convert_to_wna16_moe_kernel_format(
        backend=WNA16MoEBackend.TRITON,
        layer=torch.nn.Module(),
        quant_config=_make_quant_config(),
        input_dtype=torch.float16,
        w13=w13,
        w2=w2,
        w13_scale=w13_scale,
        w2_scale=w2_scale,
        w13_qzeros=w13_qzeros,
        w2_qzeros=w2_qzeros,
    )
    assert result is not None
    (_, _, _, _, _, _, _, _, w13_qz_out, w2_qz_out, *_rest) = result

    # fused_moe_kernel_gptq_awq expects N-first uint8, 2 packed int4 zero
    # points per byte: (E, N // 2, K // group_size).
    assert w13_qz_out.shape == (E, (2 * N) // 2, K // GROUP_SIZE)
    assert w2_qz_out.shape == (E, K // 2, N // GROUP_SIZE)
    assert w13_qz_out.dtype == torch.uint8
    assert w2_qz_out.dtype == torch.uint8


def test_gptq_triton_qzeros_reshape_is_none_safe():
    """Symmetric-format checkpoints with no zero-point tensor at all must
    not be forced through the reshape path (it would crash on `None`)."""
    w13, w2, w13_scale, w2_scale, _, _ = _make_k_first_weights()

    result = convert_to_wna16_moe_kernel_format(
        backend=WNA16MoEBackend.TRITON,
        layer=torch.nn.Module(),
        quant_config=_make_quant_config(),
        input_dtype=torch.float16,
        w13=w13,
        w2=w2,
        w13_scale=w13_scale,
        w2_scale=w2_scale,
        w13_qzeros=None,
        w2_qzeros=None,
    )
    assert result is not None
    (_, _, _, _, _, _, _, _, w13_qz_out, w2_qz_out, *_rest) = result
    assert w13_qz_out is None
    assert w2_qz_out is None
