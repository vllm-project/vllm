# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for restore_dispatched_scale_layout on mxfp8 activation scales.

The DP/EP prepare paths quantize with row-major scales for the a2a and
restore the swizzled 128x4 layout afterwards; the result must be
bit-identical to quantizing directly into the swizzled layout, and scales
for kernels that take row-major scales must pass through unchanged.
"""

import pytest
import torch

from vllm.model_executor.layers.fused_moe.config import (
    mxfp4_mxfp8_moe_quant_config,
)
from vllm.model_executor.layers.fused_moe.prepare_finalize.naive_dp_ep import (
    _unwrap_scale_and_prepare_for_moe,
)
from vllm.model_executor.layers.fused_moe.utils import (
    restore_dispatched_scale_layout,
)
from vllm.model_executor.layers.quantization.utils.mxfp8_utils import (
    mxfp8_e4m3_quantize,
)
from vllm.platforms import current_platform
from vllm.utils.flashinfer import has_flashinfer

if not current_platform.is_cuda():
    pytest.skip("CUDA required", allow_module_level=True)

if not has_flashinfer():
    # restore_dispatched_scale_layout uses flashinfer's block_scale_interleave.
    pytest.skip("flashinfer required", allow_module_level=True)


def _quantize_both_layouts(
    num_tokens: int, hidden_size: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (swizzled reference, row-major) mxfp8 scales for one input."""
    torch.manual_seed(0)
    x = torch.randn(num_tokens, hidden_size, device="cuda", dtype=torch.bfloat16)

    # The layout the kernel expects (what the no-DP path produces directly).
    _, want = mxfp8_e4m3_quantize(x, is_sf_swizzled_layout=True)
    # The layout the DP paths quantize to before the dispatch.
    _, linear = mxfp8_e4m3_quantize(x, is_sf_swizzled_layout=False)
    return want, linear


@pytest.mark.parametrize("num_tokens", [1, 5, 37, 128, 300, 511])
@pytest.mark.parametrize("hidden_size", [2048, 7168])
@torch.inference_mode()
def test_swizzle_dispatched_mxfp8_scales(num_tokens: int, hidden_size: int):
    want, linear = _quantize_both_layouts(num_tokens, hidden_size)

    got = restore_dispatched_scale_layout(
        linear, quant_dtype="mxfp8", is_scale_swizzled=True
    )

    assert got is not None
    assert torch.equal(
        got.view(torch.uint8).flatten(), want.view(torch.uint8).flatten()
    )

    # Unswizzled consumers (e.g. FLASHINFER_TRTLLM_MXFP4_MXFP8) are untouched.
    passthrough = restore_dispatched_scale_layout(
        linear, quant_dtype="mxfp8", is_scale_swizzled=False
    )
    assert passthrough is linear


@torch.inference_mode()
def test_naive_dp_ep_unwrap_swizzles_mxfp8_scales():
    """The naive DP/EP prepare path must hand the MoE kernel swizzled scales."""
    want, linear = _quantize_both_layouts(300, 7168)

    quant_config = mxfp4_mxfp8_moe_quant_config(
        w1_scale=torch.empty(1),
        w2_scale=torch.empty(1),
        is_scale_swizzled=True,
    )
    got = _unwrap_scale_and_prepare_for_moe([linear], quant_config)

    assert torch.equal(
        got.view(torch.uint8).flatten(), want.view(torch.uint8).flatten()
    )
