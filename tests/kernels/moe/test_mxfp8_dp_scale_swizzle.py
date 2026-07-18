# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Regression test for the mxfp8 activation-scale layout in the naive DP/EP
prepare path.

The naive DP/EP prepare quantizes activations with row-major ([M, K//32])
scales so the scale tensor keeps one row per token and can be dispatched
alongside the hidden states. Kernels that declare
``is_scale_swizzled=True`` (e.g. FlashInfer CUTLASS MXFP4_MXFP8) expect the
F8_128x4 interleaved layout, so ``_unwrap_scale_and_prepare_for_moe`` must
restore it after the a2a — exactly like the pre-existing nvfp4 branch.
Without this, the CUTLASS kernel reads misplaced e8m0 block exponents and
produces NaN logits whenever ``data_parallel_size > 1``.
"""

import pytest
import torch

from vllm.model_executor.layers.fused_moe.config import (
    mxfp4_mxfp8_moe_quant_config,
)
from vllm.model_executor.layers.fused_moe.prepare_finalize.naive_dp_ep import (
    _unwrap_scale_and_prepare_for_moe,
)
from vllm.model_executor.layers.quantization.utils.mxfp8_utils import (
    mxfp8_e4m3_quantize,
)
from vllm.platforms import current_platform

if not current_platform.is_cuda():
    pytest.skip("CUDA required", allow_module_level=True)


@pytest.mark.parametrize("num_tokens", [1, 5, 37, 128, 300, 511])
@pytest.mark.parametrize("hidden_size", [2048, 7168])
@torch.inference_mode()
def test_unwrap_restores_swizzled_mxfp8_scales(
    num_tokens: int, hidden_size: int
) -> None:
    torch.manual_seed(0)
    x = torch.randn(num_tokens, hidden_size, device="cuda", dtype=torch.bfloat16)

    # The layout the kernel expects (what the no-DP path produces directly).
    _, want = mxfp8_e4m3_quantize(x, is_sf_swizzled_layout=True)
    # The layout the naive DP path quantizes to before the dispatch.
    _, linear = mxfp8_e4m3_quantize(x, is_sf_swizzled_layout=False)
    if linear.ndim == 1:
        linear = linear.view(num_tokens, -1)

    quant_config = mxfp4_mxfp8_moe_quant_config(
        w1_scale=torch.empty(1),
        w2_scale=torch.empty(1),
        is_scale_swizzled=True,
    )
    got = _unwrap_scale_and_prepare_for_moe([linear], quant_config)

    assert torch.equal(
        got.view(torch.uint8).flatten(), want.view(torch.uint8).flatten()
    )
