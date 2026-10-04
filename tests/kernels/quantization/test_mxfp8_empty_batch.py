# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.model_executor.layers.quantization.utils.mxfp8_utils import (
    mxfp8_e4m3_quantize,
)


@pytest.mark.parametrize("is_sf_swizzled_layout", [False, True])
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_mxfp8_quantize_returns_empty_batch(is_sf_swizzled_layout: bool):
    x = torch.empty(0, 128, dtype=torch.bfloat16, device="cuda")

    x_q, scales = mxfp8_e4m3_quantize(x, is_sf_swizzled_layout=is_sf_swizzled_layout)

    assert x_q.shape == (0, 128)
    assert x_q.dtype == torch.float8_e4m3fn
    if is_sf_swizzled_layout:
        assert scales.shape == (0,)
    else:
        assert scales.shape == (0, 4)
