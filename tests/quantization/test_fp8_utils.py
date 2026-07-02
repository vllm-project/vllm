# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.model_executor.layers.quantization.utils.fp8_utils import (
    _upcast_e8m0_scales_for_triton,
)


@pytest.mark.skipif(
    not hasattr(torch, "float8_e8m0fnu"),
    reason="torch.float8_e8m0fnu is unavailable in this PyTorch build.",
)
def test_upcast_e8m0_scales_for_triton_decodes_only_e8m0_scales():
    e8m0 = torch.tensor([1.0, 2.0], dtype=torch.float32).to(torch.float8_e8m0fnu)
    fp32 = torch.tensor([3.0, 4.0], dtype=torch.float32)

    decoded_e8m0, unchanged_fp32 = _upcast_e8m0_scales_for_triton(e8m0, fp32)

    assert decoded_e8m0.dtype == torch.float32
    assert decoded_e8m0.is_contiguous()
    torch.testing.assert_close(decoded_e8m0, torch.tensor([1.0, 2.0]))
    assert unchanged_fp32 is fp32
