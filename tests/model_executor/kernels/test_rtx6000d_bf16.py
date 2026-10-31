# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU dispatch contracts; GPU arithmetic is covered by test_ll_bf16_gemm."""

import pytest
import torch
import torch.nn.functional as F

from vllm.model_executor.kernels.linear.cute_dsl import rtx6000d_bf16 as kernel


@pytest.mark.parametrize("shape", [(0, 16), (1, 16), (4, 16), (8, 16), (2, 3, 16)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_unprepared_projection_preserves_linear(shape, dtype):
    x = torch.randn(shape, dtype=dtype)
    weight = torch.randn(8, 16, dtype=dtype)
    torch.testing.assert_close(
        kernel.rtx6000d_bf16_linear(x, weight), F.linear(x, weight)
    )


def test_cpu_weight_never_initializes_cuda(monkeypatch):
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda *args: pytest.fail("CPU fallback must not initialize CUDA"),
    )
    layer = torch.nn.Linear(2048, 4096, bias=False, dtype=torch.bfloat16)
    layer.prefix = "model.layers.0.self_attn.o_proj"
    assert not kernel.prepare_rtx6000d_bf16(layer)


def test_fake_preserves_leading_dimensions():
    x = torch.empty(2, 3, 16, device="meta", dtype=torch.bfloat16)
    weight = torch.empty(8, 16, device="meta", dtype=torch.bfloat16)
    output = kernel._fake(x, weight)
    assert output.shape == (2, 3, 8)
    assert output.dtype == x.dtype
