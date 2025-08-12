# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# DeepGEMM Style Cutlass Grouped GEMM Test
# See https://github.com/deepseek-ai/DeepGEMM/blob/main/tests/test_core.py

import random

import pytest
import torch

from tests.kernels.utils import baseline_scaled_mm
from vllm import _custom_ops as ops
from vllm.platforms import current_platform


def cdiv(a, b):
    return (a + b - 1) // b


def per_token_cast_to_fp8(
        x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    assert x.dim() == 2
    m, n = x.shape
    pad_size = (128 - (n % 128)) % 128
    x = torch.nn.functional.pad(x,
                                (0, pad_size), value=0) if pad_size > 0 else x
    x_view = x.view(m, -1, 128)
    x_amax = x_view.abs().float().amax(dim=2).view(m, -1).clamp(1e-4)
    fp8_data = (x_view *
                (448.0 / x_amax.unsqueeze(2))).to(dtype=torch.float8_e4m3fn)
    return fp8_data.view(m, n + pad_size)[:, :n], (x_amax / 448.0).view(m, -1)


def per_block_cast_to_fp8(
        x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    assert x.dim() == 2
    m, n = x.shape
    x_padded = torch.zeros((cdiv(m, 128) * 128, cdiv(n, 128) * 128),
                           device=x.device,
                           dtype=x.dtype)
    x_padded[:m, :n] = x
    x_view = x_padded.view(-1, 128, x_padded.size(1) // 128, 128)
    x_amax = x_view.abs().float().amax(dim=(1, 3), keepdim=True).clamp(1e-4)
    x_scaled = (x_view * (448.0 / x_amax)).to(dtype=torch.float8_e4m3fn)
    return x_scaled.view_as(x_padded)[:m, :n].contiguous(), (
        x_amax / 448.0).view(x_view.size(0), x_view.size(2))


@pytest.mark.parametrize("num_groups, expected_m_per_group, k, n", [
    (4, 8192, 7168, 4096),
    (4, 8192, 2048, 7168),
    (8, 4096, 7168, 4096),
    (8, 4096, 2048, 7168),
    (32, 1024, 7168, 4096),
    (32, 1024, 2048, 7168),
])
@pytest.mark.parametrize("out_dtype", [torch.float16])
@pytest.mark.skipif(
    (lambda x: x is None or x.to_int() != 100)(
        current_platform.get_device_capability()),
    reason="Block Scaled Grouped GEMM is only supported on SM100.")
def test_cutlass_grouped_gemm(
    num_groups: int,
    expected_m_per_group: int,
    k: int,
    n: int,
    out_dtype: torch.dtype,
):
    device = "cuda"
    alignment = 128
    group_ms = [
        int(expected_m_per_group * random.uniform(0.7, 1.3))
        for _ in range(num_groups)
    ]
    m = sum([cdiv(m, alignment) * alignment for m in group_ms])

    x = torch.randn((m, k), device=device, dtype=out_dtype)
    y = torch.randn((num_groups, n, k), device=device, dtype=out_dtype)
    out = torch.empty((m, n), device=device, dtype=out_dtype)
    ref_out = torch.randn((m, n), device=device, dtype=out_dtype)

    ep_offset = [0] + [sum(group_ms[:i]) for i in range(1, num_groups)] + [m]
    pb_size = []
    for i in range(num_groups):
        pb_size.append([ep_offset[i + 1] - ep_offset[i], n, k])
    problem_sizes = torch.tensor(pb_size, device=device, dtype=torch.int32)
    expert_offsets = torch.tensor(ep_offset, device=device, dtype=torch.int32)

    x_fp8 = per_token_cast_to_fp8(x)
    y_fp8 = (torch.empty_like(y, dtype=torch.float8_e4m3fn),
             torch.empty((num_groups, cdiv(n, 128), k // 128),
                         device=device,
                         dtype=torch.float))
    for i in range(num_groups):
        y_fp8[0][i], y_fp8[1][i] = per_block_cast_to_fp8(y[i])

    for i in range(num_groups):
        a = x_fp8[0][ep_offset[i]:ep_offset[i + 1]]
        a_scale = x_fp8[1][ep_offset[i]:ep_offset[i + 1]]
        b = y_fp8[0][i].t()
        b_scale = y_fp8[1][i].t()
        baseline = baseline_scaled_mm(a, b, a_scale, b_scale, out_dtype)
        ref_out[ep_offset[i]:ep_offset[i + 1]] = baseline

    ops.cutlass_blockwise_scaled_grouped_mm(
        out,
        x_fp8[0],
        y_fp8[0],
        x_fp8[1],
        y_fp8[1],
        problem_sizes,
        expert_offsets[:-1],
    )

    torch.testing.assert_close(ref_out, out, atol=5e-1, rtol=1e-3)
