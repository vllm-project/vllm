# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.kernels.triton.topk import supports_triton_topk, triton_topk
from vllm.platforms import current_platform

pytestmark = pytest.mark.skipif(
    not current_platform.is_cuda_alike(), reason="requires a GPU"
)


def _reference(x: torch.Tensor, k: int) -> tuple[torch.Tensor, torch.Tensor]:
    # A stable descending sort orders ties by the lowest index.
    values, indices = torch.sort(x.float(), dim=-1, descending=True, stable=True)
    return values[:, :k].to(x.dtype), indices[:, :k]


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize("num_rows", [1, 7, 33, 448])
@pytest.mark.parametrize("vocab_size", [4096, 151936, 248320])
@pytest.mark.parametrize("k", [1, 8, 16, 20, 64])
def test_matches_stable_sort(dtype, num_rows, vocab_size, k):
    torch.manual_seed(0)
    x = torch.randn(num_rows, vocab_size, dtype=dtype, device="cuda")
    assert supports_triton_topk(x, k)
    values, indices = triton_topk(x, k)
    ref_values, ref_indices = _reference(x, k)
    assert indices.dtype == torch.int64
    torch.testing.assert_close(values, ref_values, rtol=0, atol=0)
    torch.testing.assert_close(indices, ref_indices, rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_ties_padding_and_strides(dtype):
    torch.manual_seed(0)
    vocab_size, k = 248320, 16
    # Coarse values force many ties, including across chunks and groups.
    x = torch.randint(-4, 4, (64, vocab_size + 128), device="cuda").to(dtype)
    x[:, vocab_size - 64 : vocab_size] = float("-inf")
    # Rows with fewer than k finite values fill the rest with -inf.
    x[0, :vocab_size] = float("-inf")
    x[0, [5, 70000, 248000]] = torch.tensor([1.0, 3.0, 2.0], dtype=dtype, device="cuda")
    x[1] = float("-inf")
    x[2, 123456] = float("inf")
    x = x[:, :vocab_size]
    values, indices = triton_topk(x, k)
    ref_values, ref_indices = _reference(x, k)
    torch.testing.assert_close(values, ref_values, rtol=0, atol=0)
    torch.testing.assert_close(indices, ref_indices, rtol=0, atol=0)
    assert indices[0, :3].tolist() == [70000, 248000, 5]
    assert indices[1].tolist() == list(range(k))


def test_negative_values_and_empty():
    x = -torch.rand(5, 8192, device="cuda") - 1.0
    values, indices = triton_topk(x, 4)
    ref_values, ref_indices = _reference(x, 4)
    torch.testing.assert_close(values, ref_values, rtol=0, atol=0)
    torch.testing.assert_close(indices, ref_indices, rtol=0, atol=0)

    values, indices = triton_topk(x[:0], 4)
    assert values.shape == (0, 4) and indices.shape == (0, 4)


def test_unsupported_shapes_fall_back():
    x = torch.randn(4, 1024, device="cuda")
    assert not supports_triton_topk(x, 8)
    assert not supports_triton_topk(torch.randn(4, 8192, device="cuda"), 128)
    assert not supports_triton_topk(torch.randn(8192, 4, device="cuda").t(), 2)
