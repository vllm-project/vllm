# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm._aiter_ops import rocm_aiter_ops
from vllm.model_executor.layers.logits_processor import _topk
from vllm.platforms import current_platform

pytestmark = pytest.mark.skipif(not current_platform.is_rocm(), reason="requires ROCm")


@pytest.fixture(autouse=True)
def aiter_enabled(monkeypatch):
    monkeypatch.setenv("VLLM_ROCM_USE_AITER", "1")
    rocm_aiter_ops.refresh_env_variables()
    if not rocm_aiter_ops.is_vocab_top_k_enabled():
        pytest.skip("requires aiter with the multi-block topk_plain fix")
    yield
    monkeypatch.undo()
    rocm_aiter_ops.refresh_env_variables()


def _check(x: torch.Tensor, k: int, values: torch.Tensor, indices: torch.Tensor):
    # Ties may pick any tied index, so compare values against a sort and check
    # that the indices point at them without repeats.
    ref = torch.sort(x.float(), dim=-1, descending=True).values[:, :k].to(x.dtype)
    torch.testing.assert_close(values, ref, rtol=0, atol=0)
    indices = indices.long()
    assert ((indices >= 0) & (indices < x.shape[-1])).all()
    torch.testing.assert_close(x.gather(-1, indices), values, rtol=0, atol=0)
    assert not (indices.sort(dim=-1).values.diff(dim=-1) == 0).any()


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize("num_rows", [1, 7, 33, 448])
@pytest.mark.parametrize("vocab_size", [4096, 151936, 248320])
@pytest.mark.parametrize("k", [1, 8, 16, 64, 128])
def test_matches_sort(dtype, num_rows, vocab_size, k):
    torch.manual_seed(0)
    x = torch.randn(num_rows, vocab_size, dtype=dtype, device="cuda")
    assert rocm_aiter_ops.supports_vocab_top_k(x, k)
    values, indices = rocm_aiter_ops.vocab_top_k(x, k)
    _check(x, k, values, indices)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("num_rows", [7, 448])
def test_ties_and_padding(dtype, num_rows):
    torch.manual_seed(0)
    vocab_size, k = 248320, 16
    # Coarse values force many ties, including across blocks.
    x = torch.randint(-4, 4, (num_rows, vocab_size), device="cuda").to(dtype)
    x[:, -64:] = float("-inf")
    x[1, 123456] = float("inf")
    values, indices = rocm_aiter_ops.vocab_top_k(x, k)
    _check(x, k, values, indices)


def test_rows_with_fewer_than_k_finite_values():
    vocab_size, k = 248320, 16
    x = torch.randn(7, vocab_size, device="cuda")
    x[0] = float("-inf")
    x[0, [5, 70000, 248000]] = torch.tensor([1.0, 3.0, 2.0], device="cuda")
    x[1] = float("-inf")
    values, indices = rocm_aiter_ops.vocab_top_k(x, k)
    assert ((indices >= 0) & (indices < vocab_size)).all()
    assert indices[0, :3].tolist() == [70000, 248000, 5]
    assert values[0, :3].tolist() == [3.0, 2.0, 1.0]
    assert values[0, 3:].eq(float("-inf")).all() and values[1].eq(float("-inf")).all()
    _check(x[2:], k, values[2:], indices[2:])


def test_empty_and_unsupported():
    values, indices = rocm_aiter_ops.vocab_top_k(torch.empty(0, 8192, device="cuda"), 4)
    assert values.shape == (0, 4) and indices.shape == (0, 4)
    assert not rocm_aiter_ops.supports_vocab_top_k(torch.randn(4, 1024), 8)
    assert not rocm_aiter_ops.supports_vocab_top_k(torch.randn(4, 8192), 256)
    assert not rocm_aiter_ops.supports_vocab_top_k(torch.randn(8192, 4).t(), 2)


def test_logits_processor_topk_uses_aiter():
    torch.manual_seed(0)
    x = torch.randn(112, 248320, dtype=torch.bfloat16, device="cuda")
    values, indices = _topk(x, 16)
    assert indices.dtype == torch.int32
    _check(x, 16, values, indices)
