# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from vllm.models.glm5next.cpu.sparse_indexer import (
    _expand_pool_ids,
    _pool_compress,
    _quantize_cache_vector,
    fwht128_quant_fp8,
)


def _reference_fwht(x: torch.Tensor) -> torch.Tensor:
    y = x.float()
    width = 1
    while width < 128:
        grouped = y.reshape(-1, 128 // (2 * width), 2, width)
        a, b = grouped.unbind(dim=2)
        y = torch.stack((a + b, a - b), dim=2).reshape(-1, 128)
        width *= 2
    return y * (128.0**-0.5)


def test_cpu_fwht_quant_uses_reference_transform():
    x = torch.randn(3, 128, dtype=torch.bfloat16)
    quant, scale = fwht128_quant_fp8(x)

    expected = _reference_fwht(x).to(torch.bfloat16).float()
    dequant = quant.float() * scale
    torch.testing.assert_close(dequant, expected, atol=1.5, rtol=0.08)
    assert quant.dtype == torch.float8_e4m3fn
    assert scale.shape == (3, 1)


def test_cpu_pool_compress_matches_independent_softmax_reference():
    keys = torch.randn(4, 128, dtype=torch.bfloat16)
    gate = torch.randn(4, 128)
    ape = torch.randn(4, 128)

    actual = _pool_compress(keys, gate, ape)
    probs = torch.softmax(gate + ape, dim=0)
    expected = (keys.float() * probs).sum(dim=0).to(keys.dtype)
    torch.testing.assert_close(actual, expected)


def test_cpu_cache_quantization_returns_glm_record():
    values, scale = _quantize_cache_vector(torch.randn(128))

    assert values.dtype == torch.uint8
    assert values.shape == (128,)
    assert scale.shape == ()
    assert torch.isfinite(scale)


@pytest.mark.parametrize(
    ("pool_ids", "seq_len", "expected"),
    [
        ([[0]], 3, [[0, 1, 2, -1, -1, -1, -1, -1]]),
        ([[0]], 5, [[0, 1, 2, 3, 4, -1, -1, -1]]),
    ],
)
def test_pool_to_token_expansion_appends_only_valid_tail(
    pool_ids, seq_len, expected
):
    ids = torch.tensor(pool_ids, dtype=torch.int32)
    actual = _expand_pool_ids(ids, torch.tensor([seq_len]), 3, max_tokens=8)
    torch.testing.assert_close(actual, torch.tensor(expected, dtype=torch.int32))
