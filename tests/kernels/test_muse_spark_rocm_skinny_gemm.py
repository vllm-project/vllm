# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""fp32-accumulating split-K GEMM for the TP-sharded MuseSpark1x projections."""

import pytest
import torch
import torch.nn.functional as F

from vllm.models.muse_spark.amd.ops.skinny_gemm import (
    _CONFIGS,
    skinny_gemm,
    skinny_gemm_config,
)
from vllm.platforms import current_platform

pytestmark = pytest.mark.skipif(
    not current_platform.is_rocm(), reason="requires a ROCm GPU"
)

# Pooler and unpooler column shards at TP8.
SHAPES = [(512, 8192), (1024, 4096)]


def _bf16_ulp(t: torch.Tensor) -> torch.Tensor:
    """One BF16 unit in the last place at each element of ``t``."""
    exponent = torch.frexp(t.float())[1]
    return torch.ldexp(torch.ones_like(t, dtype=torch.float32), exponent - 8)


def test_config_lookup():
    for (n, k), rows in _CONFIGS.items():
        for max_tokens, *_ in rows:
            assert skinny_gemm_config(n, k, max_tokens) is not None
        assert skinny_gemm_config(n, k, rows[-1][0] + 1) is None
        for max_tokens, _, block_n, block_k, splits, _ in rows:
            assert n % block_n == 0 and k % (splits * block_k) == 0
    assert skinny_gemm_config(4096, 8192, 1) is None


@pytest.mark.parametrize("n,k", SHAPES)
@pytest.mark.parametrize("num_tokens", [1, 2, 7, 8, 16, 33, 64, 100, 128])
def test_skinny_gemm_rounds_the_fp32_sum_once(n, k, num_tokens):
    """Each output is the fp32 sum rounded to BF16 once, like an fp32-accumulating
    ``F.linear``: within one BF16 ulp (plus the fp32 summation error) of an fp64
    reference, and almost always equal to that reference rounded to BF16."""
    torch.manual_seed(num_tokens * n)
    x = torch.randn(num_tokens, k, device="cuda", dtype=torch.bfloat16)
    w = (torch.randn(n, k, device="cuda") * 0.02).to(torch.bfloat16)

    out = skinny_gemm(x, w)
    ref = x.double() @ w.double().t()
    assert out.dtype == torch.bfloat16 and out.shape == (num_tokens, n)

    # fp32 summation error, far below what one BF16-rounded partial would add.
    fp32_err = 2**-20 * (x.double().abs() @ w.double().abs().t())
    bound = _bf16_ulp(ref).double() + fp32_err
    linear = F.linear(x, w)
    for y in (out, linear):
        assert ((y.double() - ref).abs() <= bound).all()
        assert (y != ref.bfloat16()).float().mean().item() < 1e-2
    # Same inputs, same order of the fixed split reduction.
    assert torch.equal(out, skinny_gemm(x, w))


@pytest.mark.parametrize("n,k", SHAPES)
def test_skinny_gemm_does_not_round_partials_to_bf16(n, k):
    """Large partial sums that cancel: a partial rounded to BF16 (8 significant
    bits) would drop the remainder of 12 that fp32 keeps exactly."""
    x = torch.ones(4, k, device="cuda", dtype=torch.bfloat16)
    w = torch.empty(n, k, device="cuda", dtype=torch.bfloat16)
    half = k // 2
    w[:, :half] = 1024.0
    w[:, half:] = -1024.0
    w[:, 0] = 1032.0
    w[:, -1] = -1020.0
    out = skinny_gemm(x, w)
    assert torch.equal(out, torch.full_like(out, 12.0))


def test_skinny_gemm_strided_rows_and_empty():
    n, k = 1024, 4096
    w = torch.randn(n, k, device="cuda", dtype=torch.bfloat16)
    base = torch.randn(16, 2 * k, device="cuda", dtype=torch.bfloat16)
    x = base[:, :k]
    assert not x.is_contiguous() and x.stride(1) == 1
    assert torch.equal(skinny_gemm(x, w), skinny_gemm(x.contiguous(), w))
    assert skinny_gemm(x[:0], w).shape == (0, n)


@pytest.mark.parametrize("n,k", SHAPES)
def test_skinny_gemm_replays_in_a_graph(n, k):
    x = torch.randn(8, k, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(n, k, device="cuda", dtype=torch.bfloat16)
    expected = skinny_gemm(x, w)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = skinny_gemm(x, w)
    x.copy_(torch.randn_like(x))
    expected = skinny_gemm(x, w)
    graph.replay()
    assert torch.equal(out, expected)
