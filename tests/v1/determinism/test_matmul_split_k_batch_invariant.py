# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The split-K path of matmul_persistent (skinny-N / long-K linears, e.g. the
DSv4 indexer's M x 4096 @ 4096 x 64): correctness against an fp64 reference and
batch invariance across M, row order and repeats. The split plan and the k-tile
size are functions of (K, N) only, so a row must come out bitwise identical
whatever batch it is computed in, even though the tile height and the
fused/two-kernel reduction follow M."""

import pytest
import torch
from utils import skip_unsupported

from vllm.model_executor.determinism import batch_invariant as bi
from vllm.platforms import current_platform

DEVICE_TYPE = current_platform.device_type


def _operands(M, K, N, dtype, seed=0):
    g = torch.Generator(device=DEVICE_TYPE).manual_seed(seed)
    x = torch.randn(M, K, device=DEVICE_TYPE, generator=g).to(dtype)
    w = (torch.randn(N, K, device=DEVICE_TYPE, generator=g) * 0.02).to(dtype)
    return x, w.t()


def test_split_k_plan_depends_only_on_k_and_n():
    """What is summed comes from (K, N); M picks none of it."""
    assert bi._split_k_plan(4096, 64) == (8, 512)
    assert bi._split_k_plan(65536, 64) == (16, 4096)
    assert bi._split_k_plan(1024, 64) == (2, 512)
    assert bi._split_k_plan(1023, 64)[0] == 1  # K < 1024: no split
    assert bi._split_k_plan(4096, 65)[0] == 1  # N > 64: one pass over K
    assert bi._split_k_plan(4096, 4096)[0] == 1
    assert bi._block_size_k(torch.float32) == 32
    assert bi._block_size_k(torch.bfloat16) == bi._block_size_k(torch.float16) == 64


@skip_unsupported
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize(
    ("M", "K", "N"),
    [
        (1, 4096, 64),  # weights_proj decode
        (64, 4096, 64),
        (4096, 4096, 64),
        (17, 4104, 64),  # K tail inside the last chunk
        (33, 4096, 63),  # N tail
        (3, 1000, 129),  # K < 1024 -> no split
        (5, 4096, 129280),  # lm_head: wide N -> no split
        (16, 1, 1),
        (0, 4096, 64),
        (7, 0, 9),
    ],
)
def test_matmul_persistent_matches_fp64_reference(M, K, N, dtype):
    x, b = _operands(M, K, N, dtype)
    g = torch.Generator(device=DEVICE_TYPE).manual_seed(1)
    bias = torch.randn(N, device=DEVICE_TYPE, generator=g).to(dtype)
    ref = (x.double() @ b.double()).float()
    tol = {"rtol": 2e-2, "atol": 2e-2}  # tl.dot on fp32 is TF32, as before
    torch.testing.assert_close(bi.matmul_persistent(x, b).float(), ref, **tol)
    torch.testing.assert_close(
        bi.matmul_persistent(x, b, bias).float(), ref + bias.float(), **tol
    )


@skip_unsupported
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize(
    ("K", "N"), [(4096, 64), (4096, 1024), (4104, 64), (512, 4096)]
)
@pytest.mark.parametrize("M", [1, 2, 7, 16, 17, 33, 64, 65, 129, 1000])
def test_matmul_persistent_batch_invariant(M, K, N, dtype):
    """Row m must not depend on the batch it is computed in: alone, in the
    batch and in a permuted batch it is the same bits, and a repeated call is
    identical. This crosses the tile-height and fused/split reduction
    boundaries, which are the parts allowed to follow M."""
    x, b = _operands(M, K, N, dtype)
    bias = torch.randn(N, device=DEVICE_TYPE).to(dtype)
    y = bi.matmul_persistent(x, b, bias)
    per_row = torch.cat([bi.matmul_persistent(x[i : i + 1], b, bias) for i in range(M)])
    assert torch.equal(y, per_row), f"batch dependence at M={M}"
    perm = torch.randperm(M, device=DEVICE_TYPE)
    assert torch.equal(bi.matmul_persistent(x[perm], b, bias), y[perm]), (
        f"position dependence at M={M}"
    )
    assert torch.equal(y, bi.matmul_persistent(x, b, bias)), (
        f"nondeterministic at M={M}"
    )
