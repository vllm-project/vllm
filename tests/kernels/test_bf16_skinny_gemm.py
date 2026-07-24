# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Numerical correctness for the bf16 skinny GEMM (decode-M GEMV) kernel.

Covers every supported (N, K) shape across M = 0/1/2/8/16/32 (M=32 exercises
the tile_m=32 MMA path), plus strided (column-slice) output. M=0 must return
an empty result without launching the kernel (empty ranks at DP/PP
boundaries)."""

import pytest
import torch

from vllm import _custom_ops as ops
from vllm.platforms import current_platform

# (N, K) pairs must match bf16_skinny_gemm_supported() in the entry .cu.
SHAPES = [
    (768, 12288),
    (1536, 12288),
    (6144, 12288),
    (2048, 2048),
    (2624, 6144),
    (2112, 7168),
    (7168, 14336),
    (512, 6144),
]
MS = [0, 1, 2, 8, 16, 32]

pytestmark = pytest.mark.skipif(
    not current_platform.is_cuda() or not current_platform.has_device_capability(90),
    reason="bf16_skinny_gemm requires CUDA SM90+",
)


def _rel_err(out: torch.Tensor, ref: torch.Tensor) -> float:
    return (out.float() - ref).norm().item() / ref.norm().clamp_min(1e-6).item()


@pytest.mark.parametrize("n,k", SHAPES)
@pytest.mark.parametrize("m", MS)
def test_bf16_skinny_gemm_matches_reference(n: int, k: int, m: int):
    torch.manual_seed(0)
    x = torch.randn(m, k, dtype=torch.bfloat16, device="cuda")
    w = torch.randn(n, k, dtype=torch.bfloat16, device="cuda")

    out = ops.bf16_skinny_gemm(x, w)
    assert out.shape == (m, n)
    assert out.dtype == torch.bfloat16

    if m == 0:  # empty batch: nothing to check beyond the empty shape
        return
    ref = x.float() @ w.float().t()
    rel = _rel_err(out, ref)
    assert rel < 2e-2, f"rel err {rel:.4f} for (m,n,k)=({m},{n},{k})"


@pytest.mark.parametrize("n,k", [(2048, 2048), (512, 6144)])
def test_bf16_skinny_gemm_strided_output(n: int, k: int):
    """Output is a column-slice of a wider padded buffer (row stride > N)."""
    torch.manual_seed(0)
    m = 8
    x = torch.randn(m, k, dtype=torch.bfloat16, device="cuda")
    w = torch.randn(n, k, dtype=torch.bfloat16, device="cuda")

    wide = torch.empty(m, n + 64, dtype=torch.bfloat16, device="cuda")
    view = wide[:, :n]
    torch.ops._C.bf16_skinny_gemm(view, x, w)

    ref = x.float() @ w.float().t()
    assert _rel_err(view, ref) < 2e-2
