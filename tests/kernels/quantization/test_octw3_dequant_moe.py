# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The portable Octave-W3 path must rebuild the same dense weights the RDNA3
kernel multiplies by: codebook lookup, group scale and block Hadamard."""

import math

import pytest
import torch

from vllm.model_executor.layers.fused_moe.experts.octw3_dequant_moe import (
    octw3_dequant,
)
from vllm.model_executor.layers.fused_moe.experts.rdna3_octw3_moe import (
    OCTW3_LEVELS,
)


def _hadamard(n: int) -> torch.Tensor:
    h = torch.ones(1, 1)
    while h.shape[0] < n:
        h = torch.cat([torch.cat([h, h], 1), torch.cat([h, -h], 1)], 0)
    return h / math.sqrt(n)


def _planar(codes: torch.Tensor) -> torch.Tensor:
    g = codes.to(torch.int64).unflatten(-1, (-1, 32))
    weights = 1 << torch.arange(32, dtype=torch.int64)
    p = torch.stack([(((g >> b) & 1) * weights).sum(-1) for b in range(3)], -1)
    return torch.where(p >= 2**31, p - 2**32, p).to(torch.int32).flatten(-2)


@pytest.mark.parametrize("n,k,had", [(64, 2560, 128), (2560, 160, 32), (32, 256, 0)])
def test_octw3_dequant_matches_dense_reference(n: int, k: int, had: int):
    torch.manual_seed(0)
    codes = torch.randint(0, 8, (5, n, k))
    scales = (torch.rand(5, n, k // 32) * 0.02 + 0.005).half()
    ref = torch.tensor(OCTW3_LEVELS)[codes] * scales.float().repeat_interleave(32, -1)
    if had:
        ref = (ref.unflatten(-1, (-1, had)) @ _hadamard(had)).flatten(-2)

    got = octw3_dequant(_planar(codes), scales, had, chunk=2)

    assert got.dtype == torch.float16
    assert ((got.float() - ref).norm() / ref.norm()).item() < 1e-3
