# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Correctness of the tiny-dot Triton fast path against the eager chain.

A Qwen MoE shared_expert_gate is a ReplicatedLinear(hidden_size, 1), so it
reaches rocm_unquantized_gemm_impl with m == n == 1 and is served by
_tiny_dot_triton instead of BLAS. The K values below are the ones that gate
actually uses.
"""

from __future__ import annotations

import pytest
import torch

from vllm.model_executor.layers.utils import _TINY_DOT_MAX_K, _tiny_dot_triton


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="Triton kernel requires a GPU"
)
@pytest.mark.parametrize("K", [32, 1024, 2048, 4096, 65536, _TINY_DOT_MAX_K])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_tiny_dot_matches_eager(K: int, dtype: torch.dtype):
    torch.manual_seed(0)
    x = (torch.randn(K, dtype=dtype, device="cuda") * 0.05).contiguous()
    w = (torch.randn(K, dtype=dtype, device="cuda") * 0.05).contiguous()

    ref = (x * w).sum(dtype=x.dtype)
    got = _tiny_dot_triton(x, w)

    # Both paths accumulate in fp32, so the only difference is the rounding
    # of the final store; the bound just has to clear bf16 ulp at this scale.
    torch.testing.assert_close(got, ref, atol=5e-3, rtol=1e-2)
