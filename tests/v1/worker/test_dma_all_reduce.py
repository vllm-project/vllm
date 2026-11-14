# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""INT4 kernels of the SDMA all-reduce against a torch model (one ROCm GPU)."""

import pytest
import torch

from vllm.platforms import current_platform

pytestmark = pytest.mark.skipif(
    not current_platform.is_rocm(), reason="SDMA all-reduce is ROCm only"
)


def _reference_int4_one_shot(xs: list[torch.Tensor], group: int = 32) -> torch.Tensor:
    """What DmaAllReduce computes: the sum over ranks of dq(q(x_r)) in fp32,
    then x's dtype."""
    acc = None
    for x in xs:
        g = x.float().reshape(-1, group)
        s = (
            torch.clamp(g.abs().amax(dim=1) / 7.0, max=65504.0)
            .to(torch.float16)
            .float()
        )
        inv = torch.where(s > 0, 1.0 / s, torch.zeros_like(s))
        q = torch.clamp(torch.floor(g * inv[:, None] + 0.5), -7, 7)
        d = q * s[:, None]
        acc = d if acc is None else acc + d
    return acc.reshape(xs[0].shape).to(xs[0].dtype)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("tokens", [1, 33, 2048])
def test_quantize_and_sum_matches_reference(dtype, tokens):
    from vllm.distributed.device_communicators.dma_all_reduce import (
        GROUP,
        _dq_sum_kernel,
        _quant_int4_kernel,
    )
    from vllm.triton_utils import triton

    hidden = 256
    torch.manual_seed(0)
    xs = [torch.randn(tokens, hidden, dtype=dtype, device="cuda") for _ in range(2)]
    ng = xs[0].numel() // GROUP
    q = [torch.empty(ng * 16, dtype=torch.uint8, device="cuda") for _ in xs]
    s = [torch.empty(ng, dtype=torch.float16, device="cuda") for _ in xs]
    for x, qi, si in zip(xs, q, s):
        _quant_int4_kernel[(triton.cdiv(ng, 128),)](
            x, qi, si, ng, BLOCK_G=128, enable_fp_fusion=False
        )
    out = torch.empty_like(xs[0])
    _dq_sum_kernel[(triton.cdiv(ng, 128),)](
        q[0], s[0], q[1], s[1], out, ng, BLOCK_G=128, num_warps=2
    )
    torch.testing.assert_close(out, _reference_int4_one_shot(xs), atol=0, rtol=0)
