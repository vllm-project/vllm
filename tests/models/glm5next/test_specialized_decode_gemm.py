# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The MTP draft's split-K BF16 projections against an FP64 PyTorch matmul,
no less accurate than cuBLAS."""

import pytest
import torch

from vllm.platforms import current_platform

CASES = [
    (shape, tokens)
    for shape in [(38720, 4096), (4096, 8192), (2048, 4096), (4096, 4096), (4096, 1536)]
    for tokens in (4, 8, 16, 32, 64)
]


@pytest.mark.skipif(
    not current_platform.is_device_capability_family(100), reason="Requires SM100"
)
@pytest.mark.parametrize("shape,tokens", CASES)
def test_split_k_projection_matches_reference(shape, tokens):
    from vllm.models.glm5next.nvidia.specialized_decode.gemm import Projection

    torch.manual_seed(0)
    n, k = shape
    weight = (torch.randn(n, k, device="cuda") * 0.02).bfloat16()
    projection = Projection(weight, [tokens])
    if not projection.supports(tokens):
        pytest.skip("no split-K tactic for this token count")
    x = torch.randn(tokens, k, device="cuda").bfloat16()
    reference = x.double() @ weight.double().T
    out = projection(x)
    cublas = torch.nn.functional.linear(x, weight)
    # One BF16 rounding (2^-9 relative) of an FP32 accumulation, as cuBLAS.
    for result in (out, cublas):
        error = (result.double() - reference).abs()
        assert (error <= reference.abs() * 2**-8 + 1e-4).all()
    assert out.shape == (tokens, n)
    torch.testing.assert_close(out, cublas, atol=2**-6, rtol=2**-7)
