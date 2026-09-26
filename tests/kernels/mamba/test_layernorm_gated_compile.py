# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The grouped gated RMSNorm Triton kernel gives the same bits when Dynamo traces
it and Inductor re-emits it as when it is launched eagerly."""

import pytest
import torch

from vllm.model_executor.layers.mamba.ops.layernorm_gated import rms_norm_gated
from vllm.platforms import current_platform

pytestmark = pytest.mark.skipif(
    not current_platform.is_cuda(), reason="requires a CUDA device"
)


@pytest.mark.parametrize("group_size", [1024, 8192])
@pytest.mark.parametrize("gated", [False, True])
@pytest.mark.parametrize("scale", [0.005, 1.0])
def test_rms_norm_gated_matches_eager_under_torch_compile(group_size, gated, scale):
    """Inductor passes the Python float ``eps`` as fp64 where the eager launch
    passes fp32; the kernel keeps ``eps`` in fp32 so both agree bitwise. Small
    activations make ``mean(x^2)`` comparable to ``eps``, where the fp64
    addition changes the rounding."""
    torch.manual_seed(0)
    x = (scale * torch.randn(578, 8192, device="cuda")).to(torch.bfloat16)
    weight = (1 + 0.1 * torch.randn(8192, device="cuda")).to(torch.bfloat16)
    z = torch.randn_like(x) if gated else None

    def norm(inp, w, gate):
        return rms_norm_gated(
            inp,
            w,
            bias=None,
            z=gate,
            eps=1e-5,
            group_size=group_size,
            norm_before_gate=False,
        )

    eager = norm(x, weight, z)
    compiled = torch.compile(norm, backend="inductor", fullgraph=True)(x, weight, z)
    assert torch.equal(eager, compiled)
