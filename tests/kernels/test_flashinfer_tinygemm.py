# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Hopper+ correctness and torch.compile tests for FlashInfer tinygemm_bf16.

Registration of ``torch.ops.vllm.tinygemm_bf16`` happens at import time in
``vllm.model_executor.layers.utils`` when FlashInfer is installed and the
device has SM90 or newer. The tests here skip at the module level otherwise.
"""

import pytest
import torch

from vllm.platforms import current_platform

if not current_platform.is_cuda() or not current_platform.has_device_capability(90):
    pytest.skip(
        "FlashInfer tinygemm_bf16 requires CUDA SM90+",
        allow_module_level=True,
    )

try:
    from vllm.utils.flashinfer import has_flashinfer
except ImportError:
    pytest.skip("flashinfer is not available", allow_module_level=True)

if not has_flashinfer():
    pytest.skip("flashinfer is not available", allow_module_level=True)

# Importing utils triggers _init_tinygemm() — which registers the custom op.
from vllm.model_executor.layers import utils  # noqa: E402

if not utils._TINYGEMM_AVAILABLE:
    pytest.skip("tinygemm custom op registration failed", allow_module_level=True)


# (M, N, K) — shapes representative of Llama-3-8B decode (QKV, MLP up/down).
SHAPES = [
    (1, 4096, 4096),
    (1, 12288, 4096),
    (1, 14336, 4096),
    (1, 4096, 14336),
    (8, 4096, 4096),
]


@pytest.mark.parametrize("m,n,k", SHAPES)
@pytest.mark.parametrize("with_bias", [False, True])
def test_tinygemm_matches_linear(m, n, k, with_bias):
    torch.manual_seed(0)
    x = torch.randn(m, k, dtype=torch.bfloat16, device="cuda")
    w = torch.randn(n, k, dtype=torch.bfloat16, device="cuda")
    bias = (
        torch.randn(n, dtype=torch.bfloat16, device="cuda")
        if with_bias
        else torch.zeros(n, dtype=torch.bfloat16, device="cuda")
    )

    out = torch.ops.vllm.tinygemm_bf16(x, w, bias)
    ref = torch.nn.functional.linear(
        x, w, bias if with_bias else None
    )

    torch.testing.assert_close(out, ref, atol=5e-2, rtol=5e-2)


def test_tinygemm_fake_impl_metadata():
    """Fake impl must return the right shape/dtype/device for compile tracing."""
    x = torch.empty(1, 4096, dtype=torch.bfloat16, device="cuda")
    w = torch.empty(4096, 4096, dtype=torch.bfloat16, device="cuda")
    bias = torch.empty(4096, dtype=torch.bfloat16, device="cuda")

    with torch._subclasses.FakeTensorMode():
        fake_x = torch.empty_like(x)
        fake_w = torch.empty_like(w)
        fake_b = torch.empty_like(bias)
        fake_out = utils._tinygemm_bf16_fake(fake_x, fake_w, fake_b)

    assert fake_out.shape == (1, 4096)
    assert fake_out.dtype == torch.bfloat16
    assert fake_out.device.type == "cuda"


@pytest.mark.parametrize("m", [1, 8])
def test_tinygemm_dispatch_compiles_fullgraph(m):
    """_tinygemm_unquantized_gemm must compile under fullgraph without breaks."""
    torch.manual_seed(0)
    k, n = 4096, 4096
    x = torch.randn(m, k, dtype=torch.bfloat16, device="cuda")
    w = torch.randn(n, k, dtype=torch.bfloat16, device="cuda")

    def fn(x, w):
        return utils._tinygemm_unquantized_gemm(None, x, w)

    compiled = torch.compile(fn, fullgraph=True, dynamic=False)

    eager_out = fn(x, w)
    compiled_out = compiled(x, w)

    torch.testing.assert_close(compiled_out, eager_out, atol=5e-2, rtol=5e-2)
