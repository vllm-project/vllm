# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

import vllm
from vllm.compilation.noop_elimination import NoOpEliminationPass
from vllm.config import (CompilationConfig, CompilationLevel, PassConfig,
                         VllmConfig)

from .backend import TestBackend


@pytest.mark.parametrize("dtype",
                         [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("num_tokens", [256, 1024])
@pytest.mark.parametrize("hidden_size", [64, 4096])
def test_noop_elimination(dtype, num_tokens, hidden_size):
    torch.set_default_device("cuda")
    torch.set_default_dtype(dtype)
    torch.manual_seed(1)

    class Model(torch.nn.Module):

        def forward(self, x):
            # Chain of reshapes
            y = x.reshape(-1, 128, 32)
            z = y.reshape(-1, 4096)
            # No-op reshape
            a = z.reshape(-1, 4096)
            # Final reshape that should remain
            b = a.reshape(-1, 128, 32)
            # No-op slice
            c = b[0:b.shape[0]]
            # The pass should replace the result of this op with `c`.
            d = torch.slice_scatter(
                torch.ones_like(c),  # Dummy tensor to be scattered into
                c,  # Source tensor
                0,  # dim
                0,  # start
                c.shape[0],  # end
            )
            return d

    vllm_config = VllmConfig(compilation_config=CompilationConfig(
        level=CompilationLevel.PIECEWISE,
        pass_config=PassConfig(enable_noop=True),
    ))
    with vllm.config.set_current_vllm_config(vllm_config):
        noop_pass = NoOpEliminationPass(vllm_config)

        backend = TestBackend(noop_pass)

        model = Model()
        # First dimension dynamic
        x = torch.rand(num_tokens, hidden_size)
        torch._dynamo.mark_dynamic(x, 0)

        result = model(x)

        model2 = torch.compile(model, backend=backend)
        result2 = model2(x)

        ATOL, RTOL = (2e-3, 2e-3)
        torch.testing.assert_close(result, result2, atol=ATOL, rtol=RTOL)

        # The no-op reshape and slice should be eliminated.
        # The chain of reshapes should be fused into a single reshape.
        assert backend.op_count(torch.ops.aten.reshape.default) == 1
        assert backend.op_count(torch.ops.aten.slice.Tensor) == 0
        assert backend.op_count(torch.ops.aten.slice_scatter.default) == 0


def test_non_noop_slice_preserved():
    """Ensure that a slice with end=-1 (dropping last row) is NOT eliminated.

    Regression test for a bug where end=-1 was treated like an inferred
    dimension (reshape semantics) leading to incorrect elimination.
    """
    torch.set_default_device("cuda")
    x = torch.randn(16, 16)

    class SliceModel(torch.nn.Module):

        def forward(self, x):
            base = x.clone()
            src = torch.ones(15, 16)
            y = torch.slice_scatter(base, src, dim=0, start=0, end=-1)
            return x[0:-1, :], y

    vllm_config = VllmConfig(compilation_config=CompilationConfig(
        level=CompilationLevel.PIECEWISE,
        pass_config=PassConfig(enable_noop=True),
    ))
    with vllm.config.set_current_vllm_config(vllm_config):
        noop_pass = NoOpEliminationPass(vllm_config)
        backend = TestBackend(noop_pass)
        model = SliceModel()
        ref = model(x)
        compiled = torch.compile(model, backend=backend)
        out = compiled(x)
        torch.testing.assert_close(ref, out)
        # The slice should remain (not a no-op).
        assert backend.op_count(torch.ops.aten.slice.Tensor) == 1
        assert backend.op_count(torch.ops.aten.slice_scatter.default) == 1
