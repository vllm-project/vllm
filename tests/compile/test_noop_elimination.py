# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

import vllm
from vllm.compilation.noop_elimination import NoOpEliminationPass
from vllm.config import (CompilationConfig, CompilationLevel, PassConfig,
                         VllmConfig)

from .backend import TestBackend


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
        return c


@pytest.mark.parametrize("dtype",
                         [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("num_tokens", [256, 1024])
@pytest.mark.parametrize("hidden_size", [64, 4096])
def test_noop_elimination(dtype, num_tokens, hidden_size):
    torch.set_default_device("cuda")
    torch.set_default_dtype(dtype)
    torch.manual_seed(1)

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
