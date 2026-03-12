# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

import vllm
from tests.compile.backend import TestBackend
from vllm.compilation.passes.utility.split_coalescing import SplitCoalescingPass
from vllm.config import CompilationConfig, CompilationMode, PassConfig, VllmConfig


class SplitCoalescingModel(torch.nn.Module):
    """Model with 3 separate split_with_sizes calls on the same input,
    simulating the B200+FP8 graph where CSE fails to merge them."""

    def __init__(self, q_size: int, kv_size: int) -> None:
        super().__init__()
        self.q_size = q_size
        self.kv_size = kv_size

    def forward(self, qkv: torch.Tensor):
        q, _, _ = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        _, k, _ = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        _, _, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        return q + 1, k + 2, v + 3


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_split_coalescing(dtype):
    torch.set_default_device("cuda")
    torch.set_default_dtype(dtype)
    torch.manual_seed(0)

    q_size, kv_size = 2048, 512

    vllm_config = VllmConfig(
        compilation_config=CompilationConfig(
            mode=CompilationMode.VLLM_COMPILE,
            pass_config=PassConfig(),
        )
    )
    with vllm.config.set_current_vllm_config(vllm_config):
        coalesce_pass = SplitCoalescingPass(vllm_config)
        backend = TestBackend(coalesce_pass)

        model = SplitCoalescingModel(q_size, kv_size)

        T = 5
        qkv = torch.randn(T, q_size + 2 * kv_size)
        torch._dynamo.mark_dynamic(qkv, 0)

        result_eager = model(qkv)

        model_compiled = torch.compile(model, backend=backend)
        result_compiled = model_compiled(qkv)

        ATOL, RTOL = (2e-3, 2e-3)
        for eager, compiled in zip(result_eager, result_compiled):
            torch.testing.assert_close(eager, compiled, atol=ATOL, rtol=RTOL)

        assert backend.op_count(torch.ops.aten.split_with_sizes.default) == 1
