# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
import torch.nn.functional as F

import vllm
from vllm.compilation.fusion_mul_pad import MulPadFusionPass
from vllm.config import (CompilationConfig, CompilationLevel, PassConfig,
                         VllmConfig)
from vllm.utils import round_up

from .backend import TestBackend


@pytest.mark.parametrize("dtype",
                         [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("shape", [(3, 12), (1, 26), (1, 256), (100, 500)])
def test_mul_pad_fusion(shape, dtype):
    torch.set_default_device("cuda")
    torch.set_default_dtype(dtype)
    torch.manual_seed(42)

    class Model(torch.nn.Module):

        def forward(self, a, b):
            # mul + pad pattern
            x = a * b
            y = F.pad(x, (0, round_up(a.shape[-1], 256) - a.shape[-1]),
                      value=0.0)
            z = x.mean(dim=-1, keepdim=True)
            return y + z

    vllm_config = VllmConfig(compilation_config=CompilationConfig(
        level=CompilationLevel.PIECEWISE,
        pass_config=PassConfig(enable_mul_pad_fusion=True),
    ))
    with vllm.config.set_current_vllm_config(vllm_config):
        fusion_pass = MulPadFusionPass(vllm_config)
        backend = TestBackend(fusion_pass)
        model = Model()

        a = torch.randn(shape, device="cuda", dtype=dtype)
        b = torch.randn([1, shape[-1]], device="cuda", dtype=dtype)
        ref = model(a, b)
        compiled = torch.compile(model, backend=backend)
        out = compiled(a, b)

        ATOL, RTOL = (1e-2, 1e-2)
        torch.testing.assert_close(ref, out, atol=ATOL, rtol=RTOL)
    assert fusion_pass.matched_count == (1 if shape[-1] % 256 != 0 else 0)


def test_non_fusion_pad_preserved():
    torch.set_default_device("cuda")
    dtype = torch.float32
    a = torch.randn(2, 8, device="cuda", dtype=dtype)
    b = torch.randn(1, 8, device="cuda", dtype=dtype)

    class Model(torch.nn.Module):

        def forward(self, a, b):
            y = F.pad(a, (0, 2), value=1.0)
            return y

    vllm_config = VllmConfig(compilation_config=CompilationConfig(
        level=CompilationLevel.PIECEWISE,
        pass_config=PassConfig(enable_fusion=True),
    ))
    with vllm.config.set_current_vllm_config(vllm_config):
        fusion_pass = MulPadFusionPass(vllm_config)
        backend = TestBackend(fusion_pass)
        model = Model()
        ref = model(a, b)
        compiled = torch.compile(model, backend=backend)
        out = compiled(a, b)
        torch.testing.assert_close(ref, out)
        assert fusion_pass.matched_count == 0
