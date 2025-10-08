# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the Pallas MOE implementation.

Run `pytest tests/kernels/moe/test_moe_pallas.py`.
"""

import pytest
import torch
import torch_xla

from vllm.model_executor.layers.fused_moe.moe_pallas import fused_moe as pallas_moe
from vllm.model_executor.layers.fused_moe.moe_torch_iterative import (
    fused_moe as torch_moe,
)
from vllm.platforms import current_platform

if not current_platform.is_tpu():
    pytest.skip("This test needs a TPU.", allow_module_level=True)

NUM_EXPERTS = [8, 64]
EP_SIZE = [1]
TOP_KS = [2, 6]


# The Pallas GMM kernel requires num_tokens * topk to be a multiple of 16
@pytest.mark.parametrize("m", [8, 16, 64, 2048])
@pytest.mark.parametrize("n", [128, 1024, 2048])
@pytest.mark.parametrize("k", [128, 511, 1024])
@pytest.mark.parametrize("e", NUM_EXPERTS)
@pytest.mark.parametrize("topk", TOP_KS)
@pytest.mark.parametrize("ep_size", EP_SIZE)
@pytest.mark.parametrize("dtype", [torch.bfloat16])
def test_pallas_moe(
    m: int,
    n: int,
    k: int,
    e: int,
    topk: int,
    ep_size: int,
    dtype: torch.dtype,
):
    import torch_xla.core.xla_model as xm

    with torch.device(xm.xla_device()):
        a = torch.randn((m, k), dtype=dtype) / 10
        w1 = torch.randn((e, 2 * n, k), dtype=dtype) / 10
        w2 = torch.randn((e, k, n), dtype=dtype) / 10

        score = torch.randn((m, e), dtype=dtype)

        # TODO: Support ep
        if ep_size > 1:
            pytest.skip("No support for ep_size > 1 yet")
        else:
            e_map = None

        # Run both implementations
        torch_output = torch_moe(
            hidden_states=a,
            w1=w1,
            w2=w2,
            gating_output=score,
            topk=topk,
            global_num_experts=e,
            expert_map=e_map,
            renormalize=False,
        )

        pallas_output = pallas_moe(
            hidden_states=a,
            w1=w1,
            w2=w2,
            gating_output=score,
            topk=topk,
            global_num_experts=e,
            expert_map=e_map,
            renormalize=False,
        )
        torch_xla.sync(wait=False)

    # Compare outputs
    torch.testing.assert_close(
        pallas_output.cpu(),
        torch_output.cpu(),
        atol=2e-2,
        rtol=0,
    )
