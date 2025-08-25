# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.model_executor.layers.fused_moe.batched_deep_gemm_moe import (
    silu_mul_fp8_quant_deep_gemm_cuda)
from vllm.model_executor.layers.fused_moe.old_batched_deep_gemm_moe import (
    silu_mul_fp8_quant_deep_gemm as gold)
from vllm.platforms import current_platform

# (E, T, H, group_size, seed)
CASES = [
    (1, 1, 128, 64, 0),
    (1, 4, 128, 128, 0),
    (2, 4, 256, 128, 0),
    (8, 32, 1024, 128, 0),
    (8, 32, 2048, 128, 0),
    (32, 64, 256, 128, 0),
    (64, 64, 256, 128, 0),
    (17, 31, 768, 128, 0),
    (256, 64, 256, 128, 0),
    (256, 16, 7168, 128, 0),
    (256, 32, 7168, 128, 0),
    (256, 64, 7168, 128, 0),
    (256, 128, 7168, 128, 0),
    (256, 256, 7168, 128, 0),
    (256, 512, 7168, 128, 0),
    (256, 1024, 7168, 128, 0),
]


@pytest.mark.parametrize("E,T,H,group_size,seed", CASES)
@torch.inference_mode()
def test_silu_mul_fp8_quant_deep_gemm(E, T, H, group_size, seed):
    current_platform.seed_everything(seed)

    # Input tensor of shape (E, T, 2*H)
    y = torch.randn((E, T, 2 * H), dtype=torch.bfloat16, device="cuda")
    tokens_per_expert = torch.randint(
        low=0,
        high=T,
        size=(E, ),
        dtype=torch.int32,
        device="cuda",
    )

    # Run the Triton kernel
    y_q, y_s = silu_mul_fp8_quant_deep_gemm_cuda(y,
                                                 tokens_per_expert,
                                                 group_size=group_size,
                                                 eps=1e-10)

    gold_y_q, gold_y_s = gold(y,
                              tokens_per_expert,
                              group_size=group_size,
                              eps=1e-10)
    torch.testing.assert_close(y_q.float(),
                               gold_y_q.float(),
                               atol=2,
                               rtol=2e-1)
    torch.testing.assert_close(y_s.float(), gold_y_s.float())
