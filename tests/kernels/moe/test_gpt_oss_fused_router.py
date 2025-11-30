# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest
import torch

from vllm.model_executor.layers.fused_moe.gpt_oss_fused_router import fused_topk_softmax


@pytest.mark.parametrize("M", [1, 32, 128, 2048])
@pytest.mark.parametrize("N", [32, 65, 128])
@pytest.mark.parametrize("topk", [1, 2, 3, 4, 5])
def test_fused_router(M, N, topk):
    device = "cuda"
    torch.manual_seed(0)

    logits = torch.randn((M, N), device=device, dtype=torch.float32)

    ref_vals, ref_indices = torch.topk(logits, topk, dim=-1)
    ref_probs = torch.softmax(ref_vals, dim=-1)

    tri_probs, tri_indices = fused_topk_softmax(logits, topk, renormalize=True)

    torch.testing.assert_close(tri_indices.long(), ref_indices)
    torch.testing.assert_close(tri_probs, ref_probs, atol=1e-4, rtol=1e-4)


if __name__ == "__main__":
    test_fused_router(128, 32, 2)
    print("Test Passed!")
