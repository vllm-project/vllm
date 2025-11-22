# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest
import torch

from vllm.model_executor.layers.fused_moe.gpt_oss_fused_router import (
    gpt_oss_custom_routing_function,
)
from vllm.model_executor.layers.fused_moe.layer import FusedMoE


@pytest.mark.parametrize("num_tokens", [10, 128, 1024])
@pytest.mark.parametrize("num_experts", [32, 65, 128])
@pytest.mark.parametrize("topk", [1, 2, 3, 4, 5])
def test_routing_consistency(num_tokens, num_experts, topk):
    torch.manual_seed(42)
    device = torch.device("cuda")

    hidden_states = torch.randn(num_tokens, 4096, device=device, dtype=torch.float16)
    router_logits = torch.randn(
        num_tokens, num_experts, device=device, dtype=torch.float32
    )

    ref_weights, ref_ids, _ = FusedMoE.select_experts(
        hidden_states=hidden_states,
        router_logits=router_logits,
        top_k=topk,
        use_grouped_topk=False,
        renormalize=True,
        custom_routing_function=None,
    )

    triton_weights, triton_ids, _ = FusedMoE.select_experts(
        hidden_states=hidden_states,
        router_logits=router_logits,
        top_k=topk,
        use_grouped_topk=False,
        renormalize=True,
        custom_routing_function=gpt_oss_custom_routing_function,
    )

    print(f"\nTesting M={num_tokens}, E={num_experts}, K={topk}")

    torch.testing.assert_close(
        triton_ids,
        ref_ids,
        msg="Expert indices mismatch between Native and Triton implementation",
    )

    torch.testing.assert_close(
        triton_weights,
        ref_weights,
        atol=1e-3,
        rtol=1e-3,
        msg="Expert weights mismatch between Native and Triton implementation",
    )


if __name__ == "__main__":
    test_routing_consistency(128, 32, 2)
    print("Consistency Test Passed!")
