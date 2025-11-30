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
@pytest.mark.parametrize("topk", [2, 4])
@pytest.mark.parametrize("renorm", [True, False])
def test_routing_consistency(num_tokens, num_experts, topk, renorm):
    torch.manual_seed(42)
    device = torch.device("cuda")

    hidden_states = torch.randn(num_tokens, 4096, device=device, dtype=torch.float16)
    router_logits = torch.randn(
        num_tokens, num_experts, device=device, dtype=torch.float32
    )

    def native_impl(logits, topk, renorm):
        if renorm:
            ref_vals, ref_indices = torch.topk(logits, topk, dim=1)
            ref_vals = torch.softmax(ref_vals, dim=1)
        else:
            ref_vals = torch.softmax(logits, dim=1)
            ref_vals, ref_indices = torch.topk(ref_vals, topk, dim=1)

        return ref_vals, ref_indices

    native_weights, native_ids = native_impl(router_logits, topk, renorm)

    ref_weights, ref_ids, _ = FusedMoE.select_experts(
        hidden_states=hidden_states,
        router_logits=router_logits,
        top_k=topk,
        use_grouped_topk=False,
        renormalize=renorm,
        custom_routing_function=None,
    )

    triton_weights, triton_ids, _ = FusedMoE.select_experts(
        hidden_states=hidden_states,
        router_logits=router_logits,
        top_k=topk,
        use_grouped_topk=False,
        renormalize=renorm,
        custom_routing_function=gpt_oss_custom_routing_function,
    )

    print(f"\nTesting M={num_tokens}, E={num_experts}, K={topk}")

    # compare triton with torch
    torch.testing.assert_close(
        triton_ids.to(native_ids.dtype),
        native_ids,
        msg="Expert indices mismatch between native and triton implementation",
    )

    torch.testing.assert_close(
        triton_weights,
        native_weights,
        atol=1e-3,
        rtol=1e-3,
        msg="Expert weights mismatch between native and triton implementation",
    )

    # compare triton with origin
    torch.testing.assert_close(
        triton_ids,
        ref_ids,
        msg="Expert indices mismatch between origin and triton implementation",
    )
    torch.testing.assert_close(
        triton_weights,
        ref_weights,
        atol=1e-3,
        rtol=1e-3,
        msg="Expert weights mismatch between origin and triton implementation",
    )
    # compare origin with torch
    torch.testing.assert_close(
        native_ids,
        ref_ids.to(native_ids.dtype),
        msg="Expert indices mismatch between origin and native implementation",
    )
    torch.testing.assert_close(
        native_weights,
        ref_weights,
        atol=1e-3,
        rtol=1e-3,
        msg="Expert weights mismatch between origin and native implementation",
    )
