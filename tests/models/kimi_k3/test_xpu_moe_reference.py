# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
import tempfile

import pytest
import torch

from tests.models.kimi_k3.kimi_moe_reference import kimi_moe_reference
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.distributed import (
    cleanup_dist_env_and_memory,
    init_distributed_environment,
    initialize_model_parallel,
)
from vllm.forward_context import set_forward_context
from vllm.models.kimi_k3.xpu.linear import KimiMoE
from vllm.platforms import current_platform
from vllm.transformers_utils.configs.kimi_linear import KimiLinearConfig
from vllm.v1.worker.workspace import init_workspace_manager

pytestmark = pytest.mark.skipif(
    not current_platform.is_xpu(),
    reason="XPU KimiMoE reference test requires an XPU device",
)


def _make_config() -> KimiLinearConfig:
    return KimiLinearConfig(
        hidden_size=64,
        num_attention_heads=1,
        hidden_act="situ",
        moe_intermediate_size=32,
        num_experts=32,
        num_experts_per_token=16,
        num_shared_experts=1,
        routed_expert_hidden_size=32,
        latent_moe_use_norm=True,
        activation_situ_beta=1.25,
        activation_situ_linear_beta=0.5,
        routed_scaling_factor=1.0,
    )


def test_xpu_kimi_moe_matches_independent_stage_reference() -> None:
    torch.xpu.set_device(0)
    device = torch.device("xpu", 0)
    file_descriptor, init_path = tempfile.mkstemp(prefix="kimi_moe_test_")
    os.close(file_descriptor)
    vllm_config = VllmConfig()

    with set_current_vllm_config(vllm_config):
        init_distributed_environment(
            world_size=1,
            rank=0,
            distributed_init_method=f"file://{init_path}",
            local_rank=0,
            backend="gloo",
        )
        initialize_model_parallel(1)
        init_workspace_manager(device)
        previous_dtype = torch.get_default_dtype()
        torch.set_default_dtype(torch.bfloat16)
        try:
            with torch.device(device):
                moe = KimiMoE(_make_config())
        finally:
            torch.set_default_dtype(previous_dtype)

        torch.manual_seed(31)
        for parameter in moe.parameters():
            parameter.data.uniform_(-0.05, 0.05)
        hidden_states = torch.randn(4, 64, dtype=torch.bfloat16, device=device) * 0.1

        reference = kimi_moe_reference(
            hidden_states.cpu(),
            gate_weight=moe.gate.weight.cpu(),
            correction_bias=moe.gate.e_score_correction_bias.cpu(),
            routed_down_weight=moe.routed_expert_down_proj.weight.cpu(),
            routed_norm_weight=moe.routed_expert_norm.weight.cpu(),
            routed_up_weight=moe.routed_expert_up_proj.weight.cpu(),
            w13_weight=moe.experts.routed_experts.w13_weight.cpu(),
            w2_weight=moe.experts.routed_experts.w2_weight.cpu(),
            shared_w13_weight=moe.shared_experts.gate_up_proj.weight.cpu(),
            shared_w2_weight=moe.shared_experts.down_proj.weight.cpu(),
            top_k=16,
            routed_scaling_factor=1.0,
            rms_norm_eps=moe.routed_expert_norm.variance_epsilon,
            situ_beta=1.25,
            situ_linear_beta=0.5,
        )
        moe.experts.routed_experts.quant_method.process_weights_after_loading(
            moe.experts.routed_experts
        )

        router_logits, _ = moe.gate(hidden_states)
        topk_weights, topk_ids = moe.experts.router.select_experts(
            hidden_states,
            router_logits,
        )
        routed_input, _ = moe.routed_expert_down_proj(hidden_states)
        shared_output = moe.shared_experts(hidden_states)
        shared_experts = moe.experts._shared_experts
        moe.experts._shared_experts = None
        with set_forward_context({}, vllm_config, num_tokens=hidden_states.shape[0]):
            routed_combined = moe.experts._forward_impl(
                routed_input,
                router_logits,
                shared_experts_input=None,
            )
        moe.experts._shared_experts = shared_experts
        with set_forward_context({}, vllm_config, num_tokens=hidden_states.shape[0]):
            output = moe(hidden_states)
        torch.xpu.synchronize()

    actual = {
        "router_logits": router_logits.cpu(),
        "topk_ids": topk_ids.cpu().long(),
        "topk_weights": topk_weights.cpu(),
        "routed_input": routed_input.cpu(),
        "routed_combined": routed_combined.cpu(),
        "shared_output": shared_output.cpu(),
        "output": output.cpu(),
    }
    cleanup_dist_env_and_memory()

    torch.testing.assert_close(
        actual["router_logits"], reference.router_logits, rtol=5e-3, atol=2e-4
    )
    actual_ids, actual_order = actual["topk_ids"].sort(dim=-1)
    reference_ids, reference_order = reference.topk_ids.sort(dim=-1)
    actual_weights = actual["topk_weights"].gather(1, actual_order)
    reference_weights = reference.topk_weights.gather(1, reference_order)
    torch.testing.assert_close(actual_ids, reference_ids)
    torch.testing.assert_close(actual_weights, reference_weights, rtol=5e-3, atol=2e-4)
    torch.testing.assert_close(
        actual["routed_input"].float(),
        reference.routed_input.float(),
        rtol=2e-2,
        atol=2e-4,
    )
    torch.testing.assert_close(
        actual["routed_combined"].float(),
        reference.routed_combined.float(),
        rtol=2e-2,
        atol=2e-4,
    )
    torch.testing.assert_close(
        actual["shared_output"].float(),
        reference.shared_output.float(),
        rtol=2e-2,
        atol=2e-4,
    )
    torch.testing.assert_close(
        actual["output"].float(),
        reference.output.float(),
        rtol=2e-2,
        atol=2e-4,
    )
