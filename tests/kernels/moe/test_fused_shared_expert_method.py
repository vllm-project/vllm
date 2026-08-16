# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import MagicMock

import torch

from vllm.model_executor.layers.fused_moe.runner.moe_runner import MoERunner
from vllm.model_executor.layers.fused_moe.runner.shared_experts import SharedExperts


def _runner_with_fused_shared_method() -> tuple[MoERunner, MagicMock]:
    runner = object.__new__(MoERunner)
    torch.nn.Module.__init__(runner)
    quant_method = MagicMock()
    quant_method.is_monolithic = False
    quant_method.mk_fuses_shared_experts = True

    raw_shared = MagicMock()
    raw_shared.shard_sequence_parallel = False
    moe_config = MagicMock()
    moe_config.dp_size = 1
    moe_config.pcp_size = 1
    moe_config.is_sequence_parallel = False
    moe_config.moe_parallel_config.enable_eplb = False
    moe_config.moe_parallel_config.use_fi_nvl_two_sided_kernels = False
    runner.moe_config = moe_config
    runner._shared_experts = SharedExperts(
        raw_shared,
        moe_config=moe_config,
        enable_dbo=False,
        mk_can_overlap_shared_experts=lambda: True,
    )
    runner.routed_experts = MagicMock()
    runner.routed_experts.quant_method = quant_method
    runner.router = MagicMock()
    runner.routed_scaling_factor = 2.5
    return runner, raw_shared


def test_fused_shared_method_does_not_launch_shared_module_separately():
    runner, raw_shared = _runner_with_fused_shared_method()
    hidden_states = torch.randn(2, 8)
    router_logits = torch.randn(2, 4)
    topk_weights = torch.randn(2, 2)
    topk_ids = torch.zeros(2, 2, dtype=torch.int32)
    fused_output = torch.randn_like(hidden_states)
    runner.router.select_experts.return_value = (topk_weights, topk_ids)
    runner.routed_experts.forward_modular.return_value = fused_output

    shared_output, output = runner._apply_quant_method(
        hidden_states,
        router_logits,
        shared_experts_input=hidden_states,
    )

    raw_shared.assert_not_called()
    passed_weights = runner.routed_experts.forward_modular.call_args.kwargs[
        "topk_weights"
    ]
    torch.testing.assert_close(passed_weights, topk_weights)
    assert shared_output is None
    assert output is fused_output


def test_fused_shared_method_does_not_rescale_combined_output():
    runner, _ = _runner_with_fused_shared_method()
    fused_output = torch.randn(2, 8)

    shared_output, output = runner._maybe_apply_routed_scale_to_output(
        None, fused_output.clone()
    )

    assert shared_output is None
    torch.testing.assert_close(output, fused_output)


def test_fused_shared_method_uses_single_output_custom_op_schema():
    runner, _ = _runner_with_fused_shared_method()
    runner.routed_experts.quant_method = SimpleNamespace(
        mk_fuses_shared_experts=True,
    )
    output = torch.randn(2, 8)

    assert runner._maybe_combine(None, output) is output
