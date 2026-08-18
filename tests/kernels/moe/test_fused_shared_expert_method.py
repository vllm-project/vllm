# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from unittest.mock import MagicMock, patch

import pytest
import torch

from vllm.model_executor.layers.fused_moe.fused_moe_method_base import (
    FusedMoEMethodBase,
)
from vllm.model_executor.layers.fused_moe.runner.moe_runner import (
    MoERunner,
    _moe_forward,
)
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
    runner.routed_output_transform = None
    runner._raw_shared_experts = raw_shared
    runner.enable_dbo = False
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

    with patch(
        "vllm.model_executor.layers.fused_moe.runner.moe_runner."
        "current_platform.is_cpu",
        return_value=True,
    ):
        assert runner._select_forward() is _moe_forward


@pytest.mark.parametrize(
    "disabled_attribute",
    ["shard_sequence_parallel", "use_fi_nvl_two_sided_kernels"],
)
def test_fused_shared_method_skips_separate_shared_when_overlap_is_disabled(
    disabled_attribute,
):
    runner, raw_shared = _runner_with_fused_shared_method()
    if disabled_attribute == "shard_sequence_parallel":
        raw_shared.shard_sequence_parallel = True
    else:
        runner.moe_config.moe_parallel_config.use_fi_nvl_two_sided_kernels = True
    hidden_states = torch.randn(2, 8)
    runner.router.select_experts.return_value = (
        torch.randn(2, 2),
        torch.zeros(2, 2, dtype=torch.int32),
    )
    runner.routed_experts.forward_modular.return_value = torch.randn_like(hidden_states)

    runner._apply_quant_method(
        hidden_states,
        torch.randn(2, 4),
        shared_experts_input=hidden_states,
    )

    raw_shared.assert_not_called()
    assert runner._shared_experts._output == [None, None]


def test_replace_quant_method_rebinds_and_reselects_forward():
    runner, raw_shared = _runner_with_fused_shared_method()
    replacement = MagicMock()
    replacement.mk_fuses_shared_experts = False
    replacement.supports_dbo = True
    selected_forward = object()
    runner._select_forward = MagicMock(return_value=selected_forward)
    runner.routed_experts._replace_quant_method.side_effect = lambda method: setattr(
        runner.routed_experts, "quant_method", method
    )

    runner._replace_quant_method(replacement)

    replacement.bind_shared_experts.assert_called_once_with(
        raw_shared,
        routed_output_transform=None,
    )
    replacement.bind_routed_scaling_factor.assert_called_once_with(2.5)
    assert runner._forward_entry is selected_forward


def test_runner_rejects_dbo_for_unsupported_method():
    runner, _ = _runner_with_fused_shared_method()
    runner.enable_dbo = True
    runner.routed_experts.quant_method.supports_dbo = False

    with pytest.raises(NotImplementedError, match="does not support DBO"):
        runner._bind_quant_method()


def test_fused_method_without_scaling_bind_fails_closed():
    method = MagicMock(spec=FusedMoEMethodBase)
    method.mk_fuses_shared_experts = True

    with pytest.raises(NotImplementedError, match="routed_scaling_factor"):
        FusedMoEMethodBase.bind_routed_scaling_factor(method, 2.0)
