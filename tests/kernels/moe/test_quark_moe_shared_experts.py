# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from vllm.model_executor.layers.quantization.quark import quark_moe


@pytest.mark.parametrize(
    "method_cls", ["QuarkOCP_MX_MoEMethod", "QuarkW8A8Int8MoEMethod"]
)
def test_quark_moe_passes_shared_experts_to_the_modular_kernel(method_cls):
    # With an async prepare/finalize the modular kernel runs the shared experts
    # (overlapped with the combine), so apply() must hand them over.
    method = SimpleNamespace(moe_kernel=Mock())
    layer = SimpleNamespace(
        w13_weight=torch.zeros(1),
        w2_weight=torch.zeros(1),
        activation=None,
        global_num_experts=4,
        apply_router_weight_on_input=False,
        expert_map=None,
    )
    shared_experts, shared_experts_input = Mock(), torch.zeros(2, 8)
    getattr(quark_moe, method_cls).apply(
        method,
        layer,
        torch.zeros(2, 8),
        torch.zeros(2, 1),
        torch.zeros(2, 1, dtype=torch.int32),
        shared_experts,
        shared_experts_input,
    )
    kwargs = method.moe_kernel.apply.call_args.kwargs
    assert kwargs["shared_experts"] is shared_experts
    assert kwargs["shared_experts_input"] is shared_experts_input
