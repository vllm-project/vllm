# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from contextlib import nullcontext
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import Mock

import pytest
import torch

from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.modular_kernel import (
    FusedMoEKernelModularImpl,
)


@pytest.mark.parametrize(
    ("source_expert_count", "backend_expert_count", "expect_mismatch"),
    [
        pytest.param(0, 3, False, id="released-source"),
        pytest.param(2, 2, True, id="real-mismatch"),
    ],
)
def test_workspace_uses_backend_expert_count(
    monkeypatch: pytest.MonkeyPatch,
    source_expert_count: int,
    backend_expert_count: int,
    expect_mismatch: bool,
) -> None:
    prepared_expert_count = 3
    fused_experts = SimpleNamespace(
        moe_config=SimpleNamespace(moe_parallel_config=None),
        moe_problem_size=Mock(return_value=(backend_expert_count, 2, 8, 4, 1)),
        a2_scale=None,
        apply=Mock(),
    )
    kernel = FusedMoEKernelModularImpl(
        prepare_finalize=cast(Any, SimpleNamespace()),
        fused_experts=cast(Any, fused_experts),
    )
    source_shape = (source_expert_count, 1, 1) if source_expert_count else (0, 0, 0)
    source_w1 = torch.empty(source_shape)
    source_w2 = torch.empty(source_shape)
    hidden_states = torch.empty((2, 4))
    topk_ids = torch.zeros((2, 1), dtype=torch.int64)
    topk_weights = torch.ones((2, 1))
    workspace_expert_counts = []

    def allocate_buffers(*args: Any) -> tuple[torch.Tensor, ...]:
        local_expert_count = cast(int, args[8])
        workspace_expert_counts.append(local_expert_count)
        if local_expert_count != prepared_expert_count:
            raise ValueError(
                f"metadata={local_expert_count}, prepared={prepared_expert_count}"
            )
        return (
            torch.empty(0),
            torch.empty(0),
            torch.empty((2, 4)),
        )

    monkeypatch.setattr(kernel, "_allocate_buffers", allocate_buffers)
    monkeypatch.setattr(
        kernel,
        "_prepare",
        Mock(
            return_value=(
                hidden_states,
                None,
                None,
                topk_ids,
                topk_weights,
            )
        ),
    )
    monkeypatch.setattr(
        kernel,
        "_finalize",
        Mock(return_value=torch.empty_like(hidden_states)),
    )

    expectation = (
        pytest.raises(ValueError, match="metadata=2, prepared=3")
        if expect_mismatch
        else nullcontext()
    )
    with expectation:
        kernel.apply(
            hidden_states=hidden_states,
            w1=source_w1,
            w2=source_w2,
            topk_ids=topk_ids,
            topk_weights=topk_weights,
            activation=MoEActivation.SILU,
            global_num_experts=8,
        )

    fused_experts.moe_problem_size.assert_called_once_with(
        hidden_states, source_w1, source_w2, topk_ids
    )
    assert workspace_expert_counts == [backend_expert_count]
