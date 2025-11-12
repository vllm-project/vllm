# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import torch

from vllm.model_executor.layers.fused_moe.naive_prepare_finalize import (
    FusedMoENaivePrepareAndFinalize,
)


class _DummyEPGroup:
    def __init__(self):
        self.dispatch_calls = []
        self.combine_calls = []

    def dispatch(self, hidden_states, router_logits, is_sequence_parallel):
        self.dispatch_calls.append(is_sequence_parallel)
        return hidden_states + 1, router_logits + 2

    def combine(self, tensor, is_sequence_parallel):
        self.combine_calls.append(is_sequence_parallel)
        return tensor + 3


def test_naive_prepare_finalize_dispatch_and_combine(monkeypatch):
    group = _DummyEPGroup()
    monkeypatch.setattr(
        "vllm.model_executor.layers.fused_moe.naive_prepare_finalize.get_ep_group",
        lambda: group,
    )

    layer = SimpleNamespace(
        is_sequence_parallel=True, shared_experts=None, zero_expert_num=0
    )
    prepare_finalize = FusedMoENaivePrepareAndFinalize()

    hidden_states = torch.zeros(2, 2)
    router_logits = torch.ones(2, 2)

    dispatched_states, dispatched_logits = prepare_finalize.preprocess_inputs(
        hidden_states, router_logits, layer
    )
    assert torch.equal(dispatched_states, hidden_states + 1)
    assert torch.equal(dispatched_logits, router_logits + 2)
    assert group.dispatch_calls == [True]

    combined = prepare_finalize.postprocess_output(
        torch.zeros_like(dispatched_states), layer
    )
    assert torch.equal(combined, torch.zeros_like(dispatched_states) + 3)
    assert group.combine_calls == [True]


def test_naive_prepare_finalize_shared_and_zero_outputs(monkeypatch):
    group = _DummyEPGroup()
    monkeypatch.setattr(
        "vllm.model_executor.layers.fused_moe.naive_prepare_finalize.get_ep_group",
        lambda: group,
    )

    prepare_finalize = FusedMoENaivePrepareAndFinalize()

    shared_tensor = torch.randn(1, 4)
    expert_tensor = torch.randn(1, 4)
    layer_with_shared = SimpleNamespace(
        is_sequence_parallel=False, shared_experts=object(), zero_expert_num=0
    )
    shared_result = prepare_finalize.postprocess_output(
        (shared_tensor, expert_tensor), layer_with_shared
    )
    assert shared_result[0] is shared_tensor
    assert torch.equal(shared_result[1], expert_tensor + 3)

    zero_tensor = torch.randn(1, 4)
    expert_tensor = torch.randn(1, 4)
    layer_with_zero = SimpleNamespace(
        is_sequence_parallel=False, shared_experts=None, zero_expert_num=1
    )
    zero_result = prepare_finalize.postprocess_output(
        (expert_tensor, zero_tensor), layer_with_zero
    )
    assert torch.equal(zero_result[0], expert_tensor + 3)
    assert zero_result[1] is zero_tensor

    # combine was invoked for the shared expert branch and the zero expert branch
    assert group.combine_calls == [False, False]
