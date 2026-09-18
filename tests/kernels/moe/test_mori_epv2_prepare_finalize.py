# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

from vllm.model_executor.layers.fused_moe.prepare_finalize.mori_epv2 import (
    MoriEPv2PrepareAndFinalize,
)


class FakeMoriEPv2Op:
    def __init__(self, recv_count):
        self.routing = object()
        self.recv_count = recv_count
        self.recv_ids = torch.tensor([[2, 3]], dtype=torch.int32).repeat(3, 1)
        self.dispatch_args = None
        self.combine_args = None

    def dispatch(self, *args, **kwargs):
        self.dispatch_args = (args, kwargs)
        return (
            torch.full((3, 4), float("nan"), dtype=torch.bfloat16),
            torch.full((3, 2), float("inf"), dtype=torch.float32),
            None,
            self.recv_ids,
            torch.tensor([self.recv_count], dtype=torch.int32),
            self.routing,
        )

    def combine(self, *args, **kwargs):
        self.combine_args = (args, kwargs)
        return torch.full((1, 4), 7, dtype=torch.bfloat16), None


@pytest.mark.parametrize("recv_count", [0, 1, 3])
def test_mori_epv2_pads_receive_capacity_and_preserves_routing(monkeypatch, recv_count):
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    op = FakeMoriEPv2Op(recv_count)
    prepare_finalize = MoriEPv2PrepareAndFinalize(op, 16, 4)
    hidden = torch.ones(1, 4, dtype=torch.bfloat16)
    weights = torch.ones(1, 2, dtype=torch.bfloat16)
    original_ids = torch.tensor([[0, 1]], dtype=torch.int64)

    prepared = prepare_finalize.prepare(
        hidden,
        weights,
        original_ids,
        num_experts=4,
        expert_map=None,
        apply_router_weight_on_input=False,
        quant_config=SimpleNamespace(),
        defer_input_quant=True,
    )
    for actual, active, padding in (
        (prepared[0], float("nan"), 0),
        (prepared[3], torch.tensor([2, 3]), -1),
        (prepared[4], float("inf"), 0),
    ):
        expected = torch.empty_like(actual)
        expected[:recv_count] = active
        expected[recv_count:] = padding
        torch.testing.assert_close(actual, expected, equal_nan=True)
    output = torch.empty_like(hidden)
    prepare_finalize.finalize(
        output,
        prepared[0],
        prepared[4],
        prepared[3],
        False,
        SimpleNamespace(),
    )

    assert op.dispatch_args is not None
    assert op.combine_args is not None
    assert op.dispatch_args[1] == {
        "return_routing": True,
        "clone_routing": False,
    }
    assert op.dispatch_args[0][1].dtype == torch.float32
    assert op.dispatch_args[0][3].dtype == torch.int32
    assert op.combine_args[0][2] is op.recv_ids
    assert op.combine_args[1]["routing"] is op.routing
    assert torch.equal(output, torch.full_like(output, 7))
