# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import torch

from vllm.model_executor.layers.fused_moe.prepare_finalize.mori_epv2 import (
    MoriEPv2PrepareAndFinalize,
)


class FakeMoriEPv2Op:
    def __init__(self):
        self.routing = object()
        self.recv_ids = torch.tensor([[2, 3]], dtype=torch.int32)
        self.dispatch_args = None
        self.combine_args = None

    def dispatch(self, *args, **kwargs):
        self.dispatch_args = (args, kwargs)
        return (
            torch.ones(1, 4, dtype=torch.bfloat16),
            torch.ones(1, 2, dtype=torch.float32),
            None,
            self.recv_ids,
            torch.tensor([1], dtype=torch.int32),
            self.routing,
        )

    def combine(self, *args, **kwargs):
        self.combine_args = (args, kwargs)
        return torch.full((1, 4), 7, dtype=torch.bfloat16), None


def test_mori_epv2_forwards_routing_handle_to_combine(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    op = FakeMoriEPv2Op()
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
    output = torch.empty_like(hidden)
    prepare_finalize.finalize(
        output,
        prepared[0],
        prepared[4],
        prepared[3],
        False,
        SimpleNamespace(),
    )

    assert op.dispatch_args[1] == {
        "return_routing": True,
        "clone_routing": False,
    }
    assert op.dispatch_args[0][1].dtype == torch.float32
    assert op.dispatch_args[0][3].dtype == torch.int32
    assert op.combine_args[0][2] is op.recv_ids
    assert op.combine_args[1]["routing"] is op.routing
    assert torch.equal(output, torch.full_like(output, 7))
