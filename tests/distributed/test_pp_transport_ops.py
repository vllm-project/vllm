# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.distributed.pp_transport.ops import (
    export_pp_dependencies,
    import_pp_dependencies,
)
from vllm.forward_context import ForwardContext, override_forward_context


class RowCache:
    def __init__(self, values):
        self.values = values

    def copy_cache_rows(self, context, rows, *, receive):
        slots = context.slot_mapping["cache"]
        values = rows[0].view(-1, self.values.shape[-1])
        if receive:
            self.values.index_copy_(0, slots, values)
        else:
            values.copy_(self.values.index_select(0, slots))


@pytest.mark.parametrize("compiled", [False, True])
def test_dependencies_resolve_runtime_slots_after_metadata_free_trace(compiled):
    source = RowCache(torch.arange(24, dtype=torch.float32).view(12, 2))
    target = RowCache(torch.full((12, 2), -1.0))
    state = torch.arange(16, dtype=torch.float32).view(8, 2)
    restored = torch.full_like(state, -1)
    cache_wire = torch.empty(2, 2, 2)  # Local token extent x PCP axis x head.
    state_wire = torch.empty(4, 2)

    def forward(hidden, state, restored, cache_wire, state_wire):
        export_pp_dependencies(hidden, [cache_wire], [state], [state_wire], "source")
        import_pp_dependencies(hidden, [cache_wire], [state_wire], [restored], "target")
        return hidden + restored[:2]

    if compiled:
        forward = torch.compile(forward, backend="aot_eager", fullgraph=True)
    hidden = torch.ones(2, 2)  # SP activations; state/cache payloads have four rows.
    context = ForwardContext(
        no_compile_layers={"source": source, "target": target},
        attn_metadata=None,
        slot_mapping={},
    )
    with override_forward_context(context):
        forward(hidden, state, restored, cache_wire, state_wire)
    torch.testing.assert_close(target.values, torch.full((12, 2), -1.0))
    for slots in (torch.tensor([1, 3, 5, 7]), torch.tensor([2, 4, 8, 10])):
        context.attn_metadata = {}
        context.slot_mapping = {"cache": slots}
        state.add_(10)
        with override_forward_context(context):
            result = forward(hidden, state, restored, cache_wire, state_wire)
        torch.testing.assert_close(
            cache_wire.view(4, 2), source.values[slots], rtol=0, atol=0
        )
        torch.testing.assert_close(
            target.values[slots], source.values[slots], rtol=0, atol=0
        )
        torch.testing.assert_close(restored[:4], state[:4], rtol=0, atol=0)
        torch.testing.assert_close(restored[4:], torch.full((4, 2), -1.0))
        torch.testing.assert_close(result, hidden + state[:2], rtol=0, atol=0)
        snapshot = state_wire.clone()
        state.add_(1)
        torch.testing.assert_close(state_wire, snapshot, rtol=0, atol=0)
