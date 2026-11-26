# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MORI EPv1 prepare/finalize: experts see only the rows that can hold received
tokens, not the capacity-sized (world_size * max_num_tokens) receive buffer."""

import importlib
import sys
import types
from types import SimpleNamespace

import pytest
import torch

import vllm.model_executor.layers.fused_moe.prepare_finalize.mori_common as mori_common


def _import_mori_prepare_finalize(monkeypatch):
    # Contract test for vLLM's adapter; no real MoRI installation or GPU needed.
    fake_mori = types.ModuleType("mori")
    fake_mori.ops = SimpleNamespace(  # type: ignore[attr-defined]
        EpDispatchCombineOp=object
    )
    monkeypatch.setitem(sys.modules, "mori", fake_mori)
    module_name = "vllm.model_executor.layers.fused_moe.prepare_finalize.mori"
    sys.modules.pop(module_name, None)
    return importlib.import_module(module_name)


class FakeOp:
    """MORI EPv1 op whose receive buffers have `capacity` rows, `recv` valid."""

    def __init__(self, capacity=256, recv=0):
        self.recv_a1 = torch.full((capacity, 4), 7.0, dtype=torch.bfloat16)
        self.recv_weights = torch.full((capacity, 2), float("inf"))
        self.recv_ids = torch.full((capacity, 2), 5, dtype=torch.int32)
        self.recv = recv
        self.combine_input: torch.Tensor | None = None
        self.combine_indices: torch.Tensor | None = None

    def dispatch(self, input, weights, scales, indices):
        return (
            self.recv_a1,
            self.recv_weights,
            None,
            self.recv_ids,
            torch.tensor([self.recv], dtype=torch.int32),
        )

    def combine(self, input, weights, indices):
        self.combine_input = input
        self.combine_indices = indices
        return (torch.zeros(input.shape[0], 4, dtype=torch.bfloat16), None)


def _run(monkeypatch, op, num_tokens_across_dp, topk_ids):
    if num_tokens_across_dp is None:
        ctx = SimpleNamespace(dp_metadata=None)
    else:
        ctx = SimpleNamespace(
            dp_metadata=SimpleNamespace(
                num_tokens_across_dp_cpu=torch.tensor(num_tokens_across_dp)
            )
        )
    monkeypatch.setattr(mori_common, "is_forward_context_available", lambda: True)
    monkeypatch.setattr(mori_common, "get_forward_context", lambda: ctx)
    monkeypatch.setattr(
        mori_common, "get_dp_group", lambda: SimpleNamespace(rank_in_group=0)
    )
    mori_module = _import_mori_prepare_finalize(monkeypatch)
    pf = mori_module.MoriPrepareAndFinalize(
        op, max_tokens_per_rank=16, num_dispatchers=4
    )
    n = topk_ids.shape[0]
    prepared = pf.prepare(
        torch.ones(n, 4, dtype=torch.bfloat16),
        torch.ones(n, 2, dtype=torch.float32),
        topk_ids,
        num_experts=8,
        expert_map=None,
        apply_router_weight_on_input=False,
        quant_config=SimpleNamespace(),
    )
    pf.finalize(
        torch.empty(n, 4, dtype=torch.bfloat16),
        prepared[0],
        prepared[4],
        prepared[3],
        False,
        SimpleNamespace(),
    )
    return prepared


@pytest.mark.parametrize(
    "num_tokens_across_dp,recv,rows",
    [
        # Sum of DP batches (6) rounded up to 32 rows, of 256 capacity rows.
        ([1, 3, 2, 0], 7, 32),
        ([33, 0, 0, 0], 20, 64),
        # Bound above capacity: keep the full buffer.
        ([100, 100, 100, 100], 200, 256),
        # No DP metadata, or stale metadata missing this rank's batch: full buffer.
        (None, 7, 256),
        ([0, 3, 2, 0], 5, 256),
    ],
)
def test_epv1_receive_rows_are_bounded(monkeypatch, num_tokens_across_dp, recv, rows):
    op = FakeOp(recv=recv)
    topk_ids = torch.tensor([[0, 1]], dtype=torch.int32)
    a1, _, meta, ids, weights = _run(monkeypatch, op, num_tokens_across_dp, topk_ids)
    for t, buf in ((a1, op.recv_a1), (ids, op.recv_ids), (weights, op.recv_weights)):
        assert t.shape[0] == rows
        assert t.data_ptr() == buf.data_ptr()  # views, no copy
    # Received rows untouched; stale rows past the receive count are invalidated.
    assert (ids[:recv] == 5).all() and (ids[recv:] == -1).all()
    assert torch.isinf(weights[:recv]).all() and (weights[recv:] == 0).all()
    assert (a1 == 7).all()
    # Rows past the bound are not touched at all.
    assert (op.recv_ids[rows:] == 5).all()
    assert int(meta.expert_num_tokens) == recv
    # combine reduces the bounded expert output with this rank's own routing.
    assert op.combine_input is not None and op.combine_input.shape[0] == rows
    assert op.combine_indices is topk_ids


def test_bound_helper_matches_dispatcher_rules(monkeypatch):
    ctx = SimpleNamespace(
        dp_metadata=SimpleNamespace(num_tokens_across_dp_cpu=torch.tensor([30, 20]))
    )
    monkeypatch.setattr(mori_common, "is_forward_context_available", lambda: True)
    monkeypatch.setattr(mori_common, "get_forward_context", lambda: ctx)
    monkeypatch.setattr(
        mori_common, "get_dp_group", lambda: SimpleNamespace(rank_in_group=0)
    )
    # TP=2 per DP rank: 2 * (30 + 20) = 100 -> 128 rows.
    assert mori_common.mori_recv_rows_bound(30, 256, 4) == 128
    # Dispatchers not a multiple of DP ranks: dispatchers * largest batch.
    assert mori_common.mori_recv_rows_bound(30, 256, 3) == 96
