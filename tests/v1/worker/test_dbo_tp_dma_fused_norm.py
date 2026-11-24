# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""`dbo_tp_all_reduce_add_rmsnorm_mxfp8`: the SDMA all-reduce fused with the
residual add + RMSNorm + MXFP8 quant of a DBO microbatch. Where it applies, and
the hand-off around its push, which must be the one of
`GroupCoordinator._all_reduce_out_place`."""

import threading
from types import SimpleNamespace

import pytest
import torch

from vllm.distributed import parallel_state
from vllm.v1.worker import ubatching

SCHEDULE = "all-reduce-overlap"
ROWS, HIDDEN = 4, 64


class _FakeDma:
    def __init__(self, calls):
        self.calls = calls

    def should_reduce(self, x):
        return True

    def push(self, x):
        self.calls.append("push")
        return "pending"

    def add_rmsnorm_mxfp8(self, pending, residual, weight, eps, also_bf16):
        assert pending == "pending"
        self.calls.append("add_rmsnorm_mxfp8")
        return "fused"


@pytest.fixture
def microbatch(monkeypatch):
    """A microbatch thread of a TP2 DBO step with the fused path on; the
    hand-off functions and the SDMA all-reduce record their calls."""
    calls = []
    comm = SimpleNamespace(dma_ar=_FakeDma(calls))
    monkeypatch.setattr(
        ubatching, "envs", SimpleNamespace(VLLM_DBO_TP_DMA_AR_FUSED_NORM=True)
    )
    monkeypatch.setattr(ubatching, "_THREAD_ID_TO_CONTEXT", {threading.get_ident(): 0})
    monkeypatch.setattr(
        parallel_state,
        "get_tp_group",
        lambda: SimpleNamespace(device_communicator=comm),
    )
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    monkeypatch.setattr(
        ubatching, "dbo_switch_to_comm_sync", lambda s=None: calls.append(("comm", s))
    )
    monkeypatch.setattr(
        ubatching,
        "dbo_yield_and_switch_from_comm_to_compute",
        lambda s=None: calls.append(("compute", s)),
    )
    return SimpleNamespace(calls=calls, comm=comm, monkeypatch=monkeypatch)


def _fused(x, residual):
    weight = torch.ones(HIDDEN, dtype=torch.bfloat16)
    return ubatching.dbo_tp_all_reduce_add_rmsnorm_mxfp8(
        x, residual, weight, 1e-6, False
    )


def test_fused_norm_hands_off_around_the_push(microbatch):
    x = torch.zeros(ROWS, HIDDEN, dtype=torch.bfloat16)
    assert _fused(x, torch.zeros_like(x)) == "fused"
    # Launch the push on the comm stream, let the other microbatch run while it
    # is in flight, then consume it on the compute stream.
    assert microbatch.calls == [
        ("comm", SCHEDULE),
        "push",
        ("compute", SCHEDULE),
        "add_rmsnorm_mxfp8",
    ]


@pytest.mark.parametrize(
    "case",
    [
        "flag off",
        "not a microbatch",
        "no SDMA all-reduce",
        "graph capture",
        "1-D input",
        "residual shape",
        "residual dtype",
        "strided residual",
    ],
)
def test_fused_norm_declines_before_enqueueing_anything(microbatch, case):
    mp = microbatch.monkeypatch
    x = torch.zeros(ROWS, HIDDEN, dtype=torch.bfloat16)
    residual = torch.zeros_like(x)
    if case == "flag off":
        mp.setattr(
            ubatching, "envs", SimpleNamespace(VLLM_DBO_TP_DMA_AR_FUSED_NORM=False)
        )
    elif case == "not a microbatch":
        mp.setattr(ubatching, "_THREAD_ID_TO_CONTEXT", {})
    elif case == "no SDMA all-reduce":
        microbatch.comm.dma_ar = None
    elif case == "graph capture":
        mp.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    elif case == "1-D input":
        x, residual = x.flatten(), residual.flatten()
    elif case == "residual shape":
        residual = residual[:-1]
    elif case == "residual dtype":
        residual = residual.float()
    elif case == "strided residual":
        residual = torch.zeros(HIDDEN, ROWS, dtype=torch.bfloat16).t()
    assert _fused(x, residual) is None
    assert microbatch.calls == []
