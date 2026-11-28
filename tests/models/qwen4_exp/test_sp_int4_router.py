# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The Qwen4Exp decoder layer's sequence-parallel collectives take AITER's INT4
ones where AITER serves them. CPU only: AITER's router and the TP group are
stand-ins."""

from types import SimpleNamespace

import pytest
import torch

from vllm.models.common.ops import sequence_parallel as common_sp
from vllm.models.qwen4_exp.amd.ops import sequence_parallel as sp

TP_SIZE = 2
HIDDEN = 16

INT4_PAIR = (sp.sp_all_gather_int4, sp.sp_reduce_scatter_int4)
COMMON_PAIR = (common_sp.sp_all_gather, sp.sp_reduce_scatter)


class _Router:
    """AITER's `SequenceParallelCollectives`: returns a marked tensor where it
    serves a call and None where it does not."""

    def __init__(self, serves: bool):
        self.serves = serves
        self.reduce_scatter_inputs: list[torch.Tensor] = []
        self.all_gather_inputs: list[torch.Tensor] = []

    def reduce_scatter(self, x):
        self.reduce_scatter_inputs.append(x)
        return (
            x.new_full((x.shape[0] // TP_SIZE, x.shape[1]), 1.0)
            if self.serves
            else None
        )

    def all_gather(self, x):
        self.all_gather_inputs.append(x)
        return (
            x.new_full((x.shape[0] * TP_SIZE, x.shape[1]), 1.0) if self.serves else None
        )


class _Tp:
    """The TP group's quick-reduce communicator, counting router requests."""

    def __init__(self, router, *, has_qr_comm=True):
        self.requests = 0
        self.router = router
        qr_comm = SimpleNamespace(sp_collectives=self._sp_collectives)
        self.group = SimpleNamespace(
            device_communicator=SimpleNamespace(
                qr_comm=qr_comm if has_qr_comm else None
            )
        )

    def _sp_collectives(self):
        self.requests += 1
        return self.router


@pytest.fixture
def common_calls(monkeypatch):
    """The common collectives, recording their inputs and returning marked tensors."""
    calls: dict[str, list[torch.Tensor]] = {"reduce_scatter": [], "all_gather": []}

    def reduce_scatter(x):
        calls["reduce_scatter"].append(x)
        return x.new_full((x.shape[0] // TP_SIZE, x.shape[1]), 2.0)

    def all_gather(x):
        calls["all_gather"].append(x)
        return x.new_full((x.shape[0] * TP_SIZE, x.shape[1]), 2.0)

    monkeypatch.setattr(common_sp, "sp_reduce_scatter", reduce_scatter)
    monkeypatch.setattr(common_sp, "sp_all_gather", all_gather)
    monkeypatch.setattr(sp, "get_tensor_model_parallel_world_size", lambda: TP_SIZE)
    return calls


def _install(monkeypatch, tp: _Tp) -> None:
    monkeypatch.setattr(sp, "get_tp_group", lambda: tp.group)


def _config(*, ubatching: bool, enable_hc_sp: bool):
    return SimpleNamespace(
        parallel_config=SimpleNamespace(
            use_ubatching=ubatching, enable_hc_sp=enable_hc_sp
        )
    )


# The ids name the state that keeps the exact pair: HC SP off, MoE SP without
# the flag, DBO (the INT4 launches need one launch order), an AITER without a
# router for this platform, and no quick-reduce communicator.
@pytest.mark.parametrize(
    ("hc_sp", "flag", "dbo", "router", "has_qr_comm", "pair", "asked"),
    [
        (True, True, False, _Router(True), True, INT4_PAIR, 1),
        (False, True, False, _Router(True), True, COMMON_PAIR, 0),
        (True, False, False, _Router(True), True, COMMON_PAIR, 0),
        (True, True, True, _Router(True), True, COMMON_PAIR, 0),
        (True, True, False, None, True, COMMON_PAIR, 1),
        (True, True, False, None, False, COMMON_PAIR, 0),
    ],
    ids=["int4", "no_hc_sp", "moe_sp", "dbo", "no_router", "no_communicator"],
)
def test_pair_selection(
    monkeypatch, hc_sp, flag, dbo, router, has_qr_comm, pair, asked
):
    tp = _Tp(router, has_qr_comm=has_qr_comm)
    _install(monkeypatch, tp)

    chosen = sp.decoder_layer_sp_collectives(
        _config(ubatching=dbo, enable_hc_sp=flag), hc_sp
    )

    assert chosen == pair
    # The router is built by the first request, a collective: a model without HC
    # SP, with MoE SP only, or with DBO must never make it.
    assert tp.requests == asked


def test_reduce_scatter_takes_the_router_result(monkeypatch, common_calls):
    router = _Router(serves=True)
    _install(monkeypatch, _Tp(router))
    x = torch.randn(5, HIDDEN)

    out = sp.sp_reduce_scatter_int4(x)

    assert (out == 1.0).all() and out.shape == (3, HIDDEN)
    # AITER's kernels need equal blocks: the router sees the rows padded to a
    # multiple of the TP size
    (seen,) = router.reduce_scatter_inputs
    assert seen.shape == (6, HIDDEN) and torch.equal(seen[:5], x) and not seen[5:].any()
    assert common_calls["reduce_scatter"] == []


def test_reduce_scatter_continues_on_the_common_path_where_the_router_declines(
    monkeypatch, common_calls
):
    router = _Router(serves=False)
    _install(monkeypatch, _Tp(router))
    x = torch.randn(5, HIDDEN)

    out = sp.sp_reduce_scatter_int4(x)

    assert (out == 2.0).all()
    assert len(router.reduce_scatter_inputs) == 1
    (seen,) = common_calls["reduce_scatter"]
    assert seen.shape == (6, HIDDEN)


def test_all_gather_takes_the_router_result(monkeypatch, common_calls):
    router = _Router(serves=True)
    _install(monkeypatch, _Tp(router))
    x = torch.randn(3, HIDDEN)

    out = sp.sp_all_gather_int4(x)

    assert (out == 1.0).all() and out.shape == (6, HIDDEN)
    assert router.all_gather_inputs[0] is x
    assert common_calls["all_gather"] == []


def test_all_gather_continues_on_the_common_path_where_the_router_declines(
    monkeypatch, common_calls
):
    router = _Router(serves=False)
    _install(monkeypatch, _Tp(router))
    x = torch.randn(3, HIDDEN)

    out = sp.sp_all_gather_int4(x)

    assert (out == 2.0).all()
    assert len(router.all_gather_inputs) == 1 and router.all_gather_inputs[0] is x
    assert common_calls["all_gather"] == [x]
