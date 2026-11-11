# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Any
from unittest.mock import MagicMock

from vllm.v1.core.sched import shortest_prefill_first as spf
from vllm.v1.core.sched.request_queue import FCFSRequestQueue
from vllm.v1.core.sched.shortest_prefill_first import ShortestPrefillFirstOrder


class _Req:
    """Hashable stand-in for Request (queues keep requests in sets)."""

    def __init__(self, **fields):
        self.__dict__.update(fields)


def _req(
    request_id: str,
    num_tokens: int = 1000,
    num_computed_tokens: int = 0,
    num_prompt_tokens: int | None = None,
    skip_reading_external_kv: bool = False,
) -> Any:
    return _Req(
        request_id=request_id,
        num_tokens=num_tokens,
        num_prompt_tokens=num_tokens
        if num_prompt_tokens is None
        else num_prompt_tokens,
        num_computed_tokens=num_computed_tokens,
        skip_reading_external_kv=skip_reading_external_kv,
    )


def _order(external_hits: dict[str, int | None] | None) -> ShortestPrefillFirstOrder:
    connector = None
    if external_hits is not None:
        connector = MagicMock()
        connector.get_known_num_matched_tokens.side_effect = lambda r: (
            external_hits.get(r.request_id)
        )
    return ShortestPrefillFirstOrder(connector)


def _queue(requests) -> FCFSRequestQueue:
    queue = FCFSRequestQueue()
    for r in requests:
        queue.add_request(r)
    return queue


def _ids(requests) -> list[str]:
    return [r.request_id for r in requests]


def _reorder(order, running=(), holding=(), waiting=()):
    running = list(running)
    holding_q, waiting_q, async_q = _queue(holding), _queue(waiting), _queue([])
    order.reorder(running, holding_q, (waiting_q, async_q))
    return _ids(running), _ids(holding_q), _ids(waiting_q)


def test_running_prefills_sorted_decodes_keep_positions():
    running = [
        _req("big", num_tokens=9000, num_computed_tokens=1000),  # 8000 left
        _req("decode", num_tokens=501, num_computed_tokens=500, num_prompt_tokens=400),
        _req("small", num_tokens=3000, num_computed_tokens=2500),  # 500 left
        _req("mid", num_tokens=5000, num_computed_tokens=3000),  # 2000 left
    ]
    ids, _, _ = _reorder(_order(None), running=running)
    assert ids == ["small", "decode", "mid", "big"]


def test_kv_holding_sorted_by_tokens_left_to_compute():
    holding = [
        _req("h1", num_tokens=100_000, num_computed_tokens=90_000),
        _req("h2", num_tokens=100_000, num_computed_tokens=99_000),
        _req("h3", num_tokens=50_000, num_computed_tokens=45_000),
    ]
    _, ids, _ = _reorder(_order({}), holding=holding)
    assert ids == ["h2", "h3", "h1"]


def test_waiting_known_hits_sorted_unknown_keep_positions():
    waiting = [
        _req("unknown_a", num_tokens=1000),
        _req("cold", num_tokens=1000),  # known hit 0 -> 1000 to compute
        _req("unknown_b", num_tokens=1000),
        _req("warm", num_tokens=1000),  # 100 to compute
        _req("skip", num_tokens=1000, skip_reading_external_kv=True),
        _req("mid", num_tokens=1000),  # 400 to compute
    ]
    hits = {"cold": 0, "warm": 900, "mid": 600, "skip": 999}
    _, _, ids = _reorder(_order(hits), waiting=waiting)
    assert ids == ["unknown_a", "warm", "unknown_b", "mid", "skip", "cold"]


def test_full_hit_still_computes_last_token():
    waiting = [_req("full", num_tokens=1000), _req("near", num_tokens=1000)]
    _, _, ids = _reorder(_order({"full": 1000, "near": 998}), waiting=waiting)
    assert ids == ["full", "near"]  # 1 vs 2 tokens left


def test_ties_and_already_sorted_keep_order():
    waiting = [_req("a"), _req("b"), _req("c")]
    _, _, ids = _reorder(_order({"a": 500, "b": 500, "c": 900}), waiting=waiting)
    assert ids == ["c", "a", "b"]
    _, _, ids = _reorder(_order({"a": 900, "b": 500, "c": 500}), waiting=waiting)
    assert ids == ["a", "b", "c"]


def test_no_connector_leaves_waiting_untouched():
    waiting = [_req("a"), _req("b", num_tokens=10)]
    _, _, ids = _reorder(_order(None), waiting=waiting)
    assert ids == ["a", "b"]


def test_waiting_window_bounds_reordering(monkeypatch):
    monkeypatch.setattr(spf, "WINDOW", 2)
    waiting = [_req("a"), _req("b"), _req("c")]
    _, _, ids = _reorder(_order({"a": 0, "b": 500, "c": 999}), waiting=waiting)
    assert ids == ["b", "a", "c"]


def test_queue_order_preserved_end_to_end():
    """Unsorted tail beyond the window stays behind the reordered head."""
    waiting = [_req(str(i)) for i in range(5)]
    hits = {"0": 0, "1": 100, "2": 900}
    _, _, ids = _reorder(_order(hits), waiting=waiting)
    assert ids == ["2", "1", "0", "3", "4"]
