# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Ported from vllm-torchtpu's tests/test_prefix_aware_scheduler.py,
# Copyright 2026 Google LLC, licensed under the Apache License, Version 2.0.

import time
from types import SimpleNamespace
from unittest.mock import MagicMock

from vllm.v1.core.sched.request_queue import FCFSRequestQueue
from vllm.v1.core.sched.shortest_prefill_first import ShortestPrefillFirstAdmission


def _make_request(
    request_id: str,
    num_tokens: int = 1000,
    arrival_time: float = 0.0,
    num_preemptions: int = 0,
    num_computed_tokens: int = 0,
    skip_reading_prefix_cache: bool = False,
) -> SimpleNamespace:
    hit_limit = 0 if skip_reading_prefix_cache else None
    return SimpleNamespace(
        request_id=request_id,
        num_tokens=num_tokens,
        arrival_time=arrival_time,
        num_preemptions=num_preemptions,
        num_computed_tokens=num_computed_tokens,
        get_prefix_cache_hit_limit=lambda: hit_limit,
        block_hashes=[object()],
    )


def _make_admission(
    requests: list[SimpleNamespace],
    warmth_map: dict[str, int],
    window: int = 64,
    enable_caching: bool = True,
    raise_on_lookup: bool = False,
) -> tuple[ShortestPrefillFirstAdmission, FCFSRequestQueue]:
    queue = FCFSRequestQueue()
    for r in requests:
        queue.add_request(r)

    def find_hit(block_hashes, max_len):
        if raise_on_lookup:
            raise RuntimeError("simulated coordinator failure")
        for r in requests:
            if r.block_hashes is block_hashes:
                return ({}, warmth_map.get(r.request_id, 0), 0)
        return ({}, 0, 0)

    kvm = MagicMock()
    kvm.enable_caching = enable_caching
    kvm.coordinator.find_longest_cache_hit.side_effect = find_hit

    admission = ShortestPrefillFirstAdmission(kvm)
    admission.WINDOW = window
    return admission, queue


def _promoted_order(admission, queue) -> list[str]:
    admission.promote(queue)
    return [r.request_id for r in queue]


def test_default_knobs():
    assert ShortestPrefillFirstAdmission.WINDOW == 64


def test_promotes_warmest_within_window():
    now = time.time()
    r_cold = _make_request("cold", num_tokens=1000, arrival_time=now - 10)
    r_warm = _make_request("warm", num_tokens=1000, arrival_time=now - 5)
    admission, queue = _make_admission([r_cold, r_warm], {"cold": 0, "warm": 800})
    assert _promoted_order(admission, queue) == ["warm", "cold"]


def test_ranks_by_uncached_tokens_not_hit_ratio():
    now = time.time()
    # 100k tokens, 90% cached -> 10k uncached tokens
    r_long = _make_request("long", num_tokens=100_000, arrival_time=now - 10)
    # 20k tokens, 80% cached -> 4k uncached tokens (fewer uncached tokens wins)
    r_short = _make_request("short", num_tokens=20_000, arrival_time=now - 5)
    admission, queue = _make_admission(
        [r_long, r_short], {"long": 90_000, "short": 16_000}
    )
    assert _promoted_order(admission, queue) == ["short", "long"]


def test_fcfs_order_preserved_when_all_cold():
    now = time.time()
    r1 = _make_request("r1", num_tokens=1000, arrival_time=now - 10)
    r2 = _make_request("r2", num_tokens=1000, arrival_time=now - 5)
    admission, queue = _make_admission([r1, r2], {"r1": 0, "r2": 0})
    assert _promoted_order(admission, queue) == ["r1", "r2"]


def test_preempted_tier_outranks_normal():
    now = time.time()
    r_preempted = _make_request(
        "preempted", num_tokens=1000, arrival_time=now - 5, num_preemptions=1
    )
    r_warm = _make_request("warm", num_tokens=1000, arrival_time=now - 10)
    admission, queue = _make_admission(
        [r_preempted, r_warm], {"preempted": 0, "warm": 900}
    )
    assert _promoted_order(admission, queue) == ["preempted", "warm"]


def test_window_bounds_evaluation():
    now = time.time()
    r1 = _make_request("r1", num_tokens=1000, arrival_time=now - 15)
    r2 = _make_request("r2", num_tokens=1000, arrival_time=now - 10)
    r_outside = _make_request("outside", num_tokens=1000, arrival_time=now - 5)
    # window = 2, so r_outside is beyond the inspection window
    admission, queue = _make_admission(
        [r1, r2, r_outside], {"r1": 100, "r2": 200, "outside": 999}, window=2
    )
    assert _promoted_order(admission, queue)[0] == "r2"


def test_single_or_empty_queue_is_noop():
    admission, queue = _make_admission([_make_request("single")], {"single": 500})
    assert _promoted_order(admission, queue) == ["single"]

    admission, queue = _make_admission([], {})
    assert _promoted_order(admission, queue) == []


def test_uses_num_computed_tokens_when_present():
    now = time.time()
    r1 = _make_request("r1", num_tokens=1000, arrival_time=now - 10)
    r2 = _make_request(
        "r2", num_tokens=1000, arrival_time=now - 5, num_computed_tokens=900
    )
    admission, queue = _make_admission([r1, r2], {"r1": 0, "r2": 0})
    assert _promoted_order(admission, queue) == ["r2", "r1"]


def test_skip_reading_prefix_cache_or_caching_disabled():
    now = time.time()
    r1 = _make_request("r1", num_tokens=1000, arrival_time=now - 10)
    r2 = _make_request(
        "r2", num_tokens=1000, arrival_time=now - 5, skip_reading_prefix_cache=True
    )
    admission, queue = _make_admission([r1, r2], {"r1": 0, "r2": 900})
    assert _promoted_order(admission, queue) == ["r1", "r2"]

    admission, queue = _make_admission(
        [r1, _make_request("r3", num_tokens=1000, arrival_time=now - 5)],
        {"r1": 0, "r3": 900},
        enable_caching=False,
    )
    assert _promoted_order(admission, queue) == ["r1", "r3"]


def test_coordinator_failure_falls_back_to_fcfs():
    now = time.time()
    r1 = _make_request("r1", num_tokens=1000, arrival_time=now - 10)
    r2 = _make_request("r2", num_tokens=1000, arrival_time=now - 5)
    admission, queue = _make_admission(
        [r1, r2], {"r1": 0, "r2": 500}, raise_on_lookup=True
    )
    assert _promoted_order(admission, queue) == ["r1", "r2"]
    assert admission._scoring_broken is True
