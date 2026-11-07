# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the Dynamic SD schedule computation (pure, no GPU)."""

import json
from pathlib import Path

import pytest

from vllm.v1.spec_decode.dynamic.schedule import (
    build_schedule,
    compress,
    expand,
    goodput_table,
    interp,
    load_acceptance,
    load_costs,
    make_schedule,
    tokens_per_step,
)


def test_interp_flat_below_first_and_monotone_above_last():
    xs, ys = [1, 8, 16], [10.0, 20.0, 30.0]
    assert interp(xs, ys, 0) == 10.0
    assert interp(xs, ys, 1) == 10.0
    assert interp(xs, ys, 4) == pytest.approx(10.0 + 10.0 * 3 / 7)
    # a falling last pair must not extrapolate downwards (division by zero guard)
    ys2 = [10.0, 30.0, 20.0]
    assert interp(xs, ys2, 100) == 20.0


def test_tokens_per_step():
    # accept_by_pos[i] = P(first i+1 drafts accepted)
    tps = tokens_per_step([0.8, 0.5])
    assert tps == {0: 1.0, 1: 1.8, 2: 2.3}
    # n-gram style: drafts on half the steps
    tps = tokens_per_step([0.8, 0.5], proposal_rate=0.5)
    assert tps == {0: 1.0, 1: 1.4, 2: 1.65}


def test_goodput_prefers_long_drafts_when_free():
    # same step time for every K: the longest draft always wins
    costs = {1: ([1, 8], [10.0, 10.0]), 2: ([1, 8], [10.0, 10.0])}
    g = goodput_table(costs, tokens_per_step([0.8, 0.5]), 8)
    assert all(max(g[b], key=g[b].get) == 2 for b in g)


def test_goodput_prefers_short_drafts_when_costly():
    # K=2 step twice as slow: K=1 wins wherever acceptance does not pay for it
    costs = {1: ([1, 8], [10.0, 10.0]), 2: ([1, 8], [20.0, 20.0])}
    g = goodput_table(costs, tokens_per_step([0.4, 0.1]), 8)
    # tokens/step: K=1 -> 1.4, K=2 -> 1.5; 1.4/10 > 1.5/20
    assert all(max(g[b], key=g[b].get) == 1 for b in g)


def test_build_schedule_eps_keeps_current_k():
    # K=2 wins at B=1; at B=2 it is only 0.3% better: below eps, no switch
    g = {1: {1: 100.0, 2: 101.0}, 2: {1: 100.0, 2: 100.3}}
    assert build_schedule(g, eps=0.005) == {1: 2, 2: 2}
    # 2% better: above eps, it switches
    g = {1: {1: 100.0, 2: 101.0}, 2: {1: 103.0, 2: 101.0}}
    assert build_schedule(g, eps=0.005) == {1: 2, 2: 1}


def test_build_schedule_merges_narrow_ranges():
    # K=2 wins only at B=5 (width 1 < 25% of 5): the range must merge away
    g = {b: {1: 100.0, 2: 90.0} for b in range(1, 9)}
    g[5] = {1: 100.0, 2: 200.0}
    best = build_schedule(g, eps=0.005, min_width_frac=0.25)
    assert set(best.values()) == {1}
    # without the width rule the narrow win would survive
    best = build_schedule(g, eps=0.005, min_width_frac=0.0)
    assert best[5] == 2


def test_compress_and_expand_round_trip():
    schedule = compress({1: 3, 2: 3, 3: 1, 4: 1})
    assert schedule == [[1, 2, 3], [3, 4, 1]]
    # gaps and the tail carry the previous K forward, like vLLM's lookup
    assert expand([[1, 4, 3], [8, 8, 1]], 10) == [0, 3, 3, 3, 3, 3, 3, 3, 1, 1, 1]


def test_make_schedule_restricts_allowed_ks():
    # K=2 is best everywhere, but only K=1 has a CUDA graph: the table uses 1
    costs = {1: ([1, 8], [10.0, 10.0]), 2: ([1, 8], [10.0, 10.0])}
    assert make_schedule(costs, [0.8, 0.5], 8, allowed_ks=[1]) == [[1, 8, 1]]
    assert make_schedule(costs, [0.8, 0.5], 8) == [[1, 8, 2]]


def test_load_costs_validates(tmp_path: Path):
    f = tmp_path / "costs_k1.json"
    f.write_text(
        json.dumps(
            {
                "rows": [
                    {"batch_size": 8, "k": 1, "step_ms": 10.0},
                    {"batch_size": 1, "k": 1, "step_ms": 5.0},
                ]
            }
        )
    )
    costs = load_costs(str(f))
    assert costs == {1: ([1, 8], [5.0, 10.0])}
    bad = tmp_path / "bad.json"
    bad.write_text(
        json.dumps(
            {
                "rows": [
                    {"batch_size": 8, "k": 1, "step_ms": 10.0},
                    {"batch_size": 8, "k": 1, "step_ms": 10.0},
                ]
            }
        )
    )
    with pytest.raises(ValueError, match="strictly increasing"):
        load_costs(str(bad))


def test_load_acceptance(tmp_path: Path):
    f = tmp_path / "acceptance.json"
    f.write_text(json.dumps({"accept_by_pos": [0.8, 0.5], "proposal_rate": 0.99}))
    accept, rate = load_acceptance(f)
    assert accept == [0.8, 0.5]
    assert rate == 1.0  # >= 0.98 counts as full
    f.write_text(json.dumps({"accept_by_pos": [0.8, 0.5], "proposal_rate": 0.5}))
    assert load_acceptance(f)[1] == 0.5
    f.write_text(json.dumps({"accept_by_pos": [1.5]}))
    with pytest.raises(ValueError, match="in \\[0, 1\\]"):
        load_acceptance(f)
