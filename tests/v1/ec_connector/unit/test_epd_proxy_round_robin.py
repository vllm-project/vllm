# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Encoder fan-out fairness in the disaggregated EPD proxy.

Exercises the REAL ``encoder_rr_assignment`` helper loaded from its
``examples/`` path -- no routing logic is re-implemented here, so a future
change to that function is what these tests exercise.

Regression target: the fan-out cursor used to restart at e_urls[0] on every
incoming request (``e_urls[i % len(e_urls)] for i in range(len(mm_items))``),
so single-item requests -- the common case -- always hit the first encoder
instance and left the rest idle. ``encoder_rr_assignment`` threads a cursor
across calls so consecutive requests keep advancing through every URL.
"""

import importlib.util
from collections import Counter
from pathlib import Path

import pytest

PROXY_REL = "examples/disaggregated/disaggregated_encoder/disagg_epd_proxy.py"


def _load_proxy_module():
    path = Path(__file__).parents[4] / PROXY_REL
    spec = importlib.util.spec_from_file_location("disagg_epd_proxy_under_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def assign():
    return _load_proxy_module().encoder_rr_assignment


def _drive(assign, e_urls, counts):
    """Feed a sequence of request item-counts through the cursor, as
    fanout_encoder_primer does one request at a time."""
    cursor = 0
    all_urls = []
    for count in counts:
        urls, cursor = assign(e_urls, cursor, count)
        assert len(urls) == count
        all_urls.append(urls)
    return all_urls


@pytest.mark.parametrize("n_urls", [1, 2, 3, 5])
def test_full_url_space_is_covered_uniformly(assign, n_urls):
    e_urls = [f"E{i}" for i in range(n_urls)]
    # Three full cycles of single-item requests.
    urls = _drive(assign, e_urls, counts=[1] * (n_urls * 3))
    hits = Counter(u for req in urls for u in req)
    assert set(hits) == set(e_urls)
    assert max(hits.values()) == min(hits.values())


def test_single_item_requests_rotate_through_all_encoders(assign):
    # Exact scenario from the bug report: 3 encoders, one image/request.
    e_urls = ["E0", "E1", "E2"]
    routed = _drive(assign, e_urls, counts=[1] * 6)
    assert [urls[0] for urls in routed] == ["E0", "E1", "E2", "E0", "E1", "E2"]


def test_cursor_stays_contiguous_across_varying_item_counts(assign):
    e_urls = ["E0", "E1", "E2"]
    routed = _drive(assign, e_urls, counts=[2, 1, 3, 1])
    assert routed == [
        ["E0", "E1"],
        ["E2"],
        ["E0", "E1", "E2"],
        ["E0"],
    ]


def test_single_encoder_always_resolves_to_it(assign):
    urls, next_cursor = assign(["E0"], 0, 4)
    assert urls == ["E0"] * 4
    assert next_cursor == 0
