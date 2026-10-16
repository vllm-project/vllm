# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Which speculative drafts a structured-output request may accept.

`StructuredOutputManager.grammar_bitmask` fills row i before it inspects
`req_tokens[i]`, so with the first -1 placeholder at index j rows 0..j carry a
real mask and rows j+1.. carry the all-permissive `_full_mask`. Drafts j..K-1
must therefore be rejected, or the request samples with no grammar constraint
at a position the model really does sample.
"""

from types import SimpleNamespace

import numpy as np
import torch

from vllm.v1.worker.gpu.spec_decode.utils import DraftTokensHandler
from vllm.v1.worker.gpu.structured_outputs import grammar_invalid_drafts


def _input_batch(req_ids, num_drafts, num_admitted=None):
    """`num_admitted` is what adaptive verification kept on device."""
    num_admitted = num_drafts if num_admitted is None else num_admitted
    rows = [n + 1 for n in num_admitted]
    return SimpleNamespace(
        req_ids=list(req_ids),
        num_reqs=len(req_ids),
        num_draft_tokens=sum(num_drafts),
        num_draft_tokens_per_req=np.array(num_drafts, dtype=np.int32),
        cu_num_logits=torch.tensor([0, *np.cumsum(rows)], dtype=torch.int32),
        expanded_local_pos=torch.cat(
            [torch.arange(n, dtype=torch.int32) for n in rows]
        ),
    )


def _invalid_rows(batch, grammar_req_ids, num_acceptable):
    mask = grammar_invalid_drafts(batch, grammar_req_ids, num_acceptable)
    return None if mask is None else mask.nonzero().flatten().tolist()


def test_reject_whole_window_when_nothing_was_backfilled():
    # Request "g" is at rows 0..3 (1 + 3 drafts), so its drafts are rows 1..3.
    batch = _input_batch(["g", "p"], [3, 3])
    assert _invalid_rows(batch, ["g"], [0]) == [1, 2, 3]


def test_keep_the_drafts_the_bitmask_could_see():
    batch = _input_batch(["g", "p"], [3, 3])
    assert _invalid_rows(batch, ["g"], [2]) == [3]


def test_follow_the_device_layout_under_adaptive_verification():
    # Scheduled 3 drafts each, but adaptive verification admitted 2 for "a" and
    # 3 for "b": the real rows are a=0..2, b=3..6, not the scheduled 0..3, 4..7.
    batch = _input_batch(["a", "b"], [3, 3], num_admitted=[2, 3])
    assert _invalid_rows(batch, ["a", "b"], [1, 0]) == [2, 4, 5, 6]


# --- DraftTokensHandler per-request retention (no CUDA: dict logic only) ---


def _handler(max_num_reqs: int = 8):
    # Skip __init__ (creates CUDA stream/event); the tests exercise the
    # retention logic, with a no-op event standing in for the copy sync.
    h = object.__new__(DraftTokensHandler)
    h.max_num_reqs = max_num_reqs
    h._draft_tokens_by_req = {}
    h._copy_pending = False
    h.copy_event = SimpleNamespace(synchronize=lambda: None)
    return h


def test_each_request_gets_its_own_drafts_back():
    # PP overlap: "g" was sampled in an earlier batch; batches in between
    # must not shadow it. Fetch returns each request's own row.
    h = _handler()
    h._draft_tokens_by_req = {
        "g": np.array([10, 11]),
        "x": np.array([12, 13]),
        "y": np.array([14, 15]),
    }
    out = h.get_draft_tokens()
    got = dict(zip(out.req_ids, out.draft_token_ids))
    assert got["g"] == [10, 11]
    assert got["x"] == [12, 13]
    assert got["y"] == [14, 15]


def test_unstructured_width_placeholder_keeps_structured_drafts():
    # A batch with no structured reqs stores only the draft width per
    # request; an earlier structured batch's real drafts are preserved and
    # the unstructured request gets a -1 placeholder of the right width.
    h = _handler()
    h._draft_tokens_by_req = {"a": [7, 8], "b": 2}
    out = h.get_draft_tokens()
    got = dict(zip(out.req_ids, out.draft_token_ids))
    assert got["a"] == [7, 8]
    assert got["b"] == [-1, -1]


def test_discard_drops_only_the_named_request():
    # discard() on a re-added (or finished) request must not touch others.
    h = _handler()
    h._draft_tokens_by_req = {"x": [1, 1], "keep": [2, 2]}
    h.discard("x")
    assert "x" not in h._draft_tokens_by_req
    assert h._draft_tokens_by_req["keep"] == [2, 2]


def test_request_bound_asserts_when_exceeded():
    # Retention is bounded by max_num_reqs: discard-on-remove keeps the dict
    # at or below the live request count; exceeding it is a bug and asserts.
    h = _handler(max_num_reqs=2)
    h._draft_tokens_by_req = {"a": [1, 1], "b": [2, 2], "c": [3, 3]}
    try:
        h._assert_request_bound()
        raised = False
    except AssertionError:
        raised = True
    assert raised, "expected the max_num_reqs bound to assert"
