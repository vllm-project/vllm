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


def _handler():
    # Skip __init__ (creates CUDA stream/event); the tests exercise the
    # retention logic, with a no-op event standing in for the copy sync.
    h = object.__new__(DraftTokensHandler)
    h.req_ids = []
    h.draft_tokens_np = None
    h.num_draft_tokens = 2
    h._latest = {}
    h._pending = []
    h.copy_event = SimpleNamespace(synchronize=lambda: None)
    return h


def _queue_batch(h, req_ids, drafts):
    h._pending.append((list(req_ids), np.array(drafts, dtype=np.int64)))


def test_each_request_gets_its_own_drafts_back():
    # PP overlap: "g" was sampled two batches ago; the batch in between must
    # not shadow it. Fold both, then fetch: "g" returns its own row.
    h = _handler()
    _queue_batch(h, ["g", "x"], [[10, 11], [12, 13]])
    _queue_batch(h, ["y"], [[14, 15]])
    out = h.get_draft_tokens()
    got = dict(zip(out.req_ids, out.draft_token_ids))
    assert got["g"] == [10, 11]
    assert got["x"] == [12, 13]
    assert got["y"] == [14, 15]


def test_unstructured_batch_gets_placeholders_retained_keep_drafts():
    # Latest batch took the fast path (no structured reqs): its request "b"
    # has no retained drafts -> placeholder; "a" from an earlier structured
    # batch keeps its real drafts.
    h = _handler()
    _queue_batch(h, ["a"], [[7, 8]])
    h.get_draft_tokens()
    h.req_ids = ["a", "b"]
    h.draft_tokens_np = None
    h.num_draft_tokens = 2
    out = h.get_draft_tokens()
    got = dict(zip(out.req_ids, out.draft_token_ids))
    assert got["a"] == [7, 8]
    assert got["b"] == [-1, -1]


def test_discard_folds_pending_before_popping():
    # discard() for a re-added request must see drafts already in flight:
    # fold first, then pop, or the request's first real drafts would be lost.
    h = _handler()
    _queue_batch(h, ["x", "keep"], [[1, 1], [2, 2]])
    h.discard("x")
    assert "x" not in h._latest
    assert h._latest["keep"] == [2, 2]
    assert h._pending == []


def test_retention_cap_evicts_least_recently_updated():
    h = _handler()
    n = DraftTokensHandler._MAX_RETAINED + 1
    rows = [[i, i] for i in range(n)]
    _queue_batch(h, [f"r{i}" for i in range(n)], rows)
    h.get_draft_tokens()
    assert len(h._latest) == DraftTokensHandler._MAX_RETAINED
    assert "r0" not in h._latest  # oldest evicted
    assert h._latest[f"r{n - 1}"] == [n - 1, n - 1]  # newest retained
