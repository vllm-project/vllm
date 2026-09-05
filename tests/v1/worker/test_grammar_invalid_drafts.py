# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Which speculative drafts a structured-output request may accept.

`StructuredOutputManager.grammar_bitmask` fills row i before it inspects
`req_tokens[i]`, so with the first -1 placeholder at index j rows 0..j carry a
real mask and rows j+1.. carry the all-permissive `_full_mask`. Drafts j..K-1
must therefore be rejected, or the request samples with no grammar constraint
at a position the model really does sample.
"""

from unittest.mock import Mock

import numpy as np
import torch

from vllm.v1.core.sched.scheduler import compute_num_acceptable_drafts
from vllm.v1.worker.gpu.model_runner import grammar_invalid_draft_positions


def test_num_acceptable_drafts_no_placeholder():
    assert compute_num_acceptable_drafts(["a"], {"a": [11, 12, 13, 14]}) == [4]


def test_num_acceptable_drafts_all_placeholders():
    assert compute_num_acceptable_drafts(["a"], {"a": [-1, -1, -1, -1]}) == [0]


def test_num_acceptable_drafts_partial_backfill():
    # validate_tokens truncated the window and padded the rest with -1.
    assert compute_num_acceptable_drafts(["a"], {"a": [11, 12, -1, -1]}) == [2]


def test_num_acceptable_drafts_missing_request():
    assert compute_num_acceptable_drafts(["a"], {}) == [0]


def test_num_acceptable_drafts_orders_by_request_id():
    spec = {"a": [11, -1, -1], "b": [21, 22, 23], "c": [-1, -1, -1]}
    assert compute_num_acceptable_drafts(["b", "c", "a"], spec) == [3, 0, 1]


def _input_batch(req_ids, num_logits_per_req, num_draft_tokens=6):
    cu = np.concatenate([[0], np.cumsum(num_logits_per_req)]).astype(np.int32)
    return Mock(
        req_ids=list(req_ids),
        cu_num_logits_np=cu,
        num_draft_tokens=num_draft_tokens,
    )


def _positions(batch, grammar_req_ids, num_acceptable):
    out = grammar_invalid_draft_positions(
        batch, grammar_req_ids, num_acceptable, torch.device("cpu")
    )
    return None if out is None else out.tolist()


def test_positions_reject_whole_window_when_nothing_was_backfilled():
    # Two requests, 4 logits rows each (1 + 3 drafts). Request "g" is at rows
    # 0..3, so its drafts sit at draft_sampled[1..3].
    batch = _input_batch(["g", "p"], [4, 4])
    assert _positions(batch, ["g"], [0]) == [1, 2, 3]


def test_positions_keep_the_drafts_the_bitmask_could_see():
    batch = _input_batch(["g", "p"], [4, 4])
    # Two real drafts: they were masked correctly and may be accepted.
    assert _positions(batch, ["g"], [2]) == [3]


def test_positions_reject_nothing_when_fully_backfilled():
    batch = _input_batch(["g", "p"], [4, 4])
    assert _positions(batch, ["g"], [3]) is None


def test_positions_are_offset_per_request():
    batch = _input_batch(["p", "g"], [4, 4])
    # "g" starts at logits row 4, so its drafts are draft_sampled[5..7].
    assert _positions(batch, ["g"], [0]) == [5, 6, 7]


def test_positions_fall_back_to_the_whole_window_without_the_field():
    # An older scheduler, or warmup, supplies no num_acceptable_drafts; be
    # conservative rather than accepting drafts whose mask is unknown.
    batch = _input_batch(["g"], [4])
    assert _positions(batch, ["g"], None) == [1, 2, 3]


def test_no_positions_without_drafts_or_grammar_requests():
    batch = _input_batch(["g"], [4], num_draft_tokens=0)
    assert _positions(batch, ["g"], [0]) is None
    batch = _input_batch(["g"], [4])
    assert _positions(batch, [], []) is None


def test_unknown_request_is_skipped():
    batch = _input_batch(["p"], [4])
    assert _positions(batch, ["g"], [0]) is None
