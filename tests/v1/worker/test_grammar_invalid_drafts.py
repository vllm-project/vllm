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

from vllm.v1.worker.gpu.model_runner import grammar_invalid_drafts


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


def test_reject_nothing_when_fully_backfilled():
    batch = _input_batch(["g", "p"], [3, 3])
    assert _invalid_rows(batch, ["g"], [3]) is None


def test_rows_are_offset_per_request():
    batch = _input_batch(["p", "g"], [3, 3])
    assert _invalid_rows(batch, ["g"], [0]) == [5, 6, 7]


def test_follow_the_device_layout_under_adaptive_verification():
    # Scheduled 3 drafts each, but adaptive verification admitted 2 for "a" and
    # 3 for "b": the real rows are a=0..2, b=3..6, not the scheduled 0..3, 4..7.
    batch = _input_batch(["a", "b"], [3, 3], num_admitted=[2, 3])
    assert _invalid_rows(batch, ["a", "b"], [1, 0]) == [2, 4, 5, 6]


def test_fall_back_to_the_whole_window_without_the_field():
    # An older scheduler, or warmup, supplies no num_acceptable_drafts; be
    # conservative rather than accepting drafts whose mask is unknown.
    batch = _input_batch(["g"], [3])
    assert _invalid_rows(batch, ["g"], None) == [1, 2, 3]


def test_nothing_without_drafts_or_grammar_requests():
    batch = _input_batch(["g"], [3])
    batch.num_draft_tokens = 0
    assert _invalid_rows(batch, ["g"], [0]) is None
    assert _invalid_rows(_input_batch(["g"], [3]), [], []) is None


def test_unknown_request_is_skipped():
    batch = _input_batch(["p"], [3])
    assert _invalid_rows(batch, ["g"], [0]) is None
