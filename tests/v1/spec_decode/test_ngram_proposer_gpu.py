# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for invalid draft accounting in the V1 GPU ngram proposer."""

import pytest
import torch

from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.spec_decode.ngram_proposer_gpu import update_scheduler_for_invalid_drafts


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("valid", [0, 2, 3])
def test_trim_returns_invalid_counts_by_request(valid):
    output = SchedulerOutput.make_empty()
    output.scheduled_cached_reqs.req_ids = ["trimmed", "full", "missing", "no_draft"]
    output.scheduled_spec_decode_tokens = {
        "trimmed": [10, 11, 12],
        "full": [20, 21, 22],
        "missing": [30, 31, 32],
    }
    output.num_scheduled_tokens = {"trimmed": 4, "full": 4, "missing": 4, "no_draft": 1}
    output.total_num_scheduled_tokens = 13
    event = torch.cuda.Event()
    event.record()

    invalid = update_scheduler_for_invalid_drafts(
        event,
        torch.tensor([3, valid, 0]),
        output,
        {"full": 0, "trimmed": 1, "no_draft": 2},
    )

    assert invalid == ({"trimmed": 3 - valid} if valid < 3 else {})
    expected = {"full": [20, 21, 22], "missing": [30, 31, 32]}
    if valid:
        expected["trimmed"] = [10, 11, 12][:valid]
    assert output.scheduled_spec_decode_tokens == expected
    assert output.num_scheduled_tokens == {
        "trimmed": 1 + valid,
        "full": 4,
        "missing": 4,
        "no_draft": 1,
    }
    assert output.total_num_scheduled_tokens == 10 + valid
