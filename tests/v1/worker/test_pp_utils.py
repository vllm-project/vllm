# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import numpy as np
import pytest

from vllm.v1.worker.gpu.pp_utils import compute_need_sampled_mask

pytestmark = pytest.mark.cpu_test


def _input_batch(*, old_computed: int, current_drafts: int, previous_drafts: int):
    return SimpleNamespace(
        num_computed_tokens_np=np.array([old_computed], dtype=np.int32),
        prefill_len_np=np.array([10], dtype=np.int32),
        max_seq_len_np=np.array([101], dtype=np.int32),
        num_scheduled_tokens=np.array([current_drafts + 1], dtype=np.int32),
        num_draft_tokens_per_req=np.array([current_drafts], dtype=np.int32),
        prev_num_draft_tokens_per_req=np.array([previous_drafts], dtype=np.int32),
    )


def test_need_sampled_mask_discounts_previous_drafts():
    batch = _input_batch(old_computed=102, current_drafts=1, previous_drafts=3)
    np.testing.assert_array_equal(compute_need_sampled_mask(batch), [True])


def test_need_sampled_mask_does_not_discount_current_drafts():
    batch = _input_batch(old_computed=100, current_drafts=3, previous_drafts=0)
    assert compute_need_sampled_mask(batch) is None
