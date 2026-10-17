# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import numpy as np
import pytest

from vllm.lora.request import LoRARequest
from vllm.v1.worker.gpu.input_batch import make_num_logits_per_req
from vllm.v1.worker.gpu.lora_utils import LoraState

pytestmark = [pytest.mark.cpu_test, pytest.mark.skip_global_cleanup]


def test_prompt_lora_mapping_aligns_with_spec_decode_logits_rows():
    """lm_head LoRA ids follow logits rows, not requests.

    Under speculative decoding cu_num_logits is one bonus row plus each
    draft row. A per-request mapping writes the adapter onto a neighbor.
    """
    state = LoraState(max_num_reqs=4)
    lora_b = LoRARequest("adapter-b", 5, "/tmp/adapter-b")
    # State slots are not batch order: slot 1 is req-a, slot 0 is req-b.
    state.add_request("req-a", 1, None)
    state.add_request("req-b", 0, lora_b)

    idx_mapping = np.array([1, 0], dtype=np.int32)
    # req-a: 1 bonus + 2 drafts. req-b: 1 bonus and no drafts.
    cu_num_logits = np.array([0, 3, 4], dtype=np.int32)
    num_logits = np.diff(cu_num_logits).astype(np.int32)
    # Query length is independent: req-a is still prefilling.
    num_scheduled = np.array([8, 1], dtype=np.int32)

    prompt, token, requests = state.make_lora_inputs(
        ["req-a", "req-b"],
        idx_mapping,
        num_scheduled,
        num_logits,
    )

    assert num_logits.tolist() == [3, 1]
    assert prompt == (0, 0, 0, 5)
    assert len(prompt) == int(cu_num_logits[-1])
    assert token == (0,) * 8 + (5,)
    assert requests == {lora_b}


def test_num_logits_per_req_counts_bonus_rows_and_draft_rows():
    """Bonus tokens and draft tokens both add logits rows."""
    drafts = np.array([0, 2, 1], dtype=np.int32)
    assert make_num_logits_per_req(3, drafts, 1).tolist() == [1, 3, 2]
    # Multiple sampled tokens per step, no drafts: still more than one row.
    assert make_num_logits_per_req(2, None, 2).tolist() == [2, 2]
    assert make_num_logits_per_req(3, None, 1).tolist() == [1, 1, 1]

    state = LoraState(max_num_reqs=2)
    lora = LoRARequest("adapter", 3, "/tmp/adapter")
    state.add_request("req-a", 0, lora)
    state.add_request("req-b", 1, None)
    num_logits = make_num_logits_per_req(2, None, 2)
    prompt, token, _requests = state.make_lora_inputs(
        ["req-a", "req-b"],
        np.array([0, 1], dtype=np.int32),
        np.array([2, 2], dtype=np.int32),
        num_logits,
    )
    assert prompt == (3, 3, 0, 0)
    assert token == (3, 3, 0, 0)
