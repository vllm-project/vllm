# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass

import numpy as np
import pytest

from vllm.v1.attention.backends.utils import reorder_batch_to_split_decodes_and_prefills


class MockInputBatch:
    def __init__(self, req_ids, num_computed_tokens_cpu, num_prompt_tokens):
        self.req_ids = req_ids
        self.num_computed_tokens_cpu = num_computed_tokens_cpu
        self.num_prompt_tokens = num_prompt_tokens

    def swap_states(self, i, j):
        self.req_ids[i], self.req_ids[j] = self.req_ids[j], self.req_ids[i]
        self.num_computed_tokens_cpu[i], self.num_computed_tokens_cpu[j] = (
            self.num_computed_tokens_cpu[j],
            self.num_computed_tokens_cpu[i],
        )
        self.num_prompt_tokens[i], self.num_prompt_tokens[j] = (
            self.num_prompt_tokens[j],
            self.num_prompt_tokens[i],
        )


class MockSchedulerOutput:
    def __init__(self, num_scheduled_tokens):
        self.num_scheduled_tokens = num_scheduled_tokens


@dataclass
class ReorderTestCase:
    # (num_scheduled_tokens, num_computed_tokens, num_prompt_tokens)
    requests: list[tuple[int, int, int]]
    expected_order: list[int]
    expected_modified: bool
    decode_threshold: int = 1


# Test cases for batch reordering
# Format: (num_scheduled, num_computed, num_prompt)
REORDER_TEST_CASES = {
    "all_decodes": ReorderTestCase(
        requests=[(1, 10, 10), (1, 20, 20), (1, 30, 30)],
        expected_order=[0, 1, 2],
        expected_modified=False,
    ),
    "all_long_extends": ReorderTestCase(
        requests=[(100, 100, 100), (200, 200, 200), (300, 300, 300)],
        expected_order=[0, 1, 2],
        expected_modified=False,
    ),
    "mixed_decodes_long_extends": ReorderTestCase(
        requests=[(100, 100, 100), (1, 10, 10), (200, 200, 200), (1, 20, 20)],
        expected_order=[3, 1, 2, 0],
        expected_modified=True,
    ),
    "already_ordered": ReorderTestCase(
        requests=[(1, 10, 10), (1, 20, 20), (100, 100, 100), (200, 0, 200)],
        expected_order=[0, 1, 2, 3],
        expected_modified=False,
    ),
    "single_request": ReorderTestCase(
        requests=[(1, 10, 10)],
        expected_order=[0],
        expected_modified=False,
    ),
    "higher_threshold": ReorderTestCase(
        requests=[(2, 10, 10), (3, 20, 20), (5, 30, 30), (6, 40, 40)],
        expected_order=[0, 1, 2, 3],
        expected_modified=False,
        decode_threshold=4,
    ),
    "decodes_at_end": ReorderTestCase(
        requests=[(100, 100, 100), (200, 200, 200), (1, 10, 10), (1, 20, 20)],
        expected_order=[2, 3, 0, 1],
        expected_modified=True,
    ),
    "decode_long_extend_prefill": ReorderTestCase(
        requests=[(100, 0, 100), (10, 50, 50), (1, 10, 10)],
        expected_order=[2, 1, 0],
        expected_modified=True,
    ),
    "long_extend_prefill_only": ReorderTestCase(
        requests=[(100, 0, 100), (10, 50, 50), (200, 0, 200), (20, 75, 75)],
        expected_order=[3, 1, 2, 0],
        expected_modified=True,
    ),
    "complicated_mixed": ReorderTestCase(
        requests=[
            (1, 20, 20),  # decode
            (1, 50, 50),  # decode
            (374, 0, 374),  # prefill
            (300, 20, 20),  # long_extend
            (1, 20, 20),  # decode
            (256, 0, 256),  # prefill
            (1, 5, 5),  # decode
            (27, 0, 27),  # prefill
            (1, 4, 4),  # decode
        ],
        expected_order=[0, 1, 6, 8, 4, 3, 2, 7, 5],
        expected_modified=True,
    ),
    "new_request_single_token_prefill": ReorderTestCase(
        requests=[
            (100, 0, 100),  # prefill
            (1, 0, 1),  # prefill (single token, still prefill)
            (50, 100, 100),  # long_extend
            (1, 10, 10),  # decode
        ],
        expected_order=[3, 2, 0, 1],
        expected_modified=True,
    ),
    "multiple_new_requests_single_token_prefill": ReorderTestCase(
        requests=[
            (1, 0, 1),  # prefill
            (1, 0, 1),  # prefill
            (1, 50, 50),  # decode
            (200, 0, 200),  # prefill
        ],
        expected_order=[2, 1, 0, 3],
        expected_modified=True,
    ),
    "four_way_already_ordered": ReorderTestCase(
        requests=[
            (1, 100, 100),  # decode
            (1, 50, 100),  # short_extend
            (10, 50, 100),  # long_extend
            (100, 0, 100),  # prefill
        ],
        expected_order=[0, 1, 2, 3],
        expected_modified=False,
    ),
    "four_way_needs_reorder": ReorderTestCase(
        requests=[
            (100, 0, 100),  # prefill
            (1, 50, 100),  # short_extend
            (1, 100, 100),  # decode
            (10, 50, 100),  # long_extend
        ],
        expected_order=[2, 1, 3, 0],
        expected_modified=True,
    ),
    "four_way_multiple_short_extends": ReorderTestCase(
        requests=[
            (2, 100, 100),  # decode
            (2, 50, 200),  # short_extend
            (2, 75, 150),  # short_extend
            (2, 200, 200),  # decode
        ],
        expected_order=[0, 3, 2, 1],
        expected_modified=True,
        decode_threshold=2,
    ),
    "four_way_spec_decode_threshold": ReorderTestCase(
        requests=[
            (5, 100, 100),  # decode
            (5, 50, 100),  # short_extend
            (5, 0, 100),  # prefill
            (10, 50, 100),  # long_extend
        ],
        expected_order=[0, 1, 3, 2],
        expected_modified=True,
        decode_threshold=5,
    ),
}


@pytest.mark.parametrize(
    "test_case", REORDER_TEST_CASES.values(), ids=REORDER_TEST_CASES.keys()
)
def test_reorder_batch_to_split_decodes_and_prefills(test_case: ReorderTestCase):
    req_ids = [f"r{i}" for i in range(len(test_case.requests))]
    num_computed_tokens = np.array([r[1] for r in test_case.requests], dtype=np.int32)
    num_scheduled_tokens = {f"r{i}": r[0] for i, r in enumerate(test_case.requests)}
    num_prompt_tokens = np.array([r[2] for r in test_case.requests], dtype=np.int32)

    input_batch = MockInputBatch(req_ids, num_computed_tokens, num_prompt_tokens)
    scheduler_output = MockSchedulerOutput(num_scheduled_tokens)

    modified = reorder_batch_to_split_decodes_and_prefills(
        input_batch, scheduler_output, decode_threshold=test_case.decode_threshold
    )

    expected_req_ids = [f"r{i}" for i in test_case.expected_order]

    assert modified == test_case.expected_modified, (
        f"Expected modified={test_case.expected_modified}, got {modified}"
    )
    assert input_batch.req_ids == expected_req_ids, (
        f"Expected order {expected_req_ids}, got {input_batch.req_ids}"
    )
