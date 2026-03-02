# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass

import numpy as np
import pytest
import torch

from vllm.v1.attention.backend import CommonAttentionMetadata
from vllm.v1.attention.backends.utils import (
    reorder_batch_to_split_decodes_and_prefills,
    split_decodes_and_prefills,
)


def _make_common_attn_metadata(
    query_lens: list[int],
    seq_lens: list[int],
    num_computed_tokens: list[int] | None = None,
):
    num_reqs = len(query_lens)
    num_tokens = sum(query_lens)
    max_query_len = max(query_lens) if query_lens else 0

    query_start_loc = torch.zeros(num_reqs + 1, dtype=torch.int32)
    for i, ql in enumerate(query_lens):
        query_start_loc[i + 1] = query_start_loc[i] + ql

    seq_lens_t = torch.tensor(seq_lens, dtype=torch.int32)

    nct = None
    if num_computed_tokens is not None:
        nct = torch.tensor(num_computed_tokens, dtype=torch.int32)

    return CommonAttentionMetadata(
        query_start_loc=query_start_loc,
        query_start_loc_cpu=query_start_loc,
        seq_lens=seq_lens_t,
        _num_computed_tokens_cpu=nct,
        num_reqs=num_reqs,
        num_actual_tokens=num_tokens,
        max_query_len=max_query_len,
        max_seq_len=max(seq_lens) if seq_lens else 0,
        block_table_tensor=torch.empty(0),
        slot_mapping=torch.empty(0),
    )


class MockInputBatch:
    def __init__(self, req_ids, num_computed_tokens_cpu):
        self.req_ids = req_ids
        self.num_computed_tokens_cpu = num_computed_tokens_cpu

    def swap_states(self, i, j):
        self.req_ids[i], self.req_ids[j] = self.req_ids[j], self.req_ids[i]
        self.num_computed_tokens_cpu[i], self.num_computed_tokens_cpu[j] = (
            self.num_computed_tokens_cpu[j],
            self.num_computed_tokens_cpu[i],
        )


class MockSchedulerOutput:
    def __init__(self, num_scheduled_tokens):
        self.num_scheduled_tokens = num_scheduled_tokens


@dataclass
class ReorderTestCase:
    requests: list[tuple[int, int]]  # (num_scheduled_tokens, num_computed_tokens)
    expected_order: list[int]
    expected_modified: bool
    decode_threshold: int = 1


# Test cases for batch reordering
REORDER_TEST_CASES = {
    "all_decodes": ReorderTestCase(
        requests=[(1, 10), (1, 20), (1, 30)],
        expected_order=[0, 1, 2],
        expected_modified=False,
    ),
    "all_prefills": ReorderTestCase(
        requests=[(100, 100), (200, 200), (300, 300)],
        expected_order=[0, 1, 2],
        expected_modified=False,
    ),
    "mixed_interleaved": ReorderTestCase(
        requests=[(100, 100), (1, 10), (200, 200), (1, 20)],
        expected_order=[3, 1, 2, 0],  # Only swap 0↔3, keep 1 and 2 in place
        expected_modified=True,
    ),
    "already_ordered": ReorderTestCase(
        requests=[(1, 10), (1, 20), (100, 100), (200, 0)],
        expected_order=[0, 1, 2, 3],
        expected_modified=False,
    ),
    "single_request": ReorderTestCase(
        requests=[(1, 10)],
        expected_order=[0],
        expected_modified=False,
    ),
    "higher_threshold": ReorderTestCase(
        requests=[(2, 10), (3, 20), (5, 30), (6, 40)],
        expected_order=[0, 1, 2, 3],
        expected_modified=False,
        decode_threshold=4,
    ),
    "decodes_at_end": ReorderTestCase(
        requests=[(100, 100), (200, 200), (1, 10), (1, 20)],
        expected_order=[2, 3, 0, 1],
        expected_modified=True,
    ),
    "decode_extend_prefill": ReorderTestCase(
        requests=[(100, 0), (10, 50), (1, 10)],
        expected_order=[2, 1, 0],
        expected_modified=True,
    ),
    "extend_prefill_only": ReorderTestCase(
        requests=[(100, 0), (10, 50), (200, 0), (20, 75)],
        expected_order=[3, 1, 2, 0],  # Only swap 0↔3, keep 1 and 2 in place
        expected_modified=True,
    ),
    "complicated_mixed_interleaved": ReorderTestCase(
        requests=[
            (1, 20),
            (1, 50),
            (374, 0),
            (300, 20),
            (1, 20),
            (256, 0),
            (1, 5),
            (27, 0),
            (1, 4),
        ],
        expected_order=[0, 1, 6, 8, 4, 3, 2, 7, 5],
        expected_modified=True,
    ),
    "new_request_single_token_prefill": ReorderTestCase(
        requests=[
            (100, 0),
            (1, 0),  # New request with only 1 token (STILL prefill)
            (50, 100),
            (1, 10),
        ],
        # Only index 3 is a true decode (has num_computed_tokens > 0)
        expected_order=[3, 2, 0, 1],
        expected_modified=True,
    ),
    "multiple_new_requests_single_token_prefill": ReorderTestCase(
        requests=[
            (1, 0),  # New prefill (1 token, no computed)
            (1, 0),  # New prefill (1 token, no computed)
            (1, 50),
            (200, 0),
        ],
        expected_order=[2, 1, 0, 3],
        expected_modified=True,
    ),
}


@pytest.mark.parametrize(
    "test_case", REORDER_TEST_CASES.values(), ids=REORDER_TEST_CASES.keys()
)
def test_reorder_batch_to_split_decodes_and_prefills(test_case: ReorderTestCase):
    req_ids = [f"r{i}" for i in range(len(test_case.requests))]
    num_computed_tokens = np.array([r[1] for r in test_case.requests], dtype=np.int32)
    num_scheduled_tokens = {f"r{i}": r[0] for i, r in enumerate(test_case.requests)}

    input_batch = MockInputBatch(req_ids, num_computed_tokens)
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


@dataclass
class SplitTestCase:
    query_lens: list[int]
    seq_lens: list[int]
    num_computed_tokens: list[int]
    decode_threshold: int
    expected: tuple[int, int, int, int]  # (num_d, num_p, num_dt, num_pt)


SPLIT_TEST_CASES = {
    "mtp_new_request_is_prefill": SplitTestCase(
        query_lens=[3],
        seq_lens=[3],
        num_computed_tokens=[0],
        decode_threshold=4,
        expected=(0, 1, 0, 3),
    ),
    "mtp_cuda_graph_synthetic_decodes": SplitTestCase(
        query_lens=[4, 4, 4],
        seq_lens=[4, 4, 4],
        num_computed_tokens=[1, 1, 1],
        decode_threshold=4,
        expected=(3, 0, 12, 0),
    ),
    "mtp_mixed_decodes_and_new_request": SplitTestCase(
        query_lens=[4, 4, 3],
        seq_lens=[100, 200, 3],
        num_computed_tokens=[96, 196, 0],
        decode_threshold=4,
        expected=(2, 1, 8, 3),
    ),
}


@pytest.mark.parametrize(
    "test_case", SPLIT_TEST_CASES.values(), ids=SPLIT_TEST_CASES.keys()
)
def test_split_decodes_and_prefills(test_case: SplitTestCase):
    meta = _make_common_attn_metadata(
        query_lens=test_case.query_lens,
        seq_lens=test_case.seq_lens,
        num_computed_tokens=test_case.num_computed_tokens,
    )
    result = split_decodes_and_prefills(
        meta, decode_threshold=test_case.decode_threshold
    )
    assert result == test_case.expected
