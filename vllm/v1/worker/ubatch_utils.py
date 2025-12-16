# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from dataclasses import dataclass
from typing import TypeAlias

import numpy as np

from vllm.config import ParallelConfig


@dataclass
class UBatchSlice:
    request_slice: slice
    token_slice: slice

    def is_empty(self) -> bool:
        return (
            self.request_slice.start == self.request_slice.stop
            or self.token_slice.start == self.token_slice.stop
        )

    @property
    def num_tokens(self) -> int:
        return self.token_slice.stop - self.token_slice.start


UBatchSlices: TypeAlias = list[UBatchSlice]


def is_second_ubatch_empty(orig_num_tokens: int, padded_num_tokens: int) -> bool:
    return (padded_num_tokens // 2) >= orig_num_tokens


def check_ubatch_thresholds(
    config: ParallelConfig, num_tokens: int, uniform_decode: bool
) -> bool:
    if not config.enable_dbo:
        return False
    if uniform_decode:
        return num_tokens >= config.dbo_decode_token_threshold
    else:
        return num_tokens >= config.dbo_prefill_token_threshold


# This just pads the second ubatch slice out to the total number of tokens
# (num_tokens + padding) since we do `create_ubatch_slices` before applying DP padding.
def _pad_out_ubatch_slices(
    ubatch_slices: UBatchSlices, num_total_tokens: int, num_reqs_padded: int
) -> UBatchSlices:
    # TODO(lucas): handle empty second ubatch
    padded_second_request_slice = slice(
        ubatch_slices[1].request_slice.start, num_reqs_padded
    )
    padded_second_token_slice = slice(
        ubatch_slices[1].token_slice.start, num_total_tokens
    )
    return [
        ubatch_slices[0],
        UBatchSlice(padded_second_request_slice, padded_second_token_slice),
    ]


def maybe_create_ubatch_slices(
    should_ubatch: bool,
    num_scheduled_tokens: np.ndarray,
    num_tokens_padded: int,
    num_reqs_padded: int,
    split_point: int | None = None,
) -> tuple[UBatchSlices | None, UBatchSlices | None]:
    if not should_ubatch:
        return None, None

    if split_point is None:
        split_point = int(num_tokens_padded) // 2

    # TODO(lucas): Refactor the gpu_model_runner.py so we can pass
    # in cu_num_tokens directly (i.e. query_start_loc)
    cu_num_tokens = np.zeros(len(num_scheduled_tokens) + 1, dtype=np.int32)
    np.cumsum(num_scheduled_tokens, dtype=np.int32, out=cu_num_tokens[1:])

    first_ubatch_token_slice = slice(0, split_point)
    second_ubatch_token_slice = slice(split_point, cu_num_tokens[-1])

    # Determine request slices using exclusive stop semantics
    # First ubatch includes requests whose tokens overlap [0, split_point)
    first_ubatch_req_stop = int(
        np.searchsorted(cu_num_tokens, split_point, side="left")
    )
    first_ubatch_req_slice = slice(0, first_ubatch_req_stop)

    # Second ubatch starts at the request that contains the split_point
    # or the request starting exactly at split_point (if on boundary)
    second_ubatch_req_start = int(
        np.searchsorted(cu_num_tokens, split_point, side="right") - 1
    )
    second_ubatch_req_slice = slice(second_ubatch_req_start, len(cu_num_tokens) - 1)

    ubatch_slices = [
        UBatchSlice(first_ubatch_req_slice, first_ubatch_token_slice),
        UBatchSlice(second_ubatch_req_slice, second_ubatch_token_slice),
    ]

    ubatch_slices_padded = _pad_out_ubatch_slices(
        ubatch_slices, num_tokens_padded, num_reqs_padded
    )

    assert sum(s.num_tokens for s in ubatch_slices_padded) == num_tokens_padded

    return ubatch_slices, ubatch_slices_padded
