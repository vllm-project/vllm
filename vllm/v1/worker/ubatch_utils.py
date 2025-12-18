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


def is_last_ubatch_empty(
    orig_num_tokens: int, padded_num_tokens: int, num_ubatches: int
) -> bool:
    return (padded_num_tokens // num_ubatches) * (num_ubatches - 1) >= orig_num_tokens


def check_ubatch_thresholds(
    config: ParallelConfig, num_tokens: int, uniform_decode: bool
) -> bool:
    if not config.use_ubatching:
        return False
    if uniform_decode:
        return num_tokens >= config.dbo_decode_token_threshold
    else:
        return num_tokens >= config.dbo_prefill_token_threshold


# This pads the last ubatch slice out to the total number of tokens
# (num_tokens + padding) since we do `create_ubatch_slices` before applying DP padding.
def _pad_out_ubatch_slices(
    ubatch_slices: UBatchSlices, num_total_tokens: int, num_reqs_padded: int
) -> UBatchSlices:
    last_slice = ubatch_slices[-1]
    padded_last_request_slice = slice(last_slice.request_slice.start, num_reqs_padded)
    padded_last_token_slice = slice(last_slice.token_slice.start, num_total_tokens)

    return ubatch_slices[:-1] + [
        UBatchSlice(padded_last_request_slice, padded_last_token_slice)
    ]


def maybe_create_ubatch_slices(
    should_ubatch: bool,
    num_scheduled_tokens: np.ndarray,
    num_tokens_padded: int,
    num_reqs_padded: int,
    num_ubatches: int,
    split_point: list[int] | int | None = None,
) -> tuple[UBatchSlices | None, UBatchSlices | None]:
    if not should_ubatch:
        return None, None

    if split_point is None:
        split_point = int(num_tokens_padded) // num_ubatches

    token_split_points = [split_point * i for i in range(1, num_ubatches)]

    # TODO(lucas): Refactor the gpu_model_runner.py so we can pass
    # in cu_num_tokens directly (i.e. query_start_loc)
    cu_num_tokens = np.zeros(len(num_scheduled_tokens) + 1, dtype=np.int32)
    np.cumsum(num_scheduled_tokens, dtype=np.int32, out=cu_num_tokens[1:])

    ubatch_slices = []
    start_token = 0

    # Add the end point to the split points to make iteration easier
    all_points = token_split_points + [cu_num_tokens[-1]]

    for end_token in all_points:
        token_slice = slice(start_token, end_token)

        # Determine request slices using exclusive stop semantics
        # Ubatch includes requests whose tokens overlap [start_token, end_token)

        # Start at the request that contains the start_token
        # or the request starting exactly at start_token (if on boundary)
        req_start = int(np.searchsorted(cu_num_tokens, start_token, side="right") - 1)

        # Stop at the request that starts at or after end_token
        req_stop = int(np.searchsorted(cu_num_tokens, end_token, side="left"))

        req_slice = slice(req_start, req_stop)
        ubatch_slices.append(UBatchSlice(req_slice, token_slice))

        start_token = end_token

    ubatch_slices_padded = _pad_out_ubatch_slices(
        ubatch_slices, num_tokens_padded, num_reqs_padded
    )

    assert sum(s.num_tokens for s in ubatch_slices_padded) == num_tokens_padded

    return ubatch_slices, ubatch_slices_padded
