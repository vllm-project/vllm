# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from dataclasses import dataclass
from typing import TypeAlias

import numpy as np


@dataclass
class UbatchSlice:
    request_slice: slice
    token_slice: slice

    def is_empty(self) -> bool:
        return self.request_slice.start == self.request_slice.stop \
            or self.token_slice.start == self.token_slice.stop


UBatchSlices: TypeAlias = list[UbatchSlice]


def create_ubatch_slices(num_scheduled_tokens: np.ndarray, split_point: int) -> tuple[UbatchSlice, UbatchSlice]:
    # TODO(lucas): Refactor the gpu_model_runner.py so we can pass
    # in cu_num_tokens directly (i.e. query_start_loc)
    cu_num_tokens = np.zeros(len(num_scheduled_tokens) + 1, dtype=np.int32)
    np.cumsum(num_scheduled_tokens, dtype=np.int32, out=cu_num_tokens[1:])

    first_ubatch_token_slice = slice(0, split_point)
    second_ubatch_token_slice = slice(split_point, cu_num_tokens[-1])

    # Determine request slices using exclusive stop semantics
    # First ubatch includes requests whose tokens overlap [0, split_point)
    first_ubatch_req_stop = int(
        np.searchsorted(cu_num_tokens, split_point, side="left"))
    first_ubatch_req_slice = slice(0, first_ubatch_req_stop)

    # Second ubatch starts at the request that contains the split_point
    # or the request starting exactly at split_point (if on boundary)
    second_ubatch_req_start = int(
        np.searchsorted(cu_num_tokens, split_point, side="right") - 1)
    second_ubatch_req_slice = slice(second_ubatch_req_start,
                                    len(cu_num_tokens) - 1)

    return [
        UbatchSlice(first_ubatch_req_slice, first_ubatch_token_slice),
        UbatchSlice(second_ubatch_req_slice, second_ubatch_token_slice)
    ]
