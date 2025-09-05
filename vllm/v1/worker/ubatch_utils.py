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


def create_slices(query_start_loc: np.ndarray, split_point: int,
                  num_tokens: int) -> tuple[slice, slice]:
    first_token_slice = slice(0, split_point)
    second_token_slice = slice(split_point, num_tokens)

    # Determine request slices using exclusive stop semantics
    # First ubatch includes requests whose tokens overlap [0, split_point)
    first_ubatch_req_stop = int(
        np.searchsorted(query_start_loc, split_point, side="left"))
    first_ubatch_req_slice = slice(0, first_ubatch_req_stop)

    # Second ubatch starts at the request that contains the split_point
    # or the request starting exactly at split_point (if on boundary)
    second_ubatch_req_start = int(
        np.searchsorted(query_start_loc, split_point, side="left") - 1)
    second_ubatch_req_slice = slice(second_ubatch_req_start,
                                    len(query_start_loc) - 1)

    return [
        UbatchSlice(first_ubatch_req_slice, first_token_slice),
        UbatchSlice(second_ubatch_req_slice, second_token_slice)
    ]
