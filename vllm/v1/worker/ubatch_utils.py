# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from dataclasses import dataclass

from typing_extensions import TypeAlias


@dataclass
class UBatchSlice:
    request_slice: slice
    token_slice: slice


UBatchSlices: TypeAlias = list[UBatchSlice]


def is_second_ubatch_empty(orig_num_tokens: int,
                           padded_num_tokens: int) -> bool:
    return (padded_num_tokens // 2) >= orig_num_tokens
