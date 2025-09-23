# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from dataclasses import dataclass

from typing_extensions import TypeAlias


@dataclass
class UBatchSlice:
    request_slice: slice
    token_slice: slice

    def is_empty(self) -> bool:
        return self.request_slice.start == self.request_slice.stop \
            or self.token_slice.start == self.token_slice.stop

    @property
    def num_tokens(self) -> int:
        return self.token_slice.stop - self.token_slice.start


UBatchSlices: TypeAlias = list[UBatchSlice]


def is_second_ubatch_empty(orig_num_tokens_per_ubatch: int,
                           padded_num_tokens_per_ubatch: int) -> bool:
    return padded_num_tokens_per_ubatch >= 2 * orig_num_tokens_per_ubatch
