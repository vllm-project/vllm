from dataclasses import dataclass
from typing import TypeAlias

@dataclass
class UbatchSlice:
    request_slice: slice
    token_slice: slice
UBatchSlices: TypeAlias = list[UbatchSlice]

def is_second_ubatch_empty(orig_num_tokens_per_ubatch: int, padded_num_tokens_per_ubatch: int) -> bool:
    return padded_num_tokens_per_ubatch >= 2 * orig_num_tokens_per_ubatch