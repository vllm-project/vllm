from dataclasses import dataclass
from typing import TypeAlias

@dataclass
class UbatchSlice:
    request_slice: slice
    token_slice: slice
UBatchSlices: TypeAlias = list[UbatchSlice]