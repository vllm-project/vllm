from typing import List

from cacheflow.utils import Device

BLANK_TOKEN_ID = -1


class LogicalTokenBlock:

    def __init__(
        self,
        block_number: int,
        block_size: int,
    ) -> None:
        self.block_number = block_number
        self.block_size = block_size

        self.token_ids = [BLANK_TOKEN_ID] * block_size
        self.num_tokens = 0

    def is_empty(self) -> bool:
        return self.num_tokens == 0

    def get_num_empty_slots(self) -> int:
        return self.block_size - self.num_tokens

    def is_full(self) -> bool:
        return self.num_tokens == self.block_size

    def append(self, token_ids: List[int]) -> None:
        assert len(token_ids) <= self.get_num_empty_slots()
        self.token_ids[self.num_tokens:self.num_tokens + len(token_ids)] = token_ids
        self.num_tokens += len(token_ids)

    def get_token_ids(self) -> List[int]:
        return self.token_ids[:self.num_tokens]


class PhysicalTokenBlock:

    def __init__(
        self,
        device: Device,
        block_number: int,
        block_size: int,
    ) -> None:
        self.device = device
        self.block_number = block_number
        self.block_size = block_size

        self.ref_count = 0
