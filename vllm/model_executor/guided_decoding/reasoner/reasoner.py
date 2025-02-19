# SPDX-License-Identifier: Apache-2.0
from abc import ABC, abstractmethod
from dataclasses import dataclass


class Reasoner(ABC):

    @abstractmethod
    def get_start_token_id(self) -> int:
        pass

    @abstractmethod
    def get_end_token_id(self) -> int:
        pass


@dataclass
class ReasonerConfig:
    start_token_id: int
    end_token_id: int

    @classmethod
    def from_reasoner(cls, reasoner: Reasoner) -> 'ReasonerConfig':
        return cls(start_token_id=reasoner.get_start_token_id(),
                   end_token_id=reasoner.get_end_token_id())

    def is_reasoning_end(self, input_ids: list[int]) -> bool:
        return self.end_token_id in input_ids
