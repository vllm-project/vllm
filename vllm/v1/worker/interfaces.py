# SPDX-License-Identifier: Apache-2.0
from abc import ABC, abstractmethod

import numpy as np

from vllm.lora.request import LoRARequest


class BaseInputBatch(ABC):

    @property
    @abstractmethod
    def req_ids(self) -> list[str]:
        raise NotImplementedError

    @abstractmethod
    def swap_states(self, i1: int, i2: int) -> None:
        raise NotImplementedError

    @abstractmethod
    def make_lora_inputs(
        self, num_scheduled_tokens: np.ndarray
    ) -> tuple[tuple[int, ...], tuple[int, ...], set[LoRARequest]]:
        raise NotImplementedError

    @property
    @abstractmethod
    def num_reqs(self) -> int:
        raise NotImplementedError
