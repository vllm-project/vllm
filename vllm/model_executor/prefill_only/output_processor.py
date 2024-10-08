from abc import ABC, abstractmethod
from typing import List

import torch

from vllm.model_executor.prefill_only.engine_io import (RequestOutput,
                                                        SchedulerOutput)


class OutputProcessor(ABC):
    """
    scheduler_output, execute_output -> OutputProcessor -> RequestOutput
    """

    @abstractmethod
    def __call__(self, scheduler_output: SchedulerOutput,
                 execute_output: torch.Tensor) -> List[RequestOutput]:
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def from_engine(cls, engine):
        raise NotImplementedError
