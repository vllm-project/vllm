
from abc import ABC
from typing import Optional
from dataclasses import dataclass


@dataclass
class ModelInput(ABC):
    pass


@dataclass
class WorkerInput:
    pass


@dataclass
class ExecuteInput(ABC):
    worker_input: Optional[WorkerInput]
    model_input: Optional[ModelInput]


class ExecuteOutput(ABC):
    pass
