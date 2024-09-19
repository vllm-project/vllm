from typing import Optional
from dataclasses import dataclass


@dataclass
class ModelInput:
    pass


@dataclass
class WorkerInput:
    pass


@dataclass
class ExecuteInput:
    worker_input: Optional[WorkerInput]
    model_input: Optional[ModelInput]


class ExecuteOutput:
    pass
