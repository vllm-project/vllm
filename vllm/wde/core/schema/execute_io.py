from dataclasses import dataclass
from typing import Optional


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
