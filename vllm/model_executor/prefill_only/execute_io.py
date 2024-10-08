from dataclasses import dataclass
from typing import Optional

import torch

from vllm.attention.prefill_only.abstract import PrefillOnlyAttentionBackend


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

    def to(self, target_device, non_blocking=False):
        for k in self.__dict__:
            self.__dict__[k] = self.__dict__[k].to(device=target_device,
                                                   non_blocking=non_blocking)


@dataclass
class ModelInputForGPU(ModelInput):
    input_ids: torch.Tensor
    positions: torch.Tensor
    attn_metadata: PrefillOnlyAttentionBackend

    def to(self, target_device, non_blocking=False):
        for k in self.__dict__:
            self.__dict__[k] = self.__dict__[k].to(device=target_device,
                                                   non_blocking=non_blocking)

    def to_dict(self):
        return self.__dict__


class PrefillOnlyExecuteInput(ExecuteInput):
    worker_input = None
    model_input: ModelInputForGPU
