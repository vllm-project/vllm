from dataclasses import dataclass

import torch

from vllm.wde.core.layers.attention import AttentionMetadata
from vllm.wde.core.schema.execute_io import ExecuteInput, ModelInput


@dataclass
class ModelInputForGPU(ModelInput):
    input_ids: torch.Tensor
    positions: torch.Tensor
    attn_metadata: AttentionMetadata

    def to(self, target_device, non_blocking=False):
        for k in self.__dict__:
            self.__dict__[k] = self.__dict__[k].to(device=target_device,
                                                   non_blocking=non_blocking)

    def to_dict(self):
        return self.__dict__


class PrefillOnlyExecuteInput(ExecuteInput):
    worker_input = None
    model_input: ModelInputForGPU
