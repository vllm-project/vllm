from dataclasses import dataclass
import torch
from vllm.wde.core.schema.execute_io import ModelInput, ExecuteInput
from vllm.wde.encode_only.layers.attention import EncodeOnlyAttentionMetadata


@dataclass
class ModelInputForGPU(ModelInput):
    input_ids: torch.Tensor
    positions: torch.Tensor
    attn_metadata: EncodeOnlyAttentionMetadata

    def to(self, target_device, non_blocking=False):
        for k in self.__dict__.keys():
            self.__dict__[k] = self.__dict__[k].to(device=target_device,
                                                   non_blocking=non_blocking)

    def to_dict(self):
        return self.__dict__


class EncodeOnlyExecuteInput(ExecuteInput):
    worker_input = None
    model_input: ModelInputForGPU
