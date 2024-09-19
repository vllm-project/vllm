from dataclasses import dataclass
from typing import List, Iterable
import torch
from vllm.wde.core.schema.engine_io import (Request, PromptInput,
                                            TextOnlyInputs, SchedulableRequest,
                                            RequestOutput, SchedulerOutput)


@dataclass
class EncodeOnlyInput(TextOnlyInputs):
    pass


@dataclass
class EncodeOnlyRequest(Request):
    inputs: PromptInput


@dataclass
class EncodeOnlySchedulableRequest(SchedulableRequest):
    inputs: TextOnlyInputs

    @property
    def num_new_tokens(self):
        return len(self.inputs.prompt_token_ids)


@dataclass
class EncodeOnlySchedulerOutput(SchedulerOutput):
    requests: Iterable[EncodeOnlyRequest]

    def is_empty(self) -> bool:
        return not self.requests


class EncodeOnlyRequestOutput(RequestOutput):

    def __init__(self, request_id: str, outputs: torch.Tensor,
                 prompt_token_ids: List[int], finished: bool):
        self.request_id = request_id
        self.prompt_token_ids = prompt_token_ids
        self.finished = finished
        self.outputs = outputs

    def __repr__(self):
        return (f"EncodeOnlyRequestOutput(request_id='{self.request_id}', "
                f"outputs={repr(self.outputs)}, "
                f"prompt_token_ids={self.prompt_token_ids}, "
                f"finished={self.finished})")
