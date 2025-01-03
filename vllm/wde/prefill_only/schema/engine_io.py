from dataclasses import dataclass
from typing import Iterable, List

import torch

from vllm.wde.core.schema.engine_io import (PromptInput, Request,
                                            RequestOutput, SchedulableRequest,
                                            SchedulerOutput, TextOnlyInputs)


@dataclass
class PrefillOnlyInput(TextOnlyInputs):
    pass


@dataclass
class PrefillOnlyRequest(Request):
    inputs: PromptInput


@dataclass
class PrefillOnlySchedulableRequest(SchedulableRequest):
    inputs: TextOnlyInputs

    @property
    def num_new_tokens(self):
        return len(self.inputs.prompt_token_ids)


@dataclass
class PrefillOnlySchedulerOutput(SchedulerOutput):
    requests: Iterable[PrefillOnlyRequest]

    def is_empty(self) -> bool:
        return not self.requests


class PrefillOnlyRequestOutput(RequestOutput):

    def __init__(self, request_id: str, outputs: torch.Tensor,
                 prompt_token_ids: List[int], finished: bool):
        self.request_id = request_id
        self.prompt_token_ids = prompt_token_ids
        self.finished = finished
        self.outputs = outputs

    def __repr__(self):
        return (f"PrefillOnlyRequestOutput(request_id='{self.request_id}', "
                f"outputs={repr(self.outputs)}, "
                f"prompt_token_ids={self.prompt_token_ids}, "
                f"finished={self.finished})")
