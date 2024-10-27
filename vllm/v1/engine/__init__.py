from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Optional, Union

import msgspec

from vllm.lora.request import LoRARequest
from vllm.sampling_params import SamplingParams

if TYPE_CHECKING:
    from vllm.inputs import DecoderOnlyInputs


class EngineCoreRequest(msgspec.Struct):

    request_id: str
    inputs: "DecoderOnlyInputs"
    sampling_params: SamplingParams
    eos_token_id: Optional[int]
    arrival_time: float
    lora_request: Optional[LoRARequest]


@dataclass
class EngineCoreOutput:

    request_id: str
    new_token_ids: List[int]
    finished: bool
    finish_reason: Optional[str] = None
    stop_reason: Union[int, str, None] = None


class EngineCoreOutputs(msgspec.Struct):

    # [num_reqs]
    outputs: List[EngineCoreOutput]
