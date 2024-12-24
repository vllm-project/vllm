import enum
from dataclasses import dataclass
from typing import List, Optional, Union

import msgspec

from vllm.lora.request import LoRARequest
from vllm.multimodal import MultiModalKwargs, MultiModalPlaceholderDict
from vllm.sampling_params import SamplingParams


@dataclass
class EngineRequest:
    request_id: str
    prompt: Optional[str]
    prompt_token_ids: List[int]
    mm_inputs: Optional[List[Optional[MultiModalKwargs]]]
    mm_hashes: Optional[List[str]]
    mm_placeholders: Optional[MultiModalPlaceholderDict]
    sampling_params: SamplingParams
    eos_token_id: Optional[int]
    arrival_time: float
    lora_request: Optional[LoRARequest]


@dataclass
class EngineAbortRequest:
    request_ids: List[str]


@dataclass
class EngineProfileRequest:
    is_start: bool


EngineRequestUnion = Union[EngineRequest, EngineAbortRequest,
                           EngineProfileRequest]


class EngineCoreOutput(
        msgspec.Struct,
        array_like=True,  # type: ignore[call-arg]
        omit_defaults=True,  # type: ignore[call-arg]
        gc=False):  # type: ignore[call-arg]

    request_id: str
    new_token_ids: List[int]
    finished: bool
    finish_reason: Optional[str] = None
    stop_reason: Union[int, str, None] = None


class EngineCoreOutputs(
        msgspec.Struct,
        array_like=True,  # type: ignore[call-arg]
        omit_defaults=True,  # type: ignore[call-arg]
        gc=False):  # type: ignore[call-arg]

    #NOTE(Nick): We could consider ways to make this more compact,
    # e.g. columnwise layout and using an int enum for finish/stop reason

    # [num_reqs]
    outputs: List[EngineCoreOutput]


class EngineRequestType(enum.Enum):
    """	
    Request types defined as hex byte strings, so it can be sent over sockets	
    without separate encoding step.	
    """
    FROM_ENGINE_CORE = b'\x00'
    FROM_ENGINE = b'\x01'
