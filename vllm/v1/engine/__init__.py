import enum
from dataclasses import dataclass
from multiprocessing.process import BaseProcess
from typing import List, Optional, Union

import msgspec

from vllm.lora.request import LoRARequest
from vllm.multimodal import MultiModalKwargs, MultiModalPlaceholderDict
from vllm.sampling_params import RequestOutputKind, SamplingParams


@dataclass
class BackgroundProcHandle:
    proc: BaseProcess
    ready_path: str
    input_path: str
    output_path: str


class DetokenizerRequest(
        msgspec.Struct,
        array_like=True,  # type: ignore[call-arg]
        omit_defaults=True,  # type: ignore[call-arg]
        gc=False):  # type: ignore[call-arg]

    request_id: str
    prompt: Optional[str]
    prompt_token_ids: List[int]
    skip_special_tokens: bool
    spaces_between_special_tokens: bool
    output_kind: RequestOutputKind

    stop: List[str]
    include_stop_str_in_output: bool


class DetokenizerOutput(
        msgspec.Struct,
        array_like=True,  # type: ignore[call-arg]
        omit_defaults=True,  # type: ignore[call-arg]
        gc=False):  # type: ignore[call-arg]

    request_id: str
    text: str
    token_ids: List[int]
    finished: bool


class DetokenizerOutputs(
        msgspec.Struct,
        array_like=True,  # type: ignore[call-arg]
        omit_defaults=True,  # type: ignore[call-arg]
        gc=False):  # type: ignore[call-arg]

    #NOTE(Nick): We could consider ways to make this more compact,
    # e.g. columnwise layout and using an int enum for finish/stop reason

    # [num_reqs]
    outputs: List[DetokenizerOutput]


@dataclass
class EngineCoreRequest:

    # NOTE: prompt and prompt_token_ids should be DecoderOnlyInput,
    # but this object is currently not playing well with msgspec
    # due to circular imports and typing we have in data.py

    request_id: str
    #NOTE(Nick): I don't think we need to pass prompt here since it should
    # always be tokenized?
    prompt: Optional[str]
    prompt_token_ids: List[int]
    mm_inputs: Optional[List[Optional[MultiModalKwargs]]]
    mm_hashes: Optional[List[str]]
    mm_placeholders: Optional[MultiModalPlaceholderDict]
    sampling_params: SamplingParams
    eos_token_id: Optional[int]
    arrival_time: float
    lora_request: Optional[LoRARequest]


class EngineCoreOutput(
        msgspec.Struct,
        array_like=True,  # type: ignore[call-arg]
        omit_defaults=True,  # type: ignore[call-arg]
        gc=False):  # type: ignore[call-arg]

    request_id: str
    prompt_token_ids: List[int]
    prompt: str
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


@dataclass
class EngineCoreProfile:
    is_start: bool


class DetokenizerRequestType(enum.Enum):
    """
    Request types defined as hex byte strings, so it can be sent over sockets
    without separate encoding step.
    """
    NEW = b'\x00'
    OUT = b'\x01'


class EngineCoreRequestType(enum.Enum):
    """
    Request types defined as hex byte strings, so it can be sent over sockets
    without separate encoding step.
    """
    ADD = b'\x00'
    ABORT = b'\x01'
    PROFILE = b'\x02'


EngineCoreRequestUnion = Union[EngineCoreRequest, EngineCoreProfile, List[str]]
