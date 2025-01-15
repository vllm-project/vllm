import enum
from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Optional, Union

import msgspec

from vllm.v1.metrics.stats import SchedulerStats

if TYPE_CHECKING:
    from vllm.lora.request import LoRARequest
    from vllm.multimodal import MultiModalKwargs
    from vllm.multimodal.inputs import PlaceholderRange
    from vllm.sampling_params import SamplingParams


@dataclass
class EngineCoreRequest:

    # NOTE: prompt and prompt_token_ids should be DecoderOnlyInput,
    # but this object is currently not playing well with msgspec
    # due to circular imports and typing we have in data.py

    request_id: str
    # NOTE(ywang96): original text prompt is needed when a request is added to
    # Detokenizer, but set to None when it is added to EngineCoreClient.
    prompt: Optional[str]
    prompt_token_ids: List[int]
    mm_inputs: Optional[List[Optional["MultiModalKwargs"]]]
    mm_hashes: Optional[List[str]]
    mm_placeholders: Optional[List["PlaceholderRange"]]
    sampling_params: "SamplingParams"
    eos_token_id: Optional[int]
    arrival_time: float
    lora_request: Optional["LoRARequest"]


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
    scheduler_stats: SchedulerStats


@dataclass
class EngineCoreProfile:
    is_start: bool


class EngineCoreRequestType(enum.Enum):
    """
    Request types defined as hex byte strings, so it can be sent over sockets
    without separate encoding step.
    """
    ADD = b'\x00'
    ABORT = b'\x01'
    PROFILE = b'\x02'


EngineCoreRequestUnion = Union[EngineCoreRequest, EngineCoreProfile, List[str]]
