# SPDX-License-Identifier: Apache-2.0

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

# These are possible values of RequestOutput.finish_reason,
# so form part of the external API.
FINISH_REASON_STRINGS = ("stop", "length", "abort")


class FinishReason(enum.IntEnum):
    """
    Reason a request finished - stop, length, or abort.

    Int rather than Str for more compact serialization.

    stop - a stop string was emitted
    length - max_tokens was consumed, or max_model_len was reached
    abort - aborted for another reason

    """
    STOP = 0
    LENGTH = 1
    ABORT = 2

    def __str__(self):
        return FINISH_REASON_STRINGS[self.value]


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
    finish_reason: Optional[FinishReason] = None
    stop_reason: Union[int, str, None] = None


class EngineCoreOutputs(
        msgspec.Struct,
        array_like=True,  # type: ignore[call-arg]
        omit_defaults=True,  # type: ignore[call-arg]
        gc=False):  # type: ignore[call-arg]

    #NOTE(Nick): We could consider ways to make this more compact,
    # e.g. columnwise layout

    # [num_reqs]
    outputs: List[EngineCoreOutput]
    scheduler_stats: SchedulerStats


@dataclass
class EngineCoreProfile:
    is_start: bool


@dataclass
class EngineCoreResetPrefixCache:
    pass


class EngineCoreRequestType(enum.Enum):
    """
    Request types defined as hex byte strings, so it can be sent over sockets
    without separate encoding step.
    """
    ADD = b'\x00'
    ABORT = b'\x01'
    PROFILE = b'\x02'
    RESET_PREFIX_CACHE = b'\x03'


EngineCoreRequestUnion = Union[EngineCoreRequest, EngineCoreProfile,
                               EngineCoreResetPrefixCache, List[str]]
