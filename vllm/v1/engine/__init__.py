import asyncio
from dataclasses import dataclass
from typing import List, Optional, Union

import msgspec

from vllm.lora.request import LoRARequest
from vllm.outputs import RequestOutput
from vllm.sampling_params import RequestOutputKind, SamplingParams

LLM_ENGINE_CORE_READY_STR = "READY"
POLLING_TIMEOUT_MS = 5000


@dataclass
class DetokenizerRequest:

    request_id: str
    prompt: Optional[str]
    prompt_token_ids: List[int]
    skip_special_tokens: bool
    spaces_between_special_tokens: bool
    output_kind: RequestOutputKind

    # Queue for streaming outputs to clients.
    output_queue: Optional[asyncio.Queue[RequestOutput]] = None


class EngineCoreRequest(msgspec.Struct):

    # NOTE: prompt and prompt_token_ids should be DecoderOnlyInput,
    # but this object is currently not playing well with msgspec
    # due to circular imports and typing we have in data.py

    request_id: str
    prompt: Optional[str]
    prompt_token_ids: List[int]
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
