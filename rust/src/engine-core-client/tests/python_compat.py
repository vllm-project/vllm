#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "msgspec>=0.19,<1",
# ]
# ///

from enum import Enum, IntEnum

import msgspec


class RequestOutputKind(Enum):
    CUMULATIVE = 0
    DELTA = 1
    FINAL_ONLY = 2


class FinishReason(IntEnum):
    STOP = 0
    LENGTH = 1
    ABORT = 2
    ERROR = 3
    REPETITION = 4


class EngineCoreSamplingParams(msgspec.Struct, dict=True):
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = 0
    seed: int | None = None
    max_tokens: int = 65536
    min_tokens: int = 0
    min_p: float = 0.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    repetition_penalty: float = 1.0
    stop_token_ids: list[int] = []
    _eos_token_id: int | None = None
    _all_stop_token_ids: set[int] = set()
    output_kind: RequestOutputKind = RequestOutputKind.CUMULATIVE


class EngineCoreRequest(
    msgspec.Struct,
    array_like=True,
    omit_defaults=True,
):
    request_id: str
    prompt_token_ids: list[int] | None
    mm_features: object | None
    sampling_params: EngineCoreSamplingParams | None
    pooling_params: object | None
    arrival_time: float
    lora_request: object | None = None
    cache_salt: str | None = None
    data_parallel_rank: int | None = None
    prompt_embeds: object | None = None
    client_index: int = 0
    current_wave: int = 0
    priority: int = 0
    trace_headers: dict[str, str] | None = None
    resumable: bool = False
    external_req_id: str | None = None
    reasoning_ended: bool | None = None


class EngineCoreOutput(
    msgspec.Struct,
    array_like=True,
    omit_defaults=True,
):
    request_id: str
    new_token_ids: list[int]
    new_logprobs: object | None = None
    new_prompt_logprobs_tensors: object | None = None
    pooling_output: object | None = None
    finish_reason: FinishReason | None = None
    stop_reason: int | str | None = None
    events: object | None = None
    kv_transfer_params: object | None = None
    trace_headers: object | None = None
    num_cached_tokens: int = 0
    num_external_computed_tokens: int = 0
    routed_experts: object | None = None
    num_nans_in_logits: int = 0


class EngineCoreOutputs(
    msgspec.Struct,
    array_like=True,
    omit_defaults=True,
):
    engine_index: int = 0
    outputs: list[EngineCoreOutput] = []
    scheduler_stats: object | None = None
    timestamp: float = 0.0
    utility_output: object | None = None
    finished_requests: set[str] | None = None
    wave_complete: int | None = None
    start_wave: int | None = None


request = EngineCoreRequest(
    request_id="req-1",
    prompt_token_ids=[11, 22],
    mm_features=None,
    sampling_params=EngineCoreSamplingParams(
        temperature=0.8,
        top_p=0.9,
        top_k=8,
        seed=None,
        max_tokens=32,
        min_tokens=1,
        min_p=0.0,
        frequency_penalty=0.0,
        presence_penalty=0.0,
        repetition_penalty=1.0,
        stop_token_ids=[151643],
        _eos_token_id=151645,
        _all_stop_token_ids={151643, 151645},
        output_kind=RequestOutputKind.FINAL_ONLY,
    ),
    pooling_params=None,
    arrival_time=42.5,
    client_index=0,
)

outputs = EngineCoreOutputs(
    outputs=[
        EngineCoreOutput(
            request_id="req-1",
            new_token_ids=[7, 8],
            finish_reason=FinishReason.LENGTH,
        )
    ],
    finished_requests={"req-1"},
)

print(msgspec.msgpack.encode(request).hex())
print(msgspec.msgpack.encode(outputs).hex())
