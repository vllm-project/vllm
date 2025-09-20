# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import enum
import time
from collections.abc import Mapping
from functools import partial
from typing import TYPE_CHECKING, Any, Callable, Optional, Union

import torch

from vllm.multimodal.inputs import MultiModalFeatureSpec
from vllm.pooling_params import PoolingParams
from vllm.sampling_params import SamplingParams
from vllm.utils import length_from_prompt_token_ids_or_embeds
from vllm.v1.engine import (EngineCoreEvent, EngineCoreEventType,
                            EngineCoreRequest, FinishReason)
from vllm.v1.structured_output.request import StructuredOutputRequest
from vllm.v1.utils import ConstantList

if TYPE_CHECKING:
    from vllm.lora.request import LoRARequest
    from vllm.v1.core.kv_cache_utils import BlockHash


class Request:

    def __init__(
        self,
        request_id: str,
        prompt_token_ids: Optional[list[int]],
        sampling_params: Optional[SamplingParams],
        pooling_params: Optional[PoolingParams],
        eos_token_id: Optional[int],
        client_index: int = 0,
        arrival_time: Optional[float] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        mm_features: Optional[list[MultiModalFeatureSpec]] = None,
        lora_request: Optional["LoRARequest"] = None,
        structured_output_request: Optional["StructuredOutputRequest"] = None,
        cache_salt: Optional[str] = None,
        priority: int = 0,
        trace_headers: Optional[Mapping[str, str]] = None,
        block_hasher: Optional[Callable[["Request"],
                                        list["BlockHash"]]] = None,
    ) -> None:
        self.request_id = request_id
        self.client_index = client_index
        self.priority = priority
        self.sampling_params = sampling_params
        self.pooling_params = pooling_params
        # Because of LoRA, the eos token id can be different for each request.
        self.eos_token_id = eos_token_id
        self.lora_request = lora_request
        self.structured_output_request = structured_output_request
        self.arrival_time = arrival_time if arrival_time is not None else \
            time.time()

        self.status = RequestStatus.WAITING
        self.use_structured_output = False
        self.events: list[EngineCoreEvent] = []
        self.stop_reason: Union[int, str, None] = None

        # P/D: Connector-specific KV transfer parameters.
        self.kv_transfer_params: Optional[dict[str, Any]] = None

        if pooling_params is not None:
            # Pooling models.
            self.max_tokens = 1
        elif sampling_params is not None:
            # Generative models.
            assert sampling_params.max_tokens is not None
            self.max_tokens = sampling_params.max_tokens
            if sampling_params.structured_outputs is not None:
                self.status = RequestStatus.WAITING_FOR_FSM
                self.use_structured_output = True

            if sampling_params.extra_args is not None:
                self.kv_transfer_params = \
                    sampling_params.extra_args.get("kv_transfer_params")
        else:
            raise ValueError(
                "sampling_params and pooling_params can't both be unset")

        self.prompt_token_ids = prompt_token_ids
        self.prompt_embeds = prompt_embeds
        self.num_prompt_tokens = length_from_prompt_token_ids_or_embeds(
            prompt_token_ids, prompt_embeds)
        self._output_token_ids: list[int] = []
        self._all_token_ids: list[int] = self.prompt_token_ids.copy(
        ) if self.prompt_token_ids is not None else [0
                                                     ] * self.num_prompt_tokens
        self.num_output_placeholders = 0  # Used in async scheduling.
        self.spec_token_ids: list[int] = []
        self.num_computed_tokens = 0
        self.cache_salt: Optional[str] = cache_salt

        # Multi-modal related
        self.mm_features = mm_features or []
        self.num_encoder_inputs = len(self.mm_features)
        self.has_encoder_inputs = self.num_encoder_inputs > 0

        # Read-only views
        # Prevent directly appending to these lists since
        # they should also be updated simultaneously.
        self.output_token_ids = ConstantList(self._output_token_ids)
        self.all_token_ids = ConstantList(self._all_token_ids)
        # trace_headers
        self.trace_headers = trace_headers
        # State
        # The number of tokens with prefix cache hits.
        self.num_cached_tokens = -1

        # The number of NaNs in logits. A value greater than 0
        # indicates that the output is corrupted
        self.num_nans_in_logits = 0

        self.block_hashes: list[BlockHash] = []
        self.get_hash_new_full_blocks: Optional[Callable[
            [], list[BlockHash]]] = None
        if block_hasher is not None:
            self.get_hash_new_full_blocks = partial(block_hasher, self)
            self.block_hashes = self.get_hash_new_full_blocks()

    @classmethod
    def from_engine_core_request(
        cls, request: EngineCoreRequest,
        block_hasher: Optional[Callable[["Request"], list["BlockHash"]]]
    ) -> "Request":
        return cls(
            request_id=request.request_id,
            client_index=request.client_index,
            prompt_token_ids=request.prompt_token_ids,
            prompt_embeds=request.prompt_embeds,
            mm_features=request.mm_features,
            sampling_params=request.sampling_params,
            pooling_params=request.pooling_params,
            eos_token_id=request.eos_token_id,
            arrival_time=request.arrival_time,
            lora_request=request.lora_request,
            structured_output_request=StructuredOutputRequest(
                sampling_params=request.sampling_params) \
                    if request.sampling_params else None,
            cache_salt=request.cache_salt,
            priority=request.priority,
            trace_headers=request.trace_headers,
            block_hasher=block_hasher,
        )

    def append_output_token_ids(
        self,
        token_ids: Union[int, list[int]],
    ) -> None:
        if isinstance(token_ids, int):
            self._output_token_ids.append(token_ids)
            self._all_token_ids.append(token_ids)
        else:
            self._output_token_ids.extend(token_ids)
            self._all_token_ids.extend(token_ids)

        if self.get_hash_new_full_blocks is not None:
            self.block_hashes.extend(self.get_hash_new_full_blocks())

    @property
    def is_output_corrupted(self) -> bool:
        return self.num_nans_in_logits > 0

    @property
    def num_tokens(self) -> int:
        return len(self._all_token_ids)

    @property
    def num_tokens_with_spec(self) -> int:
        return len(self._all_token_ids) + len(self.spec_token_ids)

    @property
    def num_output_tokens(self) -> int:
        return len(self._output_token_ids)

    def is_finished(self) -> bool:
        return RequestStatus.is_finished(self.status)

    def get_finished_reason(self) -> Union[FinishReason, None]:
        return RequestStatus.get_finished_reason(self.status)

    def get_num_encoder_tokens(self, input_id: int) -> int:
        assert input_id < len(self.mm_features)
        num_tokens = self.mm_features[input_id].mm_position.length
        return num_tokens

    def record_event(
        self,
        event_type: EngineCoreEventType,
        timestamp: Optional[float] = None,
    ) -> None:
        self.events.append(EngineCoreEvent.new_event(event_type, timestamp))

    def take_events(self) -> Optional[list[EngineCoreEvent]]:
        if not self.events:
            return None
        events, self.events = self.events, []
        return events


class RequestStatus(enum.IntEnum):
    """Status of a request."""
    WAITING = enum.auto()
    WAITING_FOR_FSM = enum.auto()
    WAITING_FOR_REMOTE_KVS = enum.auto()
    RUNNING = enum.auto()
    PREEMPTED = enum.auto()
    # Note: anything after PREEMPTED will be considered
    # as a finished status.
    FINISHED_STOPPED = enum.auto()
    FINISHED_LENGTH_CAPPED = enum.auto()
    FINISHED_ABORTED = enum.auto()
    FINISHED_IGNORED = enum.auto()

    def __str__(self):
        return self.name

    @staticmethod
    def is_finished(status: "RequestStatus") -> bool:
        return status > RequestStatus.PREEMPTED

    @staticmethod
    def get_finished_reason(
            status: "RequestStatus") -> Union[FinishReason, None]:
        return _FINISHED_REASON_MAP.get(status)


# Mapping of finished statuses to their finish reasons.
# NOTE: The ignored requests are the requests whose prompt lengths
# are longer than the model's length cap. Therefore, the stop
# reason should also be "length" as in OpenAI API.
_FINISHED_REASON_MAP = {
    RequestStatus.FINISHED_STOPPED: FinishReason.STOP,
    RequestStatus.FINISHED_LENGTH_CAPPED: FinishReason.LENGTH,
    RequestStatus.FINISHED_ABORTED: FinishReason.ABORT,
    RequestStatus.FINISHED_IGNORED: FinishReason.LENGTH,
}
