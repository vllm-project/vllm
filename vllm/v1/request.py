# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import enum
from typing import TYPE_CHECKING, Any, Optional, Union

from vllm.multimodal.inputs import MultiModalKwargs, PlaceholderRange
from vllm.sampling_params import SamplingParams
from vllm.utils import is_list_of
from vllm.v1.engine import (EngineCoreEvent, EngineCoreEventType,
                            EngineCoreRequest, FinishReason)
from vllm.v1.structured_output.request import StructuredOutputRequest
from vllm.v1.utils import ConstantList

if TYPE_CHECKING:
    from vllm.lora.request import LoRARequest

class RequestParams:
    """
    "Constant" parameters for a request. This should be static during genera    tion
     but current this is violated by `structured_output_request` which is 
     stateful
    """
    def __init__(
        self,
        request_id: str,
        prompt_token_ids: list[int],
        multi_modal_inputs: Optional[list[MultiModalKwargs]],
        multi_modal_hashes: Optional[list[str]],
        multi_modal_placeholders: Optional[list[PlaceholderRange]],
        sampling_params: SamplingParams,
        eos_token_id: Optional[int],
        client_index: int = 0,
        lora_request: Optional["LoRARequest"] = None,
        structured_output_request: Optional["StructuredOutputRequest"] = None,
        cache_salt: Optional[str] = None,
    ) -> None:
        self.request_id = request_id
        self.client_index = client_index
        self.sampling_params = sampling_params
        # Because of LoRA, the eos token id can be different for each request.
        self.eos_token_id = eos_token_id
        self.lora_request = lora_request

        # P/D: Connector-specific KV transfer parameters.
        kv_params = (None if sampling_params.extra_args is None else
                     sampling_params.extra_args.get("kv_transfer_params"))
        self.kv_transfer_params: Optional[dict[str, Any]] = kv_params

        # NOTE(lucas): Note sure if this belongs in "params" since this is
        # stateful with regards to each generated token.
        self.structured_output_request = structured_output_request

        assert sampling_params.max_tokens is not None
        self.max_tokens = sampling_params.max_tokens

        self.prompt_token_ids = prompt_token_ids
        self.num_prompt_tokens = len(self.prompt_token_ids)
        self.cache_salt: Optional[str] = cache_salt

        # Multi-modal related
        self.mm_positions = multi_modal_placeholders or []
        self.mm_inputs = multi_modal_inputs or []
        self.mm_hashes: list[str] = multi_modal_hashes or []
        self.num_encoder_inputs = len(self.mm_inputs)
        self.has_encoder_inputs = self.num_encoder_inputs > 0

        # Sanity check
        assert len(self.mm_inputs) == len(self.mm_positions)
        if self.mm_hashes:
            assert len(self.mm_inputs) == len(self.mm_hashes)

    @classmethod
    def from_engine_core_request(cls, request: EngineCoreRequest) -> "RequestParams":
        if request.mm_inputs is not None:
            assert isinstance(request.mm_inputs, list)
            assert is_list_of(request.mm_inputs, MultiModalKwargs), (
                "mm_inputs was not updated in EngineCore.add_request")

        return cls(
            request_id=request.request_id,
            client_index=request.client_index,
            prompt_token_ids=request.prompt_token_ids,
            multi_modal_inputs=request.mm_inputs,
            multi_modal_hashes=request.mm_hashes,
            multi_modal_placeholders=request.mm_placeholders,
            sampling_params=request.sampling_params,
            eos_token_id=request.eos_token_id,
            lora_request=request.lora_request,
            structured_output_request=StructuredOutputRequest(
                sampling_params=request.sampling_params),
            cache_salt=request.cache_salt,
        )

    def get_num_encoder_tokens(self, input_id: int) -> int:
        assert input_id < len(self.mm_positions)
        num_tokens = self.mm_positions[input_id].length
        return num_tokens

    @property
    def use_structured_output(self) -> bool:
        return self.sampling_params.guided_decoding is not None

class RequestGenerationState:
    """
    Track the generated tokens for a request.
    """
    
    def __init__(self, params: RequestParams) -> None:
        # convenience alias
        self.request_id = params.request_id
        self.client_index = params.client_index
        
        self.params = params

        self.stop_reason: Union[int, str, None] = None

        self._output_token_ids: list[int] = []
        self._all_token_ids: list[int] = params.prompt_token_ids.copy()

        # Read-only views
        # Prevent directly appending to these lists since
        # they should also be updated simultaneously.
        self.output_token_ids = ConstantList(self._output_token_ids)
        self.all_token_ids = ConstantList(self._all_token_ids)

    # TODO(lucas) This is a hack to reduce the `.params` additions in the PR we
    #  should probably remove it
    def __getattribute__(self, name: str) -> Any:
        if name == "params":
            return object.__getattribute__(self, name)

        params = object.__getattribute__(self, "params")
        if hasattr(params, name):
            return getattr(params, name)

        return object.__getattribute__(self, name)

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

    @property
    def num_tokens(self) -> int:
        return len(self._all_token_ids)

    @property
    def num_output_tokens(self) -> int:
        return len(self._output_token_ids)

    def is_finished(self) -> bool:
        return RequestStatus.is_finished(self.status)

    def get_finished_reason(self) -> Union[FinishReason, None]:
        return RequestStatus.get_finished_reason(self.status)


class RequestSchedulerState:
    """
    Track the scheduler state for a request. i.e. all the information the 
     scheduler needs to know to schedule the next tokens for the request.
    """
    def __init__(self, params: RequestParams) -> None:
        # Convenience alias
        self.request_id = params.request_id
        
        self.params = params

        self.status = (RequestStatus.WAITING_FOR_FSM
                       if params.sampling_params.guided_decoding is not None else
                       RequestStatus.WAITING)
        self.spec_token_ids: list[int] = []

        # TODO(lucas): these names match the current names used in the scheduler
        # but I think these names can be improved
        # This is the number of tokens that are known (prompt + any generated output tokens)
        self.num_tokens = params.num_prompt_tokens 
        # This is the tokens we have a valid KV-cache for
        self.num_computed_tokens = 0

        self.events: list[EngineCoreEvent] = []

        # State
        # The number of tokens with prefix cache hits.
        self.num_cached_tokens = -1

    # TODO(lucas) This is a hack to reduce the `.params` additions in the PR we
    #  should probably remove it
    def __getattribute__(self, name: str) -> Any:
        if name == "params":
            return object.__getattribute__(self, name)

        params = object.__getattribute__(self, "params")
        if hasattr(params, name):
            return getattr(params, name)

        return object.__getattribute__(self, name)

    @property
    def num_tokens_with_spec(self):
        return self.num_tokens + len(self.spec_token_ids)

    @property
    def num_output_tokens(self) -> int:
        return self.num_tokens - self.params.num_prompt_tokens

    def is_finished(self) -> bool:
        return RequestStatus.is_finished(self.status)

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
