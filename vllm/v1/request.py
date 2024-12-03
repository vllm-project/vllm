import enum
from typing import List, Optional, Union

from vllm.inputs import DecoderOnlyInputs, SingletonInputsAdapter, token_inputs
from vllm.lora.request import LoRARequest
from vllm.multimodal import MultiModalKwargs
from vllm.sampling_params import SamplingParams
from vllm.sequence import RequestMetrics
from vllm.v1.engine import EngineCoreRequest
from vllm.v1.utils import ConstantList


class Request:

    def __init__(
        self,
        request_id: str,
        inputs: DecoderOnlyInputs,
        sampling_params: SamplingParams,
        eos_token_id: Optional[int],
        arrival_time: float,
        lora_request: Optional[LoRARequest] = None,
    ) -> None:
        self.request_id = request_id
        self.inputs = SingletonInputsAdapter(inputs)
        self.sampling_params = sampling_params
        # Because of LoRA, the eos token id can be different for each request.
        self.eos_token_id = eos_token_id
        self.metrics = RequestMetrics(arrival_time=arrival_time,
                                      last_token_time=arrival_time,
                                      first_scheduled_time=None,
                                      first_token_time=None,
                                      time_in_queue=None)
        self.lora_request = lora_request

        self.status = RequestStatus.WAITING
        self.stop_reason: Union[int, str, None] = None
        assert sampling_params.max_tokens is not None
        self.max_tokens = sampling_params.max_tokens

        self.prompt = self.inputs.prompt
        self.prompt_token_ids = self.inputs.prompt_token_ids
        self.num_prompt_tokens = len(self.prompt_token_ids)
        self._output_token_ids: List[int] = []
        self._all_token_ids: List[int] = self.prompt_token_ids.copy()
        self.num_computed_tokens = 0

        mm_positions = self.inputs.multi_modal_placeholders
        if mm_positions:
            # FIXME(woosuk): Support other modalities.
            self.mm_positions = mm_positions.get("image", [])
        else:
            self.mm_positions = []
        # Output of the mm input mapper (e.g., image tensors).
        if self.inputs.multi_modal_inputs:
            self.mm_inputs = self.inputs.multi_modal_inputs
        else:
            self.mm_inputs: List[MultiModalKwargs] = []

    @classmethod
    def from_engine_core_request(cls, request: EngineCoreRequest) -> "Request":
        return cls(
            request_id=request.request_id,
            inputs=token_inputs(
                prompt_token_ids=request.prompt_token_ids,
                prompt=request.prompt,
                multi_modal_data=None,
                multi_modal_inputs=request.mm_inputs,
                multi_modal_placeholders=request.mm_placeholders,
                mm_processor_kwargs=None,
            ),
            sampling_params=request.sampling_params,
            eos_token_id=request.eos_token_id,
            arrival_time=request.arrival_time,
            lora_request=request.lora_request,
        )

    @property
    def output_token_ids(self) -> ConstantList[int]:
        # Prevent directly appending to the output_token_ids since
        # all_token_ids should also be updated simultaneously.
        return ConstantList(self._output_token_ids)

    @property
    def all_token_ids(self) -> ConstantList[int]:
        # Prevent directly appending to the all_token_ids since
        # output_token_ids should also be updated simultaneously
        return ConstantList(self._all_token_ids)

    def append_output_token_ids(
        self,
        token_ids: Union[int, List[int]],
    ) -> None:
        if isinstance(token_ids, int):
            token_ids = [token_ids]
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

    def get_finished_reason(self) -> Union[str, None]:
        return RequestStatus.get_finished_reason(self.status)

    def has_encoder_inputs(self) -> bool:
        return len(self.mm_inputs) > 0

    @property
    def num_encoder_inputs(self) -> int:
        return len(self.mm_positions)

    def get_num_encoder_tokens(self, input_id: int) -> int:
        assert input_id < len(self.mm_positions)
        num_tokens = self.mm_positions[input_id]["length"]
        return num_tokens


class RequestStatus(enum.IntEnum):
    """Status of a request."""
    WAITING = 0
    RUNNING = 1
    PREEMPTED = 2
    # Note: anything after PREEMPTED (2) will be considered
    # as a finished status.
    FINISHED_STOPPED = 3
    FINISHED_LENGTH_CAPPED = 4
    FINISHED_ABORTED = 5
    FINISHED_IGNORED = 6

    @staticmethod
    def is_finished(status: "RequestStatus") -> bool:
        return status > RequestStatus.PREEMPTED

    @staticmethod
    def get_finished_reason(status: "RequestStatus") -> Union[str, None]:
        return _FINISHED_REASON_MAP.get(status)


# Mapping of finished statuses to their finish reasons.
# NOTE: The ignored requests are the requests whose prompt lengths
# are longer than the model's length cap. Therefore, the stop
# reason should also be "length" as in OpenAI API.
_FINISHED_REASON_MAP = {
    RequestStatus.FINISHED_STOPPED: "stop",
    RequestStatus.FINISHED_LENGTH_CAPPED: "length",
    RequestStatus.FINISHED_ABORTED: "abort",
    RequestStatus.FINISHED_IGNORED: "length",
}
