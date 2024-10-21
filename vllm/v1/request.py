import enum
from typing import TYPE_CHECKING, List, Optional, Union

from vllm.lora.request import LoRARequest
from vllm.sampling_params import SamplingParams
from vllm.sequence import RequestMetrics

if TYPE_CHECKING:
    from vllm.inputs import DecoderOnlyInputs


class Request:

    def __init__(
        self,
        request_id: str,
        inputs: "DecoderOnlyInputs",
        sampling_params: SamplingParams,
        eos_token_id: Optional[int],
        arrival_time: float,
        lora_request: Optional[LoRARequest] = None,
    ) -> None:
        self.request_id = request_id
        self.inputs = inputs
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

        self.prompt = inputs.get("prompt")
        self.prompt_token_ids = inputs["prompt_token_ids"]
        self.num_prompt_tokens = len(self.prompt_token_ids)
        self.output_token_ids: List[int] = []
        self.output_text = ""
        self.num_computed_tokens = 0

    @property
    def num_tokens(self) -> int:
        return self.num_prompt_tokens + len(self.output_token_ids)

    @property
    def num_output_tokens(self) -> int:
        return len(self.output_token_ids)

    def is_finished(self) -> bool:
        return RequestStatus.is_finished(self.status)

    def get_finished_reason(self) -> Union[str, None]:
        return RequestStatus.get_finished_reason(self.status)


class RequestStatus(enum.IntEnum):
    """Status of a sequence."""
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
# NOTE: The ignored sequences are the sequences whose prompt lengths
# are longer than the model's length cap. Therefore, the stop
# reason should also be "length" as in OpenAI API.
_FINISHED_REASON_MAP = {
    RequestStatus.FINISHED_STOPPED: "stop",
    RequestStatus.FINISHED_LENGTH_CAPPED: "length",
    RequestStatus.FINISHED_ABORTED: "abort",
    RequestStatus.FINISHED_IGNORED: "length",
}
