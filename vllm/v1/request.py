import enum
from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Optional, Union

from vllm.lora.request import LoRARequest
from vllm.sampling_params import SamplingParams
from vllm.v1.outputs import CompletionOutput, RequestOutput

if TYPE_CHECKING:
    from vllm.inputs import DecoderOnlyInputs


class Request:

    def __init__(
        self,
        request_id: str,
        inputs: "DecoderOnlyInputs",
        arrival_time: float,
        sampling_params: SamplingParams,
        lora_request: Optional[LoRARequest] = None,
    ) -> None:
        self.request_id = request_id
        self.inputs = inputs
        self.metrics = RequestMetrics(arrival_time=arrival_time,
                                      last_token_time=arrival_time,
                                      first_scheduled_time=None,
                                      first_token_time=None,
                                      time_in_queue=None)
        self.sampling_params = sampling_params
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

        # TODO: Support `n` > 1.
        # OPTIMIZATION: Cache the request output and update it incrementally.
        # This is used to avoid creating a new RequestOutput object every step.
        self._completion_output = CompletionOutput(
            index=0,
            text=self.output_text,
            token_ids=self.output_token_ids,
            logprobs=None,  # TODO
            finish_reason=None,
            stop_reason=None,
        )
        self._request_output = RequestOutput(
            request_id=self.request_id,
            prompt=self.prompt,
            prompt_token_ids=self.prompt_token_ids,
            outputs=[self._completion_output],
            finished=False,
        )

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

    def make_output(
        self,
        num_output_tokens: int,
        output_text: str,
        finished: bool,
    ) -> RequestOutput:
        self._completion_output.text = output_text
        self._completion_output.token_ids = (
            self.output_token_ids[:num_output_tokens])
        if finished:
            self._completion_output.finish_reason = self.get_finished_reason()
            self._completion_output.stop_reason = self.stop_reason
            self._request_output.finished = finished
        return self._request_output


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


@dataclass
class RequestMetrics:
    """Metrics associated with a request.

    Attributes:
        arrival_time: The time when the request arrived.
        first_scheduled_time: The time when the request was first scheduled.
        first_token_time: The time when the first token was generated.
        time_in_queue: The time the request spent in the queue.
        finished_time: The time when the request was finished.
        scheduler_time: The time spent in the scheduler when this request was
                        being considered by the scheduler.
        model_forward_time: The time spent in the model forward pass when this
                            request was in the batch.
        model_execute_time: The time spent in the model execute function. This
                            will include model forward, block/sync across
                            workers, cpu-gpu sync time and sampling time.
    """
    arrival_time: float
    last_token_time: float
    first_scheduled_time: Optional[float]
    first_token_time: Optional[float]
    time_in_queue: Optional[float]
    finished_time: Optional[float] = None
    scheduler_time: Optional[float] = None
    model_forward_time: Optional[float] = None
    model_execute_time: Optional[float] = None
