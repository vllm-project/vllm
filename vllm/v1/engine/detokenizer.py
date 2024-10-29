from dataclasses import dataclass
from typing import Dict, List, Optional

from vllm.logger import init_logger
from vllm.outputs import CompletionOutput, RequestOutput
from vllm.sampling_params import RequestOutputKind
from vllm.transformers_utils.detokenizer_utils import (
    AnyTokenizer, convert_prompt_ids_to_tokens, detokenize_incrementally)
from vllm.transformers_utils.tokenizer import get_tokenizer
from vllm.v1.engine import DetokenizerRequest, EngineCoreOutput
from vllm.v1.engine.async_stream import AsyncStream

logger = init_logger(__name__)


@dataclass
class DetokenizerRequestState:

    # Generation data
    output_text: str
    tokens: List[str]
    token_ids: List[int]

    # Metadata for incremental detokenization
    prefix_offset: int
    read_offset: int

    # Parameters for detokenization
    skip_special_tokens: bool
    spaces_between_special_tokens: bool
    output_kind: RequestOutputKind

    # Request output (Cached + updated incrementally)
    request_output: RequestOutput

    # Streaming RequestOutputs to clients in async mode.
    stream: Optional[AsyncStream] = None

    @classmethod
    def from_new_request(
        cls,
        tokenizer: AnyTokenizer,
        request: DetokenizerRequest,
        stream: Optional[AsyncStream] = None,
    ) -> "DetokenizerRequestState":

        tokens, prefix_offset, read_offset = convert_prompt_ids_to_tokens(
            tokenizer=tokenizer,
            prompt_ids=request.prompt_token_ids,
            skip_special_tokens=request.skip_special_tokens,
        )

        request_output = cls._initialize_request_output(
            request.request_id,
            request.prompt,
            request.prompt_token_ids,
        )

        return cls(
            output_text="",
            tokens=tokens,
            # Detokenizer mutates this list, so need a unique copy.
            token_ids=request.prompt_token_ids.copy(),
            prefix_offset=prefix_offset,
            read_offset=read_offset,
            skip_special_tokens=request.skip_special_tokens,
            spaces_between_special_tokens=request.
            spaces_between_special_tokens,
            output_kind=request.output_kind,
            request_output=request_output,
            stream=stream,
        )

    @staticmethod
    def _initialize_request_output(
            request_id: str, prompt: str,
            prompt_token_ids: List[int]) -> RequestOutput:
        """Initialize a new RequestOutput object."""

        # TODO: Support `n` > 1.
        completion_output = CompletionOutput(
            index=0,
            text="",
            token_ids=[],
            cumulative_logprob=None,
            logprobs=None,  # TODO
            finish_reason=None,
            stop_reason=None,
            lora_request=None,
        )

        return RequestOutput(
            request_id=request_id,
            prompt=prompt,
            prompt_token_ids=prompt_token_ids,
            prompt_logprobs=None,  # TODO
            outputs=[completion_output],
            finished=False,
            metrics=None,
            lora_request=None,
            encoder_prompt=None,
            encoder_prompt_token_ids=None,
        )


class Detokenizer:

    def __init__(self, tokenizer_name: str, stream_mode: bool = False):
        self.tokenizer = get_tokenizer(tokenizer_name)
        self.stream_mode = stream_mode

        # Request id -> DetokenizerRequestState
        self.request_states: Dict[str, DetokenizerRequestState] = {}

    def is_request_active(self, request_id: str):
        return request_id in self.request_states

    def get_num_unfinished_requests(self):
        return len(self.request_states)

    def has_unfinished_requests(self) -> bool:
        return len(self.request_states) > 0

    def add_request(
        self,
        request: DetokenizerRequest,
        stream: Optional[AsyncStream] = None,
    ):
        """Add new request to the Detokenizer."""

        assert (request.request_id not in self.request_states)
        assert ((self.stream_mode and stream is not None)
                or (not self.stream_mode and stream is None))

        request_state = DetokenizerRequestState.from_new_request(
            self.tokenizer, request, stream)
        self.request_states[request.request_id] = request_state

    def step(
            self, encore_core_outputs: List[EngineCoreOutput]
    ) -> List[RequestOutput]:
        """Update state and request the RequestOutputs to the LLMEngine."""

        assert not self.stream_mode

        request_outputs: List[RequestOutput] = []
        for engine_core_output in encore_core_outputs:
            request_id = engine_core_output.request_id

            # Detokenize and update state.
            request_output = self._update_request_state(
                tokenizer=self.tokenizer,
                request_state=self.request_states[request_id],
                new_token_ids=engine_core_output.new_token_ids,
                finished=engine_core_output.finished,
                finish_reason=engine_core_output.finish_reason,
                stop_reason=engine_core_output.stop_reason,
            )

            # Add to RequestOutputs list.
            request_outputs.append(request_output)

            # Free completed requests.
            if engine_core_output.finished:
                self.request_states.pop(request_id)

        # Return to EngineClient.
        return request_outputs

    def step_streaming(self,
                       encore_core_outputs: List[EngineCoreOutput]) -> None:
        """Update state and put the RequestOutput in the per request queues."""

        assert self.stream_mode

        for engine_core_output in encore_core_outputs:
            request_id = engine_core_output.request_id

            # Detokenize and update state.
            request_output = self._update_request_state(
                tokenizer=self.tokenizer,
                request_state=self.request_states[request_id],
                new_token_ids=engine_core_output.new_token_ids,
                finished=engine_core_output.finished,
                finish_reason=engine_core_output.finish_reason,
                stop_reason=engine_core_output.stop_reason,
            )

            # Send the RequestOutput to the per client output queue.
            assert self.request_states[request_id].stream is not None
            self.request_states[request_id].stream.put(request_output)
            # TODO: is caching RequestOutput sound?
            # What happens if the reader from the stream falls behind?
            # Won't the object in the queue get mutated?

            # Free completed requests.
            if engine_core_output.finished:
                self.request_states[request_id].stream.finish()
                self.request_states.pop(request_id)
                logger.debug("Finished request %s.", request_id)

    @staticmethod
    def _update_request_state(
        tokenizer: AnyTokenizer,
        request_state: DetokenizerRequestState,
        new_token_ids: List[int],
        finished: bool,
        finish_reason: Optional[str],
        stop_reason: Optional[str],
    ) -> RequestOutput:
        """
        Update RequestState for the request_id by:
            1) Detokenize the new token ids incrementally.
            2) Update the RequestOutput with the new text.
        """

        # 1) Detokenize the new token ids incrementally.
        # TODO(woosuk): This method becomes very inefficient when the number of
        # new_token_ids is more than 1. We need to optimize this.
        decoded_text = ""
        for new_token_id in new_token_ids:
            request_state.token_ids.append(new_token_id)
            (new_tokens, new_decoded_token_text, prefix_offset,
             read_offset) = detokenize_incrementally(
                 tokenizer=tokenizer,
                 all_input_ids=request_state.token_ids,
                 prev_tokens=request_state.tokens,
                 prefix_offset=request_state.prefix_offset,
                 read_offset=request_state.read_offset,
                 skip_special_tokens=request_state.skip_special_tokens,
                 spaces_between_special_tokens=request_state.
                 spaces_between_special_tokens,
             )

            request_state.tokens.extend(new_tokens)
            request_state.prefix_offset = prefix_offset
            request_state.read_offset = read_offset
            request_state.output_text += new_decoded_token_text

            decoded_text += new_decoded_token_text

        # 2) Update the RequestOutput object with the new text.
        request_output = request_state.request_output
        completion_output = request_output.outputs[0]
        if request_state.output_kind == RequestOutputKind.CUMULATIVE:
            completion_output.text += decoded_text
            completion_output.token_ids = request_state.token_ids
        elif request_state.output_kind == RequestOutputKind.DELTA:
            completion_output.text = decoded_text
            num_prev_tokens = len(completion_output.token_ids)
            completion_output.token_ids = request_state.token_ids[
                num_prev_tokens:]
        elif request_state.output_kind == RequestOutputKind.FINAL_ONLY:
            if finished:
                completion_output.text = request_state.output_text
                completion_output.token_ids = request_state.token_ids
            else:
                completion_output.text = ""
                completion_output.token_ids = []

        if finished:
            completion_output.finish_reason = finish_reason
            completion_output.stop_reason = stop_reason
            request_output.finished = finished

        return request_output
