import asyncio
from dataclasses import dataclass
from typing import Dict, List, Optional

from vllm.outputs import RequestOutput
from vllm.transformers_utils.tokenizer_group import BaseTokenizerGroup
from vllm.v1.engine import EngineCoreOutput, EngineCoreRequest
from vllm.v1.engine.detokenizer import DetokenizerOutput
from vllm.v1.engine.logprobs import LogprobsOutput
from vllm.v1.engine.output_processor_utils import RequestState
from vllm.v1.metrics.stats import IterationStats


@dataclass
class OutputProcessorOutput:

    request_outputs: List[RequestOutput]
    reqs_to_abort: List[str]
    iteration_stats: IterationStats


class OutputProcessor:
    """Process EngineCoreOutputs into RequestOutputs."""

    def __init__(
        self,
        tokenizer: BaseTokenizerGroup,
        log_stats: bool,
    ):
        self.log_stats = log_stats
        self.tokenizer = tokenizer
        self.request_states: Dict[str, RequestState] = {}

    def is_request_active(self, request_id: str) -> bool:
        return request_id in self.request_states

    def get_num_unfinished_requests(self):
        return len(self.request_states)

    def has_unfinished_requests(self) -> bool:
        return len(self.request_states) > 0

    def abort_requests(
        self,
        request_ids: List[str],
    ) -> None:
        for request_id in request_ids:
            self.request_states.pop(request_id, None)

    def add_request(
        self,
        request: EngineCoreRequest,
        queue: Optional[asyncio.Queue[RequestOutput]] = None,
    ) -> None:
        request_id = request.request_id
        if request_id in self.request_states:
            raise ValueError(f"Request id {request_id} already running.")

        self.request_states[request_id] = RequestState.from_new_request(
            tokenizer=self.tokenizer.get_lora_tokenizer(request.lora_request),
            request=request,
            queue=queue)

    def process_outputs(
        self,
        engine_core_outputs: List[EngineCoreOutput],
    ) -> OutputProcessorOutput:
        """
        Process the EngineCoreOutputs:
        1) Compute stats for logging
        2) Detokenize
        3) Create and handle RequestOutput objects:
            * If there is a queue (for usage with AsyncLLM), 
              put the RequestOutput objects into the queue for
              handling by the per-request generate() tasks.

            * If there is no queue (for usage with LLMEngine), 
              return a list of RequestOutput objects.

        ****************** NOTE FOR DEVELOPERS ******************

        VLLM V1 minimizes the number of python loops over the full
        batch to ensure system overheads are minimized. This is the 
        only function that should loop over EngineCoreOutputs.

        If you need to touch every element of the batch, implement a
        method called XXXClass.update_from_output() to be called
        within the loop below. For examples, see:
            * IterationStats.update_from_output()
            * Detokenizer.update_from_output()
        
        TODO(rob): add Protocol makes update_from_output explicit.
        
        **********************************************************
        """

        request_outputs: List[RequestOutput] = []
        reqs_to_abort: List[str] = []
        iteration_stats = IterationStats(self.log_stats)
        for engine_core_output in engine_core_outputs:
            req_id = engine_core_output.request_id
            req_state = self.request_states.get(req_id)
            if req_state is None:
                # Ignore output for already-aborted request.
                continue

            # 1) Compute stats for this iteration.
            iteration_stats.update_from_output(engine_core_output,
                                               req_state.is_prefilling,
                                               req_state.prompt_len)
            req_state.is_prefilling = False

            # 2) Detokenize the token ids into text.
            if detokenizer_output := req_state.detokenizer.update_from_output(
                    engine_core_output):
                # Detect if detokenizer updated `finish_reason`
                engine_core_output.finish_reason = (
                    detokenizer_output.finish_reason)

            # 3) Compute sample and prompt logprobs for request,
            #    if required.
            logprobs_output = req_state.logprobs_processor.update_from_output(
                engine_core_output)

            # 4) Create and handle RequestOutput objects.
            if request_output := self._make_request_output(
                    req_state, logprobs_output, detokenizer_output):
                if req_state.queue is not None:
                    # AsyncLLM: put into queue for handling by generate().
                    req_state.queue.put_nowait(request_output)
                else:
                    # LLMEngine: return list of RequestOutputs.
                    request_outputs.append(request_output)

                # Free completed requests.
                if request_output.finished:
                    self.request_states.pop(req_id)
                    if not engine_core_output.finished:
                        # If req not finished in EngineCore, but Detokenizer
                        # detected stop string, abort needed in EngineCore.
                        reqs_to_abort.append(req_id)

        return OutputProcessorOutput(
            request_outputs=request_outputs,
            reqs_to_abort=reqs_to_abort,
            iteration_stats=iteration_stats,
        )

    def _make_request_output(
        self,
        request_state: RequestState,
        logprobs_output: Optional[LogprobsOutput],
        detokenizer_output: Optional[DetokenizerOutput],
    ) -> Optional[RequestOutput]:

        if detokenizer_output is None:
            # Only happens with FINAL request output kind when
            # we are not on the final step
            return None
        assert logprobs_output is not None

        request_output = RequestOutput.new(
            request_state.request_id,
            request_state.prompt,
            request_state.prompt_token_ids,
            logprobs_output.logprobs,
            logprobs_output.prompt_logprobs,
            logprobs_output.cumulative_logprob,
            detokenizer_output.output_text,
            detokenizer_output.token_ids,
            detokenizer_output.finished,
        )
        if detokenizer_output.finished:
            completion_output = request_output.outputs[0]
            completion_output.finish_reason = detokenizer_output.finish_reason
            completion_output.stop_reason = detokenizer_output.stop_reason

        return request_output
