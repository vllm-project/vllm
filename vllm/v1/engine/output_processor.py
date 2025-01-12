from dataclasses import dataclass
from typing import Dict, List, Optional

from vllm.outputs import RequestOutput
from vllm.v1.engine import EngineCoreOutputs
from vllm.v1.engine.detokenizer import DetokenizerOutput
from vllm.v1.engine.request_state import RequestState
from vllm.v1.metrics.stats import IterationStats


@dataclass
class OutputProcessorOutput:
    """Output of the OutputProcessor.step() function."""

    request_outputs: List[RequestOutput]
    reqs_to_abort: List[str]
    iteration_stats: IterationStats


class OutputProcessor:

    def __init__(
        self,
        request_states: Dict[str, RequestState],
        log_stats: bool,
    ):
        self.request_states = request_states
        self.log_stats = log_stats

    def make_request_output(
        self,
        request_state: RequestState,
        detokenizer_output: Optional[DetokenizerOutput],
    ) -> Optional[RequestOutput]:

        if detokenizer_output is None:
            return None

        request_output = RequestOutput.new(
            request_state.request_id,
            request_state.prompt,
            request_state.prompt_token_ids,
            detokenizer_output.output_text,
            detokenizer_output.token_ids,
            detokenizer_output.finished,
        )
        if detokenizer_output.finished:
            completion_output = request_output.outputs[0]
            completion_output.finish_reason = detokenizer_output.finish_reason
            completion_output.stop_reason = detokenizer_output.stop_reason

        return request_output

    def process_outputs(self,
                        outputs: EngineCoreOutputs) -> OutputProcessorOutput:
        """
        Process the EngineCoreOutputs:
        1) Compute stats for logging
        2) Detokenize
        3) Create and handle RequestOutput objects:
            * If self.stream_outputs (for usage with AsyncLLM), 
              we put RequestOutput objects into the asyncio queue
              for handling by the per-request generate() tasks.
            * If not self.stream_outputs (for usage with LLMEngine), 
              we return a list of RequestOutput objects.

        ****************** NOTE FOR DEVELOPERS ******************

        VLLM V1 minimizes the number of python loops over the full
        batch to ensure system overheads are minimized. This is the 
        only function that should loop over EngineCoreOutputs.

        If you need to touch every element of the batch, implement a
        method called XXXClass.update_from_output() to be called
        within the loop below. For examples, see:
            * IterationStats.update_from_output()
            * Detokenizer.update_from_output()
        
        **********************************************************
        """

        request_outputs: List[RequestOutput] = []
        reqs_to_abort: List[str] = []
        iteration_stats = IterationStats(self.log_stats)
        for engine_core_output in outputs.outputs:
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
            detokenizer_output = req_state.detokenizer.update_from_output(
                engine_core_output)

            # 3) Create and handle RequestOutput objects.
            if request_output := self.make_request_output(
                    req_state, detokenizer_output):
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
