# SPDX-License-Identifier: Apache-2.0

import asyncio
from dataclasses import dataclass
from typing import Dict, List, Optional

from vllm.outputs import RequestOutput
from vllm.sampling_params import RequestOutputKind
from vllm.transformers_utils.tokenizer import AnyTokenizer
from vllm.transformers_utils.tokenizer_group import BaseTokenizerGroup
from vllm.v1.engine import EngineCoreOutput, EngineCoreRequest, FinishReason
from vllm.v1.engine.detokenizer import IncrementalDetokenizer
from vllm.v1.engine.logprobs import LogprobsProcessor
from vllm.v1.metrics.stats import IterationStats, RequestStateStats


@dataclass
class OutputProcessorOutput:

    request_outputs: List[RequestOutput]
    reqs_to_abort: List[str]
    iteration_stats: IterationStats


class RequestState:

    def __init__(
        self,
        request_id: str,
        output_kind: RequestOutputKind,
        prompt: Optional[str],
        prompt_token_ids: List[int],
        logprobs_processor: LogprobsProcessor,
        detokenizer: IncrementalDetokenizer,
        arrival_time: float,
        queue: Optional[asyncio.Queue[RequestOutput]],
    ):
        self.request_id = request_id
        self.output_kind = output_kind
        self.prompt = prompt
        self.prompt_token_ids = prompt_token_ids
        self.prompt_len = len(prompt_token_ids)
        self.logprobs_processor = logprobs_processor
        self.detokenizer = detokenizer
        self.is_prefilling = True
        self.queue = queue

        self.stats = RequestStateStats(last_token_time=arrival_time)

    @classmethod
    def from_new_request(
        cls,
        tokenizer: AnyTokenizer,
        request: EngineCoreRequest,
        queue: Optional[asyncio.Queue[RequestOutput]] = None,
    ) -> "RequestState":
        return cls(
            request_id=request.request_id,
            output_kind=request.sampling_params.output_kind,
            prompt=request.prompt,
            prompt_token_ids=request.prompt_token_ids,
            logprobs_processor=LogprobsProcessor.from_new_request(
                tokenizer=tokenizer,
                request=request,
            ),
            detokenizer=IncrementalDetokenizer.from_new_request(
                tokenizer=tokenizer,
                request=request,
            ),
            arrival_time=request.arrival_time,
            queue=queue,
        )


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
        iteration_stats: Optional[IterationStats] = None,
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

        If you need to touch every element of the batch, do it from
        within the loop below.
        
        **********************************************************
        """

        request_outputs: List[RequestOutput] = []
        reqs_to_abort: List[str] = []
        if not iteration_stats:
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
                                               req_state.prompt_len,
                                               req_state.stats)

            new_token_ids = engine_core_output.new_token_ids
            finish_reason = engine_core_output.finish_reason

            # TODO(andy): prompt logprobs + chunked prefill can
            # result in engine core returning an output for a
            # partial prefill (in order to send back partial
            # prompt logprobs.) This breaks the invariant that
            # process_outputs is only operating on engine core
            # outputs associated with non-partial completions.
            # Currently this is handled by having `is_prefilling`
            # check for new decoded tokens, indicating that
            # the completion is not partial.
            #
            # Follow up will aggregate partial prompt logprobs
            # in the EngineCore.
            req_state.is_prefilling = not new_token_ids

            # 2) Detokenize the token ids into text and check for stop
            #    strings.
            stop_reason = req_state.detokenizer.update(new_token_ids)
            if stop_reason:
                finish_reason = FinishReason.STOP

            # 3) Compute sample and prompt logprobs for request,
            #    if required.
            req_state.logprobs_processor.update_from_output(engine_core_output)

            # 4) Create and handle RequestOutput objects.
            if request_output := self._make_request_output(
                    req_state, new_token_ids, finish_reason, stop_reason):
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

                    # Track per-request stats.
                    assert finish_reason is not None
                    iteration_stats.update_from_finished_request(
                        finish_reason, request_output, req_state.stats)

        return OutputProcessorOutput(
            request_outputs=request_outputs,
            reqs_to_abort=reqs_to_abort,
            iteration_stats=iteration_stats,
        )

    @staticmethod
    def _make_request_output(
        request_state: RequestState,
        new_token_ids: List[int],
        finish_reason: Optional[FinishReason],
        stop_reason: Optional[str],
    ) -> Optional[RequestOutput]:

        finished = finish_reason is not None
        output_kind = request_state.output_kind
        # In follow up, we will switch to invariant where EngineCore
        # does not stream partial prefills.
        if not finished and (request_state.is_prefilling
                             or output_kind == RequestOutputKind.FINAL_ONLY):
            # Only the final output is required in FINAL_ONLY mode.
            return None

        detokenizer = request_state.detokenizer
        logprobs_processor = request_state.logprobs_processor

        delta = output_kind == RequestOutputKind.DELTA
        logprobs = logprobs_processor.logprobs
        if delta:
            if logprobs:
                logprobs = logprobs[-len(new_token_ids):]
            # Side effect: logprobs processor forgets prompt logprobs
            prompt_logprobs = logprobs_processor.pop_prompt_logprobs()
        else:
            prompt_logprobs = logprobs_processor.prompt_logprobs

        request_output = RequestOutput.new(
            request_id=request_state.request_id,
            prompt=request_state.prompt,
            prompt_token_ids=request_state.prompt_token_ids,
            text=detokenizer.get_next_output_text(finished, delta),
            token_ids=new_token_ids if delta else detokenizer.output_token_ids,
            logprobs=logprobs,
            prompt_logprobs=prompt_logprobs,
            cumulative_logprob=logprobs_processor.cumulative_logprob,
            finished=finished,
        )
        if finished:
            completion_output = request_output.outputs[0]
            completion_output.finish_reason = str(finish_reason)
            completion_output.stop_reason = stop_reason

        return request_output
