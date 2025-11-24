# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.v1.engine import EngineCoreOutput, EngineCoreRequest
from vllm.v1.engine.output_processor import (
    OutputProcessor,
    RequestOutputCollector,
    RequestState,
)
from vllm.v1.engine.parallel_sampling import ParentRequest


class StreamingOutputProcessor(OutputProcessor):
    def add_request(
        self,
        request: EngineCoreRequest,
        prompt: str | None,
        parent_req: ParentRequest | None = None,
        request_index: int = 0,
        queue: RequestOutputCollector | None = None,
    ) -> None:
        request_id = request.request_id
        if request_id in self.request_states:
            req_state = self.request_states[request_id]
            if req_state.prompt and prompt:
                req_state.prompt += prompt
            req_state.prompt_token_ids.extend(request.prompt_token_ids)
            req_state.prompt_embeds = request.prompt_embeds
            if req_state.stats is not None:
                req_state.stats.arrival_time = request.arrival_time
            req_state.is_prefilling = True
        else:
            req_state = RequestState.from_new_request(
                tokenizer=self.tokenizer,
                request=request,
                prompt=prompt,
                parent_req=parent_req,
                request_index=request_index,
                queue=queue,
                log_stats=self.log_stats,
            )
            self.request_states[request_id] = req_state
            self.lora_states.add_request(req_state)
            if parent_req:
                self.parent_requests[parent_req.request_id] = parent_req

    def _is_finished(self, engine_core_output: EngineCoreOutput) -> bool:
        return (
            engine_core_output.finish_reason is not None
            and engine_core_output.close_session
        )
