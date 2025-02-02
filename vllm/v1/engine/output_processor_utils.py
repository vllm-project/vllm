# SPDX-License-Identifier: Apache-2.0

import asyncio
from typing import List, Optional

from vllm.outputs import RequestOutput
from vllm.sampling_params import RequestOutputKind
from vllm.transformers_utils.detokenizer_utils import AnyTokenizer
from vllm.v1.engine import EngineCoreRequest
from vllm.v1.engine.detokenizer import IncrementalDetokenizer
from vllm.v1.engine.logprobs import LogprobsProcessor
from vllm.v1.metrics.stats import RequestStateStats


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
