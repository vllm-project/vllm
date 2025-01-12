from typing import List, Optional

import asyncio

from vllm.outputs import RequestOutput
from vllm.transformers_utils.detokenizer_utils import AnyTokenizer
from vllm.v1.engine import EngineCoreRequest
from vllm.v1.engine.detokenizer import Detokenizer


class RequestState:
    """RequestState for AsyncLLM and LLMEngine."""

    def __init__(
        self,
        request_id: str,
        prompt: Optional[str],
        prompt_token_ids: List[int],
        detokenizer: Detokenizer,
        queue: Optional[asyncio.Queue[RequestOutput]],
    ):
        self.request_id = request_id
        self.prompt = prompt
        self.prompt_token_ids = prompt_token_ids
        self.prompt_len = len(prompt_token_ids)
        self.detokenizer = detokenizer
        self.is_prefilling = True
        self.queue = queue

    @classmethod
    def from_new_request(
        cls,
        tokenizer: AnyTokenizer,
        request: EngineCoreRequest,
        queue: Optional[asyncio.Queue[RequestOutput]] = None,
    ) -> "RequestState":
        return cls(
            request_id=request.request_id,
            prompt=request.prompt,
            prompt_token_ids=request.prompt_token_ids,
            detokenizer=Detokenizer.from_new_request(
                tokenizer=tokenizer,
                request=request,
            ),
            queue=queue,
        )
