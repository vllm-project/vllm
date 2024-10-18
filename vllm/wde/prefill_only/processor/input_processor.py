import time
from typing import Optional

from vllm.wde.core.inputs.tokenizer import Tokenizer
from vllm.wde.core.llm_engine import LLMEngine
from vllm.wde.core.processor.input_processor import (InputProcessor,
                                                     RequestProcessor)
from vllm.wde.core.schema.engine_io import (Params, PromptInput, TextPrompt,
                                            TokensPrompt)
from vllm.wde.prefill_only.schema.engine_io import (
    PrefillOnlyInput, PrefillOnlyRequest, PrefillOnlySchedulableRequest)


class PrefillOnlyModelInputProcessor(InputProcessor):

    @classmethod
    def from_engine(cls, engine: LLMEngine):
        return cls()

    def __call__(self,
                 request_id: str,
                 inputs: Optional[PromptInput] = None,
                 params: Optional[Params] = None,
                 arrival_time: Optional[float] = None) -> PrefillOnlyRequest:
        if not arrival_time:
            arrival_time = time.time()
        request = PrefillOnlyRequest(request_id=str(request_id),
                                     inputs=inputs,
                                     arrival_time=arrival_time)
        return request


class PrefillOnlyModelRequestProcessor(RequestProcessor):

    def __init__(self, tokenizer: Tokenizer):
        self.tokenizer = tokenizer

    @classmethod
    def from_engine(cls, engine: LLMEngine):
        return cls(engine.tokenizer)

    def __call__(self,
                 request: PrefillOnlyRequest) -> PrefillOnlySchedulableRequest:
        inputs = request.inputs

        if isinstance(inputs, str):
            inputs = {"prompt": inputs}
        elif isinstance(input, TextPrompt):
            inputs = {"prompt": inputs.prompt}
        elif isinstance(input, TokensPrompt):
            inputs = {"prompt_token_ids", inputs.prompt_token_ids}

        if "prompt_token_ids" not in inputs:
            tokenizer = self.tokenizer

            prompt_token_ids = tokenizer.encode(inputs["prompt"])
        else:
            prompt_token_ids = inputs["prompt_token_ids"]

        schedulable_request = PrefillOnlySchedulableRequest(
            request_id=request.request_id,
            inputs=PrefillOnlyInput(prompt_token_ids=prompt_token_ids,
                                    prompt=inputs.get("prompt")),
            arrival_time=request.arrival_time)

        return schedulable_request
