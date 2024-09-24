from typing import List

import torch

from vllm.wde.core.llm_engine import LLMEngine
from vllm.wde.core.processor.output_processor import OutputProcessor
from vllm.wde.prefill_only.schema.engine_io import (PrefillOnlyRequestOutput,
                                                    PrefillOnlySchedulerOutput)


class DecodeOnlyHiddenStatesOutputProcessor(OutputProcessor):

    def __init__(self):
        pass

    @classmethod
    def from_engine(cls, engine: LLMEngine):
        return cls()

    def __call__(
            self, scheduler_output: PrefillOnlySchedulerOutput,
            execute_output: torch.Tensor) -> List[PrefillOnlyRequestOutput]:

        request_outputs = []
        offset = 0
        for request in scheduler_output.requests:
            prompt_token_ids = request.inputs.prompt_token_ids
            n_tokens = len(prompt_token_ids)
            request_outputs.append(
                PrefillOnlyRequestOutput(
                    request_id=request.request_id,
                    prompt_token_ids=prompt_token_ids,
                    finished=True,
                    # last pooling
                    outputs=execute_output[offset + n_tokens - 1]))
            offset += n_tokens
        return request_outputs
