from typing import List, Sequence, Union

import torch

from vllm.wde.core.llm_engine import LLMEngine
from vllm.wde.core.processor.output_processor import OutputProcessor
from vllm.wde.core.schema.engine_io import ValidationError
from vllm.wde.prefill_only.schema.engine_io import (PrefillOnlyRequestOutput,
                                                    PrefillOnlySchedulerOutput)


class PrefillOnlyModelOutputProcessor(OutputProcessor):

    def __init__(self):
        pass

    @classmethod
    def from_engine(cls, engine: LLMEngine):
        return cls()

    def __call__(
        self, scheduler_output: PrefillOnlySchedulerOutput,
        execute_output: Union[torch.Tensor, Sequence[torch.Tensor]]
    ) -> List[PrefillOnlyRequestOutput]:
        if isinstance(execute_output, torch.Tensor):
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
                        outputs=execute_output[offset:offset + n_tokens]))
                offset += n_tokens
            return request_outputs
        elif isinstance(execute_output, (list, tuple)):
            sequence_output, pooled_output = execute_output
            request_outputs = []
            for request, outputs in zip(scheduler_output.requests,
                                        pooled_output):
                prompt_token_ids = request.inputs.prompt_token_ids
                request_outputs.append(
                    PrefillOnlyRequestOutput(request_id=request.request_id,
                                             prompt_token_ids=prompt_token_ids,
                                             finished=True,
                                             outputs=outputs))
            return request_outputs
        else:
            raise ValidationError("Output format not supported")
