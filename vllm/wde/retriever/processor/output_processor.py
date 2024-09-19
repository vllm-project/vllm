from typing import List

import torch

from vllm.wde.encode_only.processor.output_processor import (
    EncodeOnlyModelOutputProcessor)
from vllm.wde.encode_only.schema.engine_io import EncodeOnlySchedulerOutput
from vllm.wde.retriever.schema.engine_io import EmbeddingRequestOutput


class RetrieverModelOutputProcessor(EncodeOnlyModelOutputProcessor):

    def __call__(self, scheduler_output: EncodeOnlySchedulerOutput,
                 execute_output: torch.Tensor) -> List[EmbeddingRequestOutput]:
        request_outputs = []
        for request, outputs in zip(scheduler_output.requests, execute_output):
            prompt_token_ids = request.inputs.prompt_token_ids
            request_outputs.append(
                EmbeddingRequestOutput(request_id=request.request_id,
                                       prompt_token_ids=prompt_token_ids,
                                       finished=True,
                                       outputs=outputs))
        return request_outputs
