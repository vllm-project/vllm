from typing import List

import torch

from vllm.wde.core.schema.engine_io import RequestOutput


class EmbeddingRequestOutput(RequestOutput):

    def __init__(self, request_id: str, outputs: torch.Tensor,
                 prompt_token_ids: List[int], finished: bool):
        self.request_id = request_id
        self.prompt_token_ids = prompt_token_ids
        self.finished = finished
        self.outputs = outputs

    def __repr__(self):
        """
        Returns a string representation of an EmbeddingRequestOutput instance.

        The representation includes the request_id and the number of outputs,
        providing a quick overview of the embedding request's results.

        Returns:
            str: A string representation of the EmbeddingRequestOutput instance.
        """
        return (f"EmbeddingRequestOutput(request_id='{self.request_id}', "
                f"outputs={repr(self.outputs)}, "
                f"prompt_token_ids={self.prompt_token_ids}, "
                f"finished={self.finished})")
