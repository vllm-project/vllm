import os
from enum import IntEnum

import torch
import torch.nn as nn

from vllm.model_executor.pooling_metadata import (PoolingMetadata,
                                                  PoolingTensors)
from vllm.sequence import EmbeddingSequenceGroupOutput, PoolerOutput


class PoolingType(IntEnum):
    """Enumeration for different types of pooling methods."""
    LAST = 0
    ALL = 1
    CLS = 2
    STEP = 3


class Pooler(nn.Module):
    """A layer that pools specific information from hidden states.

    This layer does the following:
    1. Extracts specific tokens or aggregates data based on pooling method.
    2. Normalizes output if specified.
    3. Returns structured results as `PoolerOutput`.

    Attributes:
        pooling_type: The type of pooling to use (LAST, ALL, CLS).
        normalize: Whether to normalize the pooled data.
    """

    def __init__(self,
                 pooling_type: PoolingType,
                 normalize: bool,
                 softmax: bool = False):
        super().__init__()

        self.pooling_type = pooling_type
        self.normalize = normalize
        self.softmax = softmax
        returned_token_ids = os.environ.get('RETURNED_TOKEN_IDS', '648,387')
        self.returned_token_ids = list(map(int, returned_token_ids.split(",")))
        self.step_tag_id = int(os.environ.get('STEP_TOKEN_ID', -1))

    def forward(
        self,
        hidden_states: torch.Tensor,
        pooling_metadata: PoolingMetadata,
    ) -> PoolerOutput:
        """Pools specific information from hidden states based on metadata."""

        prompt_lens = PoolingTensors.from_pooling_metadata(
            pooling_metadata, hidden_states.device).prompt_lens

        if self.pooling_type is PoolingType.CLS:
            first_token_flat_indices = torch.zeros_like(prompt_lens)
            first_token_flat_indices[1:] += torch.cumsum(prompt_lens,
                                                         dim=0)[:-1]
            pooled_data = hidden_states[first_token_flat_indices]
        elif self.pooling_type == PoolingType.LAST:
            last_token_flat_indices = torch.cumsum(prompt_lens, dim=0) - 1
            pooled_data = hidden_states[last_token_flat_indices]
        elif self.pooling_type == PoolingType.ALL:
            offset = 0
            pooled_data = []
            for prompt_len in prompt_lens:
                pooled_data.append(hidden_states[offset:offset + prompt_len])
                offset += prompt_len
        elif self.pooling_type == PoolingType.STEP:
            logits = hidden_states[:, self.returned_token_ids].softmax(dim=-1)
            offset = 0
            pooled_data = []
            for prompt_len, seq_data_i in zip(
                    prompt_lens, pooling_metadata.seq_data.values()):
                if self.step_tag_id == -1:
                    pooled_data.append(logits[offset:offset + prompt_len])
                else:
                    step_idxs = torch.tensor(
                        seq_data_i.prompt_token_ids) == self.step_tag_id
                    pooled_data.append(logits[offset:offset +
                                              prompt_len][step_idxs])
                offset += prompt_len
        else:
            raise ValueError(f"Invalid pooling type: {self.pooling_type}")

        if self.normalize:
            pooled_data = nn.functional.normalize(pooled_data, p=2, dim=1)

        if self.softmax:
            pooled_data = nn.functional.softmax(pooled_data, dim=-1)

        pooled_outputs = [
            EmbeddingSequenceGroupOutput(data.tolist()) for data in pooled_data
        ]

        return PoolerOutput(outputs=pooled_outputs)
