from dataclasses import dataclass
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
    MEAN = 3


@dataclass
class PoolingConfig:
    """A class that configures the pooling operation which
      only applies to sentence-transformers models. 
      More at: https://www.sbert.net/

    Attributes:
        pooling_type (str): The type of pooling to use. 
        normalize (bool): Whether to normalize the pooled data.

    Methods:
        get_pooling_type(pooling_type_name): Returns the pooling 
        type enum value corresponding to the given string.
    """

    def __init__(self, pooling_type: str, normalize: bool):
        self.pooling_type = self.get_pooling_type(pooling_type)
        self.normalize = normalize

    def get_pooling_type(self, pooling_type_name: str) -> PoolingType:
        pooling_types = PoolingType.__dict__.items()
        return PoolingType(
            next((value for key, value in pooling_types
                  if key.lower() in pooling_type_name), 2))


class Pooler(nn.Module):
    """A layer that pools specific information from hidden states.

    This layer does the following:
    1. Extracts specific tokens or aggregates data based on pooling method.
    2. Normalizes output if specified.
    3. Returns structured results as `PoolerOutput`.

    Attributes:
        pooling_type: The type of pooling to use (LAST, ALL, CLS, MEAN).
        normalize: Whether to normalize the pooled data.
    """

    def __init__(self, pooling_type: PoolingType, normalize: bool):
        super().__init__()

        self.pooling_type = pooling_type
        self.normalize = normalize

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
        elif self.pooling_type == PoolingType.MEAN:
            # Calculate mean pooling
            cumsum = torch.cumsum(hidden_states, dim=0)
            start_indices = torch.cat([
                torch.tensor([0], device=hidden_states.device),
                torch.cumsum(prompt_lens[:-1], dim=0)
            ])
            end_indices = torch.cumsum(prompt_lens, dim=0)
            pooled_data = (
                cumsum[end_indices - 1] - cumsum[start_indices] +
                hidden_states[start_indices]) / prompt_lens.unsqueeze(1)
        else:
            raise ValueError(f"Invalid pooling type: {self.pooling_type}")

        if self.normalize:
            pooled_data = nn.functional.normalize(pooled_data, p=2, dim=1)

        pooled_outputs = [
            EmbeddingSequenceGroupOutput(data.tolist()) for data in pooled_data
        ]

        return PoolerOutput(outputs=pooled_outputs)
