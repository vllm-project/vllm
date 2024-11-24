from enum import IntEnum
from typing import List, Optional

import torch
import torch.nn as nn

from vllm.config import PoolerConfig
from vllm.model_executor.pooling_metadata import (PoolingMetadata,
                                                  PoolingTensors)
from vllm.sequence import EmbeddingSequenceGroupOutput, PoolerOutput


class PoolingType(IntEnum):
    """Enumeration for different types of pooling methods."""
    LAST = 0
    ALL = 1
    CLS = 2
    STEP = 3
    MEAN = 4


class Pooler(nn.Module):
    """A layer that pools specific information from hidden states.

    This layer does the following:
    1. Extracts specific tokens or aggregates data based on pooling method.
    2. Normalizes output if specified.
    3. Returns structured results as `PoolerOutput`.

    Attributes:
        pooling_type: The type of pooling to use.
        normalize: Whether to normalize the pooled data.
    """

    def __init__(
        self,
        pooling_type: PoolingType,
        normalize: bool,
        softmax: bool,
        step_tag_id: Optional[int] = None,
        returned_token_ids: Optional[List[int]] = None,
    ):
        super().__init__()

        self.pooling_type = pooling_type
        self.normalize = normalize
        self.softmax = softmax
        self.step_tag_id = step_tag_id
        self.returned_token_ids = returned_token_ids

    @classmethod
    def from_config_with_defaults(
        cls,
        pooler_config: PoolerConfig,
        pooling_type: PoolingType,
        normalize: bool,
        softmax: bool,
        step_tag_id: Optional[int] = None,
        returned_token_ids: Optional[List[int]] = None,
    ) -> Optional["Pooler"]:
        if pooler_config is None:
            return None
        return cls(
            pooling_type=PoolingType[pooler_config.pooling_type]
            if pooler_config.pooling_type is not None else pooling_type,
            normalize=pooler_config.normalize
            if pooler_config.normalize is not None else normalize,
            softmax=pooler_config.softmax
            if pooler_config.softmax is not None else softmax,
            step_tag_id=pooler_config.step_tag_id
            if pooler_config.step_tag_id is not None else step_tag_id,
            returned_token_ids=pooler_config.returned_token_ids
            if pooler_config.returned_token_ids is not None else
            returned_token_ids,
        )

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
        elif self.pooling_type == PoolingType.STEP:
            returned_token_ids = self.returned_token_ids
            if returned_token_ids is not None and len(returned_token_ids) > 0:
                hidden_states = hidden_states[:, returned_token_ids]

            step_tag_id = self.step_tag_id

            offset = 0
            pooled_data = []
            for prompt_len, seq_data_i in zip(
                    prompt_lens, pooling_metadata.seq_data.values()):
                pooled_data_i = hidden_states[offset:offset + prompt_len]
                if step_tag_id is not None:
                    token_ids = torch.tensor(seq_data_i.prompt_token_ids)
                    pooled_data_i = pooled_data_i[token_ids == step_tag_id]

                offset += prompt_len
                pooled_data.append(pooled_data_i)
        else:
            raise ValueError(f"Invalid pooling type: {self.pooling_type}")

        if self.normalize:
            if isinstance(pooled_data, list):
                pooled_data = [
                    nn.functional.normalize(data, p=2, dim=1)
                    for data in pooled_data
                ]
            else:
                pooled_data = nn.functional.normalize(pooled_data, p=2, dim=1)

        if self.softmax:
            if isinstance(pooled_data, list):
                pooled_data = [
                    nn.functional.softmax(data, dim=-1) for data in pooled_data
                ]
            else:
                pooled_data = nn.functional.softmax(pooled_data, dim=-1)

        pooled_outputs = [
            EmbeddingSequenceGroupOutput(data.tolist()) for data in pooled_data
        ]

        return PoolerOutput(outputs=pooled_outputs)
