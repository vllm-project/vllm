# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from enum import IntEnum
from typing import Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing_extensions import assert_never

from vllm.config import ModelConfig, PoolerConfig
from vllm.model_executor.pooling_metadata import (  # noqa: E501
    PoolingMetadata as V0PoolingMetadata)
from vllm.model_executor.pooling_metadata import PoolingTensors
from vllm.sequence import PoolerOutput, PoolingSequenceGroupOutput
from vllm.transformers_utils.config import (
    get_cross_encoder_activation_function)
from vllm.v1.pool.metadata import PoolingMetadata as V1PoolingMetadata

PoolingMetadata = Union[V0PoolingMetadata, V1PoolingMetadata]


class PoolingType(IntEnum):
    """Enumeration for different types of pooling methods."""
    LAST = 0
    ALL = 1
    CLS = 2
    STEP = 3
    MEAN = 4


class SimplePooler(nn.Module):
    """A layer that pools specific information from hidden states.

    This layer does the following:
    1. Extracts specific tokens or aggregates data based on pooling method.
    2. Normalizes output if specified.
    3. Returns structured results as `PoolerOutput`.

    Attributes:
        pooling_type: The type of pooling to use.
        normalize: Whether to normalize the pooled data.
    """

    @staticmethod
    def from_pooling_type(
        pooling_type: PoolingType,
        *,
        normalize: bool,
        softmax: bool,
        step_tag_id: Optional[int] = None,
        returned_token_ids: Optional[list[int]] = None,
    ) -> "SimplePooler":
        if pooling_type == PoolingType.LAST:
            assert step_tag_id is None and returned_token_ids is None
            return LastPool(normalize=normalize, softmax=softmax)
        if pooling_type == PoolingType.ALL:
            assert step_tag_id is None and returned_token_ids is None
            return AllPool(normalize=normalize, softmax=softmax)
        if pooling_type == PoolingType.CLS:
            assert step_tag_id is None and returned_token_ids is None
            return CLSPool(normalize=normalize, softmax=softmax)
        if pooling_type == PoolingType.MEAN:
            assert step_tag_id is None and returned_token_ids is None
            return MeanPool(normalize=normalize, softmax=softmax)
        if pooling_type == PoolingType.STEP:
            return StepPool(normalize=normalize,
                            softmax=softmax,
                            step_tag_id=step_tag_id,
                            returned_token_ids=returned_token_ids)

        assert_never(pooling_type)

    def __init__(self, *, normalize: bool, softmax: bool) -> None:
        super().__init__()

        self.head = PoolerHead(normalize=normalize, softmax=softmax)

    def get_prompt_lens(
        self,
        hidden_states: Union[torch.Tensor, list[torch.Tensor]],
        pooling_metadata: PoolingMetadata,
    ) -> torch.Tensor:
        if isinstance(pooling_metadata, V1PoolingMetadata):
            return pooling_metadata.prompt_lens
        assert isinstance(hidden_states, torch.Tensor)
        return PoolingTensors.from_pooling_metadata(
            pooling_metadata, hidden_states.device).prompt_lens

    def extract_states(
        self,
        hidden_states: Union[torch.Tensor, list[torch.Tensor]],
        pooling_metadata: PoolingMetadata,
    ) -> Union[list[torch.Tensor], torch.Tensor]:
        raise NotImplementedError

    def build_output(self, data: torch.Tensor) -> PoolingSequenceGroupOutput:
        return PoolingSequenceGroupOutput(data)

    def forward(
        self,
        hidden_states: Union[torch.Tensor, list[torch.Tensor]],
        pooling_metadata: PoolingMetadata,
    ) -> PoolerOutput:
        pooled_data = self.extract_states(hidden_states, pooling_metadata)
        pooled_data = self.head(pooled_data, pooling_metadata)
        pooled_outputs = [self.build_output(data) for data in pooled_data]
        return PoolerOutput(outputs=pooled_outputs)


class CLSPool(SimplePooler):

    def extract_states(
        self,
        hidden_states: Union[torch.Tensor, list[torch.Tensor]],
        pooling_metadata: PoolingMetadata,
    ) -> Union[list[torch.Tensor], torch.Tensor]:
        prompt_lens = self.get_prompt_lens(hidden_states, pooling_metadata)

        if isinstance(hidden_states, list):
            result = []
            for req_state, prompt_len in zip(hidden_states, prompt_lens):
                assert prompt_len == req_state.shape[0], \
                    "partial prefill not supported with CLS pooling"
                result.append(req_state[0])
            return result

        first_token_flat_indices = torch.zeros_like(prompt_lens)
        first_token_flat_indices[1:] += torch.cumsum(prompt_lens, dim=0)[:-1]
        return hidden_states[first_token_flat_indices]


class LastPool(SimplePooler):

    def extract_states(
        self,
        hidden_states: Union[torch.Tensor, list[torch.Tensor]],
        pooling_metadata: PoolingMetadata,
    ) -> Union[list[torch.Tensor], torch.Tensor]:
        if isinstance(hidden_states, list):
            return [h[-1] for h in hidden_states]

        prompt_lens = self.get_prompt_lens(hidden_states, pooling_metadata)

        last_token_flat_indices = torch.cumsum(prompt_lens, dim=0) - 1
        return hidden_states[last_token_flat_indices]


class AllPool(SimplePooler):

    def extract_states(
        self,
        hidden_states: Union[torch.Tensor, list[torch.Tensor]],
        pooling_metadata: PoolingMetadata,
    ) -> Union[list[torch.Tensor], torch.Tensor]:
        prompt_lens = self.get_prompt_lens(hidden_states, pooling_metadata)

        if isinstance(hidden_states, list):
            for req_state, prompt_len in zip(hidden_states, prompt_lens):
                assert prompt_len == req_state.shape[0], \
                    "partial prefill not supported with ALL pooling"
            return hidden_states

        offset = 0
        pooled_data = list[torch.Tensor]()
        for prompt_len in prompt_lens:
            pooled_data.append(hidden_states[offset:offset + prompt_len])
            offset += prompt_len

        return pooled_data


class MeanPool(SimplePooler):

    def extract_states(
        self,
        hidden_states: Union[torch.Tensor, list[torch.Tensor]],
        pooling_metadata: PoolingMetadata,
    ) -> Union[list[torch.Tensor], torch.Tensor]:
        prompt_lens = self.get_prompt_lens(hidden_states, pooling_metadata)

        if isinstance(hidden_states, list):
            result = []
            for req_state, prompt_len in zip(hidden_states, prompt_lens):
                assert prompt_len == req_state.shape[0], \
                    "partial prefill not supported with mean pooling"
                result.append(torch.mean(req_state, dim=0,
                                         dtype=torch.float32))
            return result

        # Use float32 for torch.cumsum in MeanPool,
        # otherwise precision will be lost significantly.
        cumsum = torch.cumsum(hidden_states, dim=0, dtype=torch.float32)

        start_indices = torch.cat([
            torch.tensor([0], device=hidden_states.device),
            torch.cumsum(prompt_lens[:-1], dim=0)
        ])
        end_indices = torch.cumsum(prompt_lens, dim=0)
        return (cumsum[end_indices - 1] - cumsum[start_indices] +
                hidden_states[start_indices]) / prompt_lens.unsqueeze(1)


class StepPool(SimplePooler):

    def __init__(
        self,
        *,
        normalize: bool,
        softmax: bool,
        step_tag_id: Optional[int] = None,
        returned_token_ids: Optional[list[int]] = None,
    ):
        super().__init__(normalize=normalize, softmax=softmax)

        self.step_tag_id = step_tag_id
        self.returned_token_ids = returned_token_ids

    def get_prompt_token_ids(
        self,
        pooling_metadata: PoolingMetadata,
    ) -> list[torch.Tensor]:
        if isinstance(pooling_metadata, V1PoolingMetadata):
            return [
                pooling_metadata.prompt_token_ids[i, :num]
                for i, num in enumerate(pooling_metadata.prompt_lens)
            ]
        return [
            torch.tensor(seq_data_i.prompt_token_ids)
            for seq_data_i in pooling_metadata.seq_data.values()
        ]

    def extract_states(
        self,
        hidden_states: Union[torch.Tensor, list[torch.Tensor]],
        pooling_metadata: PoolingMetadata,
    ) -> Union[list[torch.Tensor], torch.Tensor]:
        prompt_lens = self.get_prompt_lens(hidden_states, pooling_metadata)
        prompt_token_ids = self.get_prompt_token_ids(pooling_metadata)

        pooled_data_lst = list[torch.Tensor]()
        if isinstance(hidden_states, list):
            for req_state, prompt_len in zip(hidden_states, prompt_lens):
                assert prompt_len == req_state.shape[0], \
                    "partial prefill not supported with step pooling"
            pooled_data_lst = hidden_states
        else:
            offset = 0
            for prompt_len in prompt_lens:
                pooled_data_i = hidden_states[offset:offset + prompt_len]
                offset += prompt_len
                pooled_data_lst.append(pooled_data_i)

        pooled_data = list[torch.Tensor]()
        returned_token_ids = self.returned_token_ids
        step_tag_id = self.step_tag_id

        for data, token_id in zip(pooled_data_lst, prompt_token_ids):
            if returned_token_ids is not None and len(returned_token_ids) > 0:
                data = data[:, returned_token_ids]

            if step_tag_id is not None:
                data = data[token_id == step_tag_id]
            pooled_data.append(data)

        return pooled_data


class PoolerHead(nn.Module):

    def __init__(self, *, normalize: bool, softmax: bool) -> None:
        super().__init__()

        self.normalize = normalize
        self.softmax = softmax

    def forward(self, pooled_data: Union[list[torch.Tensor], torch.Tensor],
                pooling_metadata: PoolingMetadata):

        # Using float32 in PoolerHead
        if isinstance(pooled_data, list):
            for i in range(len(pooled_data)):
                pooled_data[i] = pooled_data[i].to(torch.float32)
        else:
            pooled_data = pooled_data.to(torch.float32)

        if isinstance(pooling_metadata, V0PoolingMetadata):
            dimensions_list = [
                pooling_param.dimensions
                for _, pooling_param in pooling_metadata.seq_groups
            ]
        else:
            assert isinstance(pooled_data, list)
            dimensions_list = [
                pooling_param.dimensions
                for pooling_param in pooling_metadata.pooling_params
            ]
        if any(d is not None for d in dimensions_list):
            # change the output dimension
            assert len(pooled_data) == len(dimensions_list)
            pooled_data = [
                vecs if d is None else vecs[..., :d]
                for vecs, d in zip(pooled_data, dimensions_list)
            ]

        if self.normalize:
            if isinstance(pooled_data, list):
                pooled_data = [
                    F.normalize(data, p=2, dim=-1) for data in pooled_data
                ]
            else:
                pooled_data = F.normalize(pooled_data, p=2, dim=-1)

        if self.softmax:
            if isinstance(pooled_data, list):
                pooled_data = [
                    F.softmax(data, dim=-1)
                    if data.shape[-1] >= 2 else F.sigmoid(data)
                    for data in pooled_data
                ]
            else:
                if pooled_data.shape[-1] >= 2:
                    pooled_data = F.softmax(pooled_data, dim=-1)
                else:
                    pooled_data = F.sigmoid(pooled_data)

        return pooled_data


class Pooler(nn.Module):

    @classmethod
    def from_config_with_defaults(
        cls,
        pooler_config: PoolerConfig,
        pooling_type: PoolingType,
        normalize: bool,
        softmax: bool,
        step_tag_id: Optional[int] = None,
        returned_token_ids: Optional[list[int]] = None,
    ) -> SimplePooler:
        return SimplePooler.from_pooling_type(
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


class ClassifierPooler(nn.Module):
    """A pooling layer for classification tasks.

    This layer does the following:
    1. Applies a classification layer to the hidden states.
    2. Optionally applies a pooler layer.
    3. Applies an activation function to the output. In the case of
       classification models it is either sigmoid or softmax. In the
       case of scoring models, the same behavior is configuration
       dependent, as in the sentence-transformers library.
    """

    def __init__(
        self,
        config: ModelConfig,
        classifier: nn.Module,
        pooler: Optional[nn.Module] = None,
    ):
        super().__init__()
        self.classifier = classifier
        self.pooler = pooler

        if config.task == "score":
            self.default_activation_function = \
                get_cross_encoder_activation_function(config.hf_config)
        elif config.task == "classify":
            self.default_activation_function = nn.Sigmoid() \
                if config.hf_config.num_labels == 1 else nn.Softmax()
        else:
            raise NotImplementedError(f"task={config.task!r} is not supported"
                                      " with the classification pooler")

    def get_prompt_lens(
        self,
        hidden_states: Union[torch.Tensor, list[torch.Tensor]],
        pooling_metadata: PoolingMetadata,
    ) -> torch.Tensor:
        if isinstance(pooling_metadata, V1PoolingMetadata):
            return pooling_metadata.prompt_lens
        assert isinstance(hidden_states, torch.Tensor)
        return PoolingTensors.from_pooling_metadata(
            pooling_metadata, hidden_states.device).prompt_lens

    def forward(
        self,
        hidden_states: Union[torch.Tensor, list[torch.Tensor]],
        pooling_metadata: PoolingMetadata,
    ) -> PoolerOutput:
        """Pools sentence pair scores from the hidden_states."""
        prompt_lens = self.get_prompt_lens(hidden_states, pooling_metadata)

        pooled_data = list[torch.Tensor]()
        if isinstance(hidden_states, list):
            for req_state, prompt_len in zip(hidden_states, prompt_lens):
                assert prompt_len == req_state.shape[0], \
                    "partial prefill not supported with classifier"
            pooled_data = hidden_states
        else:
            offset = 0
            for prompt_len in prompt_lens:
                pooled_data_i = hidden_states[offset:offset + prompt_len]
                offset += prompt_len
                pooled_data.append(pooled_data_i)

        offset = 0
        pooled_data_lst = []
        for pooled_data_i in pooled_data:

            if self.pooler is not None:
                final_shape_tensor = self.pooler(pooled_data_i)
            else:
                final_shape_tensor = self.classifier(pooled_data_i)

            pooled_data_lst.append(final_shape_tensor)

        pooled_output = torch.stack(pooled_data_lst)

        if self.pooler is not None:
            # apply classifier once on the full batch if possible
            pooled_output = self.classifier(pooled_output)

        scores = self.default_activation_function(pooled_output).squeeze(-1)

        pooled_outputs = [PoolingSequenceGroupOutput(data) for data in scores]
        return PoolerOutput(outputs=pooled_outputs)
