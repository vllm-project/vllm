# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import IntEnum
from typing import Callable, Optional, TypeVar, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from vllm.config import ModelConfig, PoolerConfig
from vllm.model_executor.pooling_metadata import (  # noqa: E501
    PoolingMetadata as V0PoolingMetadata)
from vllm.model_executor.pooling_metadata import PoolingTensors
from vllm.sequence import PoolerOutput, PoolingSequenceGroupOutput
from vllm.transformers_utils.config import (
    get_classification_activation_function,
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


@dataclass(frozen=True)
class ResolvedPoolingConfig:
    pooling_type: PoolingType

    normalize: bool
    softmax: bool
    step_tag_id: Optional[int]
    returned_token_ids: Optional[list[int]]

    @classmethod
    def from_config_with_defaults(
        cls,
        pooler_config: PoolerConfig,
        pooling_type: PoolingType,
        normalize: bool,
        softmax: bool,
        step_tag_id: Optional[int] = None,
        returned_token_ids: Optional[list[int]] = None,
    ) -> "ResolvedPoolingConfig":
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


def get_prompt_lens(
    hidden_states: Union[torch.Tensor, list[torch.Tensor]],
    pooling_metadata: PoolingMetadata,
) -> torch.Tensor:
    if isinstance(pooling_metadata, V1PoolingMetadata):
        return pooling_metadata.prompt_lens

    assert isinstance(hidden_states, torch.Tensor)
    return PoolingTensors.from_pooling_metadata(
        pooling_metadata, hidden_states.device).prompt_lens


def build_output(all_data: torch.Tensor) -> PoolerOutput:
    all_outputs = [PoolingSequenceGroupOutput(data) for data in all_data]
    return PoolerOutput(outputs=all_outputs)


class BasePooler(nn.Module):

    def forward(
        self,
        hidden_states: Union[torch.Tensor, list[torch.Tensor]],
        pooling_metadata: PoolingMetadata,
    ) -> PoolerOutput:
        raise NotImplementedError


class PoolingMethod(ABC):

    @staticmethod
    def from_pooling_type(pooling_type: PoolingType) -> "PoolingMethod":
        if pooling_type == PoolingType.LAST:
            return LastPool()
        if pooling_type == PoolingType.ALL:
            return AllPool()
        if pooling_type == PoolingType.CLS:
            return CLSPool()
        if pooling_type == PoolingType.MEAN:
            return MeanPool()

        raise NotImplementedError(f"Unsupported method: {pooling_type}")

    @abstractmethod
    def forward_one(
            self,
            hidden_states: torch.Tensor,
            prompt_len: Optional[torch.Tensor] = None,  # len(hidden_states)
    ) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def forward_all(
        self,
        hidden_states: torch.Tensor,
        prompt_lens: torch.Tensor,
    ) -> Union[list[torch.Tensor], torch.Tensor]:
        raise NotImplementedError

    def extract_states(
        self,
        hidden_states: Union[torch.Tensor, list[torch.Tensor]],
        pooling_metadata: PoolingMetadata,
    ) -> Union[list[torch.Tensor], torch.Tensor]:
        prompt_lens = get_prompt_lens(hidden_states, pooling_metadata)

        if isinstance(hidden_states, list):
            return [
                self.forward_one(h, prompt_len)
                for h, prompt_len in zip(hidden_states, prompt_lens)
            ]

        return self.forward_all(hidden_states, prompt_lens)


class CLSPool(PoolingMethod):

    def forward_one(
        self,
        hidden_states: torch.Tensor,
        prompt_len: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        assert prompt_len is None or prompt_len == hidden_states.shape[0], \
            "partial prefill not supported with CLS pooling"

        return hidden_states[0]

    def forward_all(
        self,
        hidden_states: torch.Tensor,
        prompt_lens: torch.Tensor,
    ) -> Union[list[torch.Tensor], torch.Tensor]:
        first_token_flat_indices = torch.zeros_like(prompt_lens)
        first_token_flat_indices[1:] += torch.cumsum(prompt_lens, dim=0)[:-1]
        return hidden_states[first_token_flat_indices]


class LastPool(PoolingMethod):

    def forward_one(
        self,
        hidden_states: torch.Tensor,
        prompt_len: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return hidden_states[-1]

    def forward_all(
        self,
        hidden_states: torch.Tensor,
        prompt_lens: torch.Tensor,
    ) -> Union[list[torch.Tensor], torch.Tensor]:
        last_token_flat_indices = torch.cumsum(prompt_lens, dim=0) - 1
        return hidden_states[last_token_flat_indices]


class AllPool(PoolingMethod):

    def forward_one(
        self,
        hidden_states: torch.Tensor,
        prompt_len: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        assert prompt_len is None or prompt_len == hidden_states.shape[0], \
            "partial prefill not supported with ALL pooling"

        return hidden_states

    def forward_all(
        self,
        hidden_states: torch.Tensor,
        prompt_lens: torch.Tensor,
    ) -> Union[list[torch.Tensor], torch.Tensor]:
        offset = 0
        pooled_data = list[torch.Tensor]()

        for prompt_len in prompt_lens:
            pooled_data.append(hidden_states[offset:offset + prompt_len])
            offset += prompt_len

        return pooled_data


class MeanPool(PoolingMethod):

    def forward_one(
        self,
        hidden_states: torch.Tensor,
        prompt_len: torch.Tensor,
    ) -> torch.Tensor:
        assert prompt_len == hidden_states.shape[0], \
            "partial prefill not supported with ALL pooling"

        return hidden_states.mean(dim=0, dtype=torch.float32)

    def forward_all(
        self,
        hidden_states: torch.Tensor,
        prompt_lens: torch.Tensor,
    ) -> Union[list[torch.Tensor], torch.Tensor]:
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


class PoolerHead(nn.Module):

    @classmethod
    def from_config_with_defaults(
        cls,
        pooler_config: PoolerConfig,
        pooling_type: PoolingType,
        normalize: bool,
        softmax: bool,
    ) -> "PoolerHead":
        resolved_config = ResolvedPoolingConfig.from_config_with_defaults(
            pooler_config=pooler_config,
            pooling_type=pooling_type,
            normalize=normalize,
            softmax=softmax,
            step_tag_id=None,
            returned_token_ids=None,
        )

        return cls.from_config(resolved_config)

    @classmethod
    def from_config(cls, pooler_config: ResolvedPoolingConfig) -> "PoolerHead":
        return cls(normalize=pooler_config.normalize,
                   softmax=pooler_config.softmax)

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

        # for matryoshka representation
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
            if len(set(dimensions_list)) == 1 and not isinstance(
                    pooled_data, list):
                # if all dimensions are the same
                d = dimensions_list[0]
                pooled_data = pooled_data[..., :d]
            else:
                pooled_data = [
                    vecs if d is None else vecs[..., :d]
                    for vecs, d in zip(pooled_data, dimensions_list)
                ]

        return self.activation(pooled_data)

    def activation(self, pooled_data: Union[list[torch.Tensor], torch.Tensor]):
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

        # shape:
        # classify (& score) -> (batch_size, num_classes)
        # embed -> (batch_size, embedding_dim) or list(embedding_dim)
        #          (batch_size, dimensions) or list(dimensions) if using MRL
        return pooled_data


class SimplePooler(BasePooler):
    """A layer that pools specific information from hidden states.

    This layer does the following:
    1. Extracts specific tokens or aggregates data based on pooling method.
    2. Normalizes output if specified.
    3. Returns structured results as `PoolerOutput`.

    Attributes:
        pooling_type: The type of pooling to use.
        normalize: Whether to normalize the pooled data.
    """

    @classmethod
    def from_config_with_defaults(
        cls,
        pooler_config: PoolerConfig,
        pooling_type: PoolingType,
        normalize: bool,
        softmax: bool,
    ) -> "SimplePooler":
        resolved_config = ResolvedPoolingConfig.from_config_with_defaults(
            pooler_config=pooler_config,
            pooling_type=pooling_type,
            normalize=normalize,
            softmax=softmax,
        )
        assert resolved_config.pooling_type != PoolingType.LAST

        return cls.from_config(resolved_config)

    @classmethod
    def from_config(
        cls,
        pooler_config: ResolvedPoolingConfig,
    ) -> "SimplePooler":
        method = PoolingMethod.from_pooling_type(pooler_config.pooling_type)
        head = PoolerHead.from_config(pooler_config)

        return cls(method, head)

    def __init__(self, method: PoolingMethod, head: PoolerHead) -> None:
        super().__init__()

        self.method = method
        self.head = head

    def forward(
        self,
        hidden_states: Union[torch.Tensor, list[torch.Tensor]],
        pooling_metadata: PoolingMetadata,
    ) -> PoolerOutput:
        pooled_data = self.method.extract_states(hidden_states,
                                                 pooling_metadata)
        pooled_data = self.head(pooled_data, pooling_metadata)
        return build_output(pooled_data)


class StepPooler(BasePooler):

    @classmethod
    def from_config(cls, pooler_config: ResolvedPoolingConfig) -> "StepPooler":
        assert pooler_config.pooling_type == PoolingType.STEP

        return cls(
            PoolerHead.from_config(pooler_config),
            step_tag_id=pooler_config.step_tag_id,
            returned_token_ids=pooler_config.returned_token_ids,
        )

    def __init__(
        self,
        head: PoolerHead,
        *,
        step_tag_id: Optional[int] = None,
        returned_token_ids: Optional[list[int]] = None,
    ) -> None:
        super().__init__()

        self.head = head
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
        prompt_lens = get_prompt_lens(hidden_states, pooling_metadata)
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

    def forward(
        self,
        hidden_states: Union[torch.Tensor, list[torch.Tensor]],
        pooling_metadata: PoolingMetadata,
    ) -> PoolerOutput:
        pooled_data = self.extract_states(hidden_states, pooling_metadata)
        pooled_data = self.head(pooled_data, pooling_metadata)
        return build_output(pooled_data)


class Pooler(nn.Module):

    @staticmethod
    def from_config_with_defaults(
        pooler_config: PoolerConfig,
        pooling_type: PoolingType,
        normalize: bool,
        softmax: bool,
        step_tag_id: Optional[int] = None,
        returned_token_ids: Optional[list[int]] = None,
    ) -> BasePooler:
        resolved_config = ResolvedPoolingConfig.from_config_with_defaults(
            pooler_config=pooler_config,
            pooling_type=pooling_type,
            normalize=normalize,
            softmax=softmax,
            step_tag_id=step_tag_id,
            returned_token_ids=returned_token_ids,
        )

        if pooling_type == PoolingType.STEP:
            return StepPooler.from_config(resolved_config)

        return SimplePooler.from_config(resolved_config)


PoolerFn = Callable[[torch.Tensor], torch.Tensor]

_T = TypeVar("_T", bound=Union[torch.Tensor, list[torch.Tensor]])
PoolerActivation = Callable[[_T], _T]


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
        pooler: Optional[PoolerFn] = None,
        act_fn: Optional[PoolerActivation] = None,
    ) -> None:
        super().__init__()

        self.classifier = classifier
        self.pooler = pooler

        self.classification_act_fn = get_classification_activation_function(
            config.hf_config) if act_fn is None else act_fn
        self.cross_encoder_act_fn = get_cross_encoder_activation_function(
            config.hf_config) if act_fn is None else act_fn

    def _get_act_fn(self, use_cross_encoder: bool):
        return (self.cross_encoder_act_fn
                if use_cross_encoder else self.classification_act_fn)

    def forward(
        self,
        hidden_states: Union[torch.Tensor, list[torch.Tensor]],
        pooling_metadata: PoolingMetadata,
    ) -> PoolerOutput:
        """Pools sentence pair scores from the hidden_states."""
        prompt_lens = get_prompt_lens(hidden_states, pooling_metadata)

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

        if isinstance(pooling_metadata, V0PoolingMetadata):
            use_cross_encoder_list = [
                pooling_param.use_cross_encoder
                for _, pooling_param in pooling_metadata.seq_groups
            ]
        else:
            use_cross_encoder_list = [
                pooling_param.use_cross_encoder
                for pooling_param in pooling_metadata.pooling_params
            ]

        # shape of scores: (batch_size, num_labels)
        if all(use_cross_encoder == use_cross_encoder_list[0]
               for use_cross_encoder in use_cross_encoder_list):
            act_fn = self._get_act_fn(use_cross_encoder_list[0])
            scores = act_fn(pooled_output)
        else:
            scores = torch.stack([
                self._get_act_fn(use_cross_encoder)(vecs)
                for use_cross_encoder, vecs in zip(use_cross_encoder_list,
                                                   pooled_output)
            ])

        return build_output(scores)
