# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from abc import ABC, abstractmethod
from collections.abc import Mapping, Set
from dataclasses import dataclass
from enum import IntEnum
from itertools import groupby
from typing import Callable, Optional, TypeVar, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PretrainedConfig

from vllm.config import ModelConfig, PoolerConfig
from vllm.model_executor.pooling_metadata import (  # noqa: E501
    PoolingMetadata as V0PoolingMetadata)
from vllm.model_executor.pooling_metadata import PoolingTensors
from vllm.pooling_params import PoolingParams
from vllm.sequence import PoolerOutput, PoolingSequenceGroupOutput
from vllm.tasks import PoolingTask
from vllm.utils import resolve_obj_by_qualname
from vllm.v1.pool.metadata import PoolingMetadata as V1PoolingMetadata

PoolingMetadata = Union[V0PoolingMetadata, V1PoolingMetadata]
PoolingFn = Callable[
    [Union[torch.Tensor, list[torch.Tensor]], PoolingMetadata],
    Union[torch.Tensor, list[torch.Tensor]]]
ClassifierFn = Callable[[torch.Tensor], torch.Tensor]


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
    task: PoolingTask

    @classmethod
    def from_config(
        cls,
        task: PoolingTask,
        pooler_config: PoolerConfig,
    ) -> "ResolvedPoolingConfig":
        assert pooler_config.pooling_type is not None
        return cls(task=task,
                   pooling_type=PoolingType[pooler_config.pooling_type])


@dataclass(frozen=True)
class PoolingParamsUpdate:
    requires_token_ids: bool = False
    """Set this flag to enable `get_prompt_token_ids` for your pooler."""

    def apply(self, params: PoolingParams) -> None:
        params.requires_token_ids = self.requires_token_ids


class Pooler(nn.Module, ABC):
    """The interface required for all poolers used in pooling models in vLLM."""

    @staticmethod
    def for_encode(pooler_config: PoolerConfig):
        if pooler_config.pooling_type == "STEP":
            return StepPooler()

        resolved_config = ResolvedPoolingConfig(task="encode",
                                                pooling_type=PoolingType.ALL)

        return SimplePooler.from_config(resolved_config)

    @staticmethod
    def for_embed(pooler_config: PoolerConfig):
        resolved_config = ResolvedPoolingConfig.from_config(
            task="embed",
            pooler_config=pooler_config,
        )

        return SimplePooler.from_config(resolved_config)

    @staticmethod
    def for_classify(
        pooler_config: PoolerConfig,
        classifier: Optional[ClassifierFn],
    ):
        resolved_config = ResolvedPoolingConfig.from_config(
            task="classify",
            pooler_config=pooler_config,
        )

        pooling = PoolingMethod.from_pooling_type(resolved_config.pooling_type)

        return ClassifierPooler(
            pooling=pooling,
            classifier=classifier,
        )

    @abstractmethod
    def get_supported_tasks(self) -> Set[PoolingTask]:
        """Determine which pooling tasks are supported."""
        raise NotImplementedError

    def get_pooling_updates(self, task: PoolingTask) -> PoolingParamsUpdate:
        """
        Construct the updated pooling parameters to use for a supported task.
        """
        return PoolingParamsUpdate()

    @abstractmethod
    def forward(
        self,
        hidden_states: Union[list[torch.Tensor], torch.Tensor],
        pooling_metadata: PoolingMetadata,
    ) -> PoolerOutput:
        raise NotImplementedError


def get_prompt_lens(
    hidden_states: Union[torch.Tensor, list[torch.Tensor]],
    pooling_metadata: PoolingMetadata,
) -> torch.Tensor:
    if isinstance(pooling_metadata, V1PoolingMetadata):
        return pooling_metadata.prompt_lens

    return PoolingTensors.from_pooling_metadata(
        pooling_metadata, hidden_states[0].device).prompt_lens


def get_prompt_token_ids(
        pooling_metadata: PoolingMetadata) -> list[torch.Tensor]:
    if isinstance(pooling_metadata, V1PoolingMetadata):
        assert pooling_metadata.prompt_token_ids is not None, (
            "Please set `requires_token_ids=True` in `get_pooling_updates`")

        return [
            pooling_metadata.prompt_token_ids[i, :num]
            for i, num in enumerate(pooling_metadata.prompt_lens)
        ]

    return [
        torch.tensor(seq_data_i.prompt_token_ids)
        for seq_data_i in pooling_metadata.seq_data.values()
    ]


def get_pooling_params(
        pooling_metadata: PoolingMetadata) -> list[PoolingParams]:
    if isinstance(pooling_metadata, V0PoolingMetadata):
        pooling_params = [p for _, p in pooling_metadata.seq_groups]
    else:
        pooling_params = pooling_metadata.pooling_params
    return pooling_params


def get_tasks(pooling_metadata: PoolingMetadata) -> list[PoolingTask]:
    pooling_params = get_pooling_params(pooling_metadata)

    tasks: list[PoolingTask] = [
        task for pooling_param in pooling_params
        if (task := pooling_param.task) is not None
    ]
    assert len(pooling_params) == len(tasks)

    return tasks


def get_classification_activation_function(config: PretrainedConfig):
    return PoolerClassify()


def get_cross_encoder_activation_function(config: PretrainedConfig):
    function_name: Optional[str] = None
    if (hasattr(config, "sentence_transformers")
            and "activation_fn" in config.sentence_transformers):
        function_name = config.sentence_transformers["activation_fn"]
    elif (hasattr(config, "sbert_ce_default_activation_function")
          and config.sbert_ce_default_activation_function is not None):
        function_name = config.sbert_ce_default_activation_function

    if function_name is not None:
        assert function_name.startswith("torch.nn.modules."), (
            "Loading of activation functions is restricted to "
            "torch.nn.modules for security reasons")
        fn = resolve_obj_by_qualname(function_name)()
        return PoolerActivation.wraps(fn)

    return PoolerScore()


def build_output(
    all_data: Union[torch.Tensor, list[torch.Tensor]], ) -> PoolerOutput:
    all_outputs = [PoolingSequenceGroupOutput(data) for data in all_data]
    return PoolerOutput(outputs=all_outputs)


class PoolingMethod(nn.Module, ABC):

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
    def get_supported_tasks(self) -> Set[PoolingTask]:
        raise NotImplementedError

    def get_pooling_updates(self, task: PoolingTask) -> PoolingParamsUpdate:
        return PoolingParamsUpdate()

    @abstractmethod
    def forward_one(
        self,
        hidden_states: torch.Tensor,
        prompt_len: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Note:
            `prompt_len=None` means `prompt_len=len(hidden_states)`.
        """
        raise NotImplementedError

    @abstractmethod
    def forward_all(
        self,
        hidden_states: torch.Tensor,
        prompt_lens: torch.Tensor,
    ) -> Union[list[torch.Tensor], torch.Tensor]:
        raise NotImplementedError

    def forward(
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

    def get_supported_tasks(self) -> Set[PoolingTask]:
        return {"encode", "embed", "classify", "score"}

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

    def get_supported_tasks(self) -> Set[PoolingTask]:
        return {"encode", "embed", "classify", "score"}

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

    def get_supported_tasks(self) -> Set[PoolingTask]:
        return {"encode"}

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
        return list(hidden_states.split_with_sizes(prompt_lens.tolist()))


class MeanPool(PoolingMethod):

    def get_supported_tasks(self) -> Set[PoolingTask]:
        return {"encode", "embed", "classify", "score"}

    def forward_one(
        self,
        hidden_states: torch.Tensor,
        prompt_len: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        assert prompt_len is None or prompt_len == hidden_states.shape[0], \
            "partial prefill not supported with MEAN pooling"

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


_T = TypeVar("_T", torch.Tensor, list[torch.Tensor])


class BasePoolerActivation(nn.Module, ABC):

    @abstractmethod
    def forward(self, pooled_data: _T) -> _T:
        # shape:
        # classify (& score) -> (batch_size, num_classes)
        # embed -> (batch_size, embedding_dim) or list(embedding_dim)
        #          (batch_size, dimensions) or list(dimensions) if using MRL
        raise NotImplementedError


class PoolerActivation(BasePoolerActivation):

    @staticmethod
    def wraps(module: nn.Module):
        if isinstance(module, nn.Identity):
            return PoolerIdentity()
        if isinstance(module, (nn.Sigmoid, nn.Softmax)):
            return PoolerClassify()

        return LambdaPoolerActivation(module)

    @abstractmethod
    def forward_chunk(self, pooled_data: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def forward(self, pooled_data: _T) -> _T:
        if isinstance(pooled_data, list):
            return [self.forward_chunk(data) for data in pooled_data]

        return self.forward_chunk(pooled_data)


class PoolerIdentity(PoolerActivation):

    def forward_chunk(self, pooled_data: torch.Tensor) -> torch.Tensor:
        return pooled_data


class PoolerNormalize(PoolerActivation):

    def forward_chunk(self, pooled_data: torch.Tensor) -> torch.Tensor:
        x = F.normalize(pooled_data.float(), p=2, dim=-1)
        return x.to(pooled_data.dtype)


class PoolerClassify(PoolerActivation):

    def forward_chunk(self, pooled_data: torch.Tensor) -> torch.Tensor:
        num_labels = pooled_data.shape[-1]
        if num_labels < 2:
            return F.sigmoid(pooled_data.float()).to(pooled_data.dtype)

        return F.softmax(pooled_data.float(), dim=-1).to(pooled_data.dtype)


class PoolerScore(PoolerActivation):

    def forward_chunk(self, pooled_data: torch.Tensor) -> torch.Tensor:
        num_labels = pooled_data.shape[-1]
        if num_labels < 2:
            return F.sigmoid(pooled_data.float()).to(pooled_data.dtype)

        return pooled_data


class LambdaPoolerActivation(PoolerActivation):

    def __init__(self, fn: Callable[[torch.Tensor], torch.Tensor]):
        super().__init__()

        self.fn = fn

    def forward_chunk(self, pooled_data: torch.Tensor) -> torch.Tensor:
        return self.fn(pooled_data)


class PoolerHead(nn.Module):

    def __init__(self, activation: PoolerActivation) -> None:
        super().__init__()
        self.activation = activation

    def forward(self, pooled_data: Union[list[torch.Tensor], torch.Tensor],
                pooling_metadata: PoolingMetadata):

        return self.activation(pooled_data)


class EmbeddingPoolerHead(PoolerHead):

    def __init__(self) -> None:
        super().__init__(activation=PoolerNormalize())

    def forward(self, pooled_data: Union[list[torch.Tensor], torch.Tensor],
                pooling_metadata: PoolingMetadata):

        pooling_params = get_pooling_params(pooling_metadata)

        # for matryoshka representation
        dimensions_list = [
            pooling_param.dimensions for pooling_param in pooling_params
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

        # for normalize
        flags = [p.normalize for p in pooling_params]
        if len(set(flags)) == 1:
            if flags[0]:
                pooled_data = self.activation(pooled_data)
        else:
            pooled_data = [
                self.activation(vecs) if f else vecs
                for vecs, f in zip(pooled_data, flags)
            ]

        return pooled_data


class RewardPoolerHead(PoolerHead):

    def __init__(self) -> None:
        super().__init__(activation=PoolerClassify())

    def forward(self, pooled_data: Union[list[torch.Tensor], torch.Tensor],
                pooling_metadata: PoolingMetadata):
        pooling_params = get_pooling_params(pooling_metadata)

        # for softmax
        flags = [p.softmax for p in pooling_params]
        if len(set(flags)) == 1:
            if flags[0]:
                pooled_data = self.activation(pooled_data)
        else:
            pooled_data = [
                self.activation(vecs) if f else vecs
                for vecs, f in zip(pooled_data, flags)
            ]

        return pooled_data


class SimplePooler(Pooler):
    """A layer that pools specific information from hidden states.

    This layer does the following:
    1. Extracts specific tokens or aggregates data based on pooling method.
    2. Normalizes output if specified.
    3. Returns structured results as `PoolerOutput`.
    """

    @classmethod
    def from_config(
        cls,
        pooler_config: ResolvedPoolingConfig,
    ) -> "SimplePooler":
        pooling = PoolingMethod.from_pooling_type(pooler_config.pooling_type)
        if pooler_config.task == "embed":
            head = EmbeddingPoolerHead()
        elif pooler_config.task == "encode":
            head = RewardPoolerHead()
        else:
            raise NotImplementedError(f"Unknown task: {pooler_config.task}")
        return cls(pooling, head)

    def __init__(self, pooling: PoolingMethod, head: PoolerHead) -> None:
        super().__init__()

        self.pooling = pooling
        self.head = head

    def get_supported_tasks(self) -> Set[PoolingTask]:
        return self.pooling.get_supported_tasks()

    def get_pooling_updates(self, task: PoolingTask) -> PoolingParamsUpdate:
        return self.pooling.get_pooling_updates(task)

    def forward(
        self,
        hidden_states: Union[torch.Tensor, list[torch.Tensor]],
        pooling_metadata: PoolingMetadata,
    ) -> PoolerOutput:
        pooled_data = self.pooling(hidden_states, pooling_metadata)
        pooled_data = self.head(pooled_data, pooling_metadata)
        return build_output(pooled_data)


class StepPooler(Pooler):

    def __init__(self, ) -> None:
        super().__init__()

        self.pooling = AllPool()
        self.head = RewardPoolerHead()

    def extract_states(
        self,
        hidden_states: Union[torch.Tensor, list[torch.Tensor]],
        pooling_metadata: PoolingMetadata,
    ) -> Union[list[torch.Tensor], torch.Tensor]:
        pooled_data_lst = self.pooling(hidden_states, pooling_metadata)
        prompt_token_ids = get_prompt_token_ids(pooling_metadata)

        pooled_data = list[torch.Tensor]()

        pooling_params = get_pooling_params(pooling_metadata)

        for data, token_id, pooling_param in zip(pooled_data_lst,
                                                 prompt_token_ids,
                                                 pooling_params):
            step_tag_id = pooling_param.step_tag_id
            returned_token_ids = pooling_param.returned_token_ids

            if returned_token_ids is not None and len(returned_token_ids) > 0:
                data = data[:, returned_token_ids]

            if step_tag_id is not None:
                data = data[token_id == step_tag_id]
            pooled_data.append(data)

        return pooled_data

    def get_supported_tasks(self) -> Set[PoolingTask]:
        return {"encode"}

    def get_pooling_updates(self, task: PoolingTask) -> PoolingParamsUpdate:
        return PoolingParamsUpdate(requires_token_ids=True)

    def forward(
        self,
        hidden_states: Union[torch.Tensor, list[torch.Tensor]],
        pooling_metadata: PoolingMetadata,
    ) -> PoolerOutput:
        pooled_data = self.extract_states(hidden_states, pooling_metadata)
        pooled_data = self.head(pooled_data, pooling_metadata)
        return build_output(pooled_data)


class ClassifierPooler(Pooler):
    """A pooling layer for classification tasks.

    This layer does the following:
    1. Applies a classification layer to the hidden states.
    2. Optionally applies a pooler layer.
    3. Applies an activation function to the output.
    """

    @staticmethod
    def act_fn_for_seq_cls(config: ModelConfig):
        return get_classification_activation_function(config.hf_config)

    @staticmethod
    def act_fn_for_cross_encoder(config: ModelConfig):
        return get_cross_encoder_activation_function(config.hf_config)

    def __init__(
        self,
        pooling: PoolingFn,
        classifier: Optional[ClassifierFn],
        act_fn: Optional[PoolerActivation] = None,
    ) -> None:
        super().__init__()

        self.pooling = pooling
        self.classifier = classifier
        self.act_fn = act_fn or PoolerClassify()

    def get_supported_tasks(self) -> Set[PoolingTask]:
        return {"classify", "score"}

    def forward(
        self,
        hidden_states: Union[torch.Tensor, list[torch.Tensor]],
        pooling_metadata: PoolingMetadata,
    ) -> PoolerOutput:
        pooled_data = self.pooling(hidden_states, pooling_metadata)

        if self.classifier is not None:
            # apply classifier once on the full batch if possible
            if isinstance(pooled_data, torch.Tensor):
                pooled_data = self.classifier(pooled_data)
            elif len({data.shape for data in pooled_data}) <= 1:
                pooled_data = self.classifier(torch.stack(pooled_data))
            else:
                pooled_data = [self.classifier(data) for data in pooled_data]

        pooling_params = get_pooling_params(pooling_metadata)
        flags = [p.activation for p in pooling_params]

        if len(set(flags)) == 1:
            scores = self.act_fn(pooled_data) if flags[0] else pooled_data
        else:
            scores = [
                self.act_fn(vecs) if f else vecs
                for vecs, f in zip(pooled_data, flags)
            ]

        return build_output(scores)


class DispatchPooler(Pooler):
    """Dispatches calls to a sub-pooler based on the pooling task."""

    def __init__(self, poolers_by_task: Mapping[PoolingTask, Pooler]) -> None:
        super().__init__()

        for task, pooler in poolers_by_task.items():
            if task not in pooler.get_supported_tasks():
                raise ValueError(
                    f"{pooler=} does not support {task=}. "
                    f"Supported tasks: {pooler.get_supported_tasks()}")

        self.poolers_by_task = poolers_by_task

    def get_supported_tasks(self) -> Set[PoolingTask]:
        return set(self.poolers_by_task)

    def get_pooling_updates(self, task: PoolingTask) -> PoolingParamsUpdate:
        return self.poolers_by_task[task].get_pooling_updates(task)

    def forward(
        self,
        hidden_states: Union[torch.Tensor, list[torch.Tensor]],
        pooling_metadata: PoolingMetadata,
    ) -> PoolerOutput:
        poolers_by_task = self.poolers_by_task

        if isinstance(hidden_states, list):
            hidden_states_lst = hidden_states
        else:
            prompt_lens = get_prompt_lens(hidden_states, pooling_metadata)
            hidden_states_lst = list(hidden_states.split(prompt_lens.tolist()))

        outputs = list[PoolingSequenceGroupOutput]()
        offset = 0
        for task, group in groupby(get_tasks(pooling_metadata)):
            if not (pooler := poolers_by_task.get(task)):
                raise ValueError(
                    f"Unsupported task: {task} "
                    f"Supported tasks: {self.get_supported_tasks()}")

            num_items = len(list(group))
            group_output: PoolerOutput = pooler(
                hidden_states_lst[offset:offset + num_items],
                pooling_metadata[offset:offset + num_items],
            )

            outputs.extend(group_output.outputs)
            offset += num_items

        return PoolerOutput(outputs)
