# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping, Set
from dataclasses import dataclass
from itertools import groupby
from typing import TypeAlias, TypeVar

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PretrainedConfig

from vllm.config import ModelConfig, get_current_vllm_config
from vllm.config.pooler import PoolerConfig, PoolingTypeStr
from vllm.logger import init_logger
from vllm.model_executor.models.adapters import _load_st_projector
from vllm.pooling_params import PoolingParams
from vllm.tasks import PoolingTask
from vllm.utils.import_utils import resolve_obj_by_qualname
from vllm.v1.outputs import PoolerOutput, TokenPoolerOutput, TokenwisePoolerOutput
from vllm.v1.pool.metadata import PoolingMetadata

logger = init_logger(__name__)

PoolingFn = Callable[
    [torch.Tensor | list[torch.Tensor], PoolingMetadata],
    torch.Tensor | list[torch.Tensor],
]
ClassifierFn = Callable[[torch.Tensor], torch.Tensor]


TokenPoolingMethodOutput: TypeAlias = torch.Tensor | list[torch.Tensor]
TokenwisePoolingMethodOutput: TypeAlias = list[torch.Tensor] | list[torch.Tensor | None]
TokenwisePoolingMethodOutputItem: TypeAlias = torch.Tensor | None
PoolingMethodOutput: TypeAlias = TokenPoolingMethodOutput | TokenwisePoolingMethodOutput

TokenPoolerHeadOutput: TypeAlias = torch.Tensor | list[torch.Tensor]
TokenwisePoolerHeadOutput: TypeAlias = torch.Tensor | None


@dataclass(frozen=True)
class ResolvedPoolingConfig:
    pooling_type: PoolingTypeStr
    task: PoolingTask

    @classmethod
    def from_config(
        cls,
        task: PoolingTask,
        pooler_config: PoolerConfig,
    ) -> "ResolvedPoolingConfig":
        assert pooler_config.pooling_type is not None
        return cls(task=task, pooling_type=pooler_config.pooling_type)


@dataclass(frozen=True)
class PoolingParamsUpdate:
    requires_token_ids: bool = False
    """Set this flag to enable `get_prompt_token_ids` for your pooler."""

    def apply(self, params: PoolingParams) -> None:
        params.requires_token_ids = self.requires_token_ids


def get_classification_activation_function(config: PretrainedConfig):
    # Implement alignment with transformers ForSequenceClassificationLoss
    # https://github.com/huggingface/transformers/blob/57bb6db6ee4cfaccc45b8d474dfad5a17811ca60/src/transformers/loss/loss_utils.py#L92
    problem_type = getattr(config, "problem_type", "")
    if problem_type == "regression":
        return PoolerIdentity()
    if problem_type == "single_label_classification":
        return PoolerClassify()
    if problem_type == "multi_label_classification":
        return PoolerMultiLabelClassify()
    return PoolerClassify()


def get_cross_encoder_activation_function(config: PretrainedConfig):
    function_name: str | None = None
    if (
        hasattr(config, "sentence_transformers")
        and "activation_fn" in config.sentence_transformers
    ):
        function_name = config.sentence_transformers["activation_fn"]
    elif (
        hasattr(config, "sbert_ce_default_activation_function")
        and config.sbert_ce_default_activation_function is not None
    ):
        function_name = config.sbert_ce_default_activation_function

    if function_name is not None:
        assert function_name.startswith("torch.nn.modules."), (
            "Loading of activation functions is restricted to "
            "torch.nn.modules for security reasons"
        )
        fn = resolve_obj_by_qualname(function_name)()
        return PoolerActivation.wraps(fn)

    return PoolerClassify()


class PoolingMethod(nn.Module, ABC):
    @staticmethod
    def from_pooling_type(pooling_type: PoolingTypeStr) -> "PoolingMethod":
        if pooling_type == "LAST":
            return LastPool()
        if pooling_type == "ALL":
            return AllPool()
        if pooling_type == "CLS":
            return CLSPool()
        if pooling_type == "MEAN":
            return MeanPool()
        if pooling_type == "STEP":
            raise ValueError(
                "'STEP' pooling is handled by StepPooler "
                "and is not a standalone PoolingMethod."
            )

        raise NotImplementedError(f"Unsupported method: {pooling_type!r}")

    @abstractmethod
    def get_supported_tasks(self) -> Set[PoolingTask]:
        raise NotImplementedError

    def get_pooling_updates(self, task: PoolingTask) -> PoolingParamsUpdate:
        return PoolingParamsUpdate()

    @abstractmethod
    def forward(
        self,
        hidden_states: torch.Tensor,
        pooling_metadata: PoolingMetadata,
    ) -> PoolingMethodOutput:
        raise NotImplementedError


class CLSPool(PoolingMethod):
    def get_supported_tasks(self) -> Set[PoolingTask]:
        return {"token_embed", "token_classify", "embed", "classify", "score"}

    def forward(
        self,
        hidden_states: torch.Tensor,
        pooling_metadata: PoolingMetadata,
    ) -> TokenPoolingMethodOutput:
        pooling_cursor = pooling_metadata.get_pooling_cursor()
        assert not pooling_cursor.is_partial_prefill(), (
            "partial prefill not supported with CLS pooling"
        )

        return hidden_states[pooling_cursor.first_token_indices_gpu]


class LastPool(PoolingMethod):
    def get_supported_tasks(self) -> Set[PoolingTask]:
        return {"token_embed", "token_classify", "embed", "classify", "score"}

    def forward(
        self,
        hidden_states: torch.Tensor,
        pooling_metadata: PoolingMetadata,
    ) -> TokenPoolingMethodOutput:
        pooling_cursor = pooling_metadata.get_pooling_cursor()
        return hidden_states[pooling_cursor.last_token_indices_gpu]


class AllPool(PoolingMethod):
    def __init__(self):
        super().__init__()

        vllm_config = get_current_vllm_config()
        self.enable_chunked_prefill = (
            vllm_config.scheduler_config.enable_chunked_prefill
        )

    def get_supported_tasks(self) -> Set[PoolingTask]:
        return {"token_embed", "token_classify"}

    def forward(
        self,
        hidden_states: torch.Tensor,
        pooling_metadata: PoolingMetadata,
    ) -> TokenwisePoolingMethodOutput:
        pooling_cursor = pooling_metadata.get_pooling_cursor()
        hidden_states_all = hidden_states.split(
            pooling_cursor.num_scheduled_tokens_cpu.tolist()
        )
        hidden_states_lst = [hidden_states_all[i] for i in pooling_cursor.index]

        if not self.enable_chunked_prefill:
            return hidden_states_lst

        pooling_states = pooling_metadata.pooling_states

        # If chunked_prefill is enabled
        # 1. first store the chunked hidden_states in pooling_states.hidden_states_cache
        for p, hs_chunk in zip(pooling_states, hidden_states_lst):
            p.hidden_states_cache.append(hs_chunk)

        # 2. Once prefill is finished, send hidden_states_cache to PoolerHead
        output_list = list[torch.Tensor | None]()
        for p, finished in zip(pooling_states, pooling_cursor.is_finished()):
            if finished:
                hidden_states_cache = p.hidden_states_cache
                if len(hidden_states_cache) == 1:
                    output_list.append(hidden_states_cache[0])
                else:
                    output_list.append(torch.concat(hidden_states_cache, dim=0))
                p.clean()
            else:
                output_list.append(None)

        return output_list


class MeanPool(PoolingMethod):
    def get_supported_tasks(self) -> Set[PoolingTask]:
        return {"token_embed", "token_classify", "embed", "classify", "score"}

    def forward(
        self,
        hidden_states: torch.Tensor,
        pooling_metadata: PoolingMetadata,
    ) -> TokenPoolingMethodOutput:
        pooling_cursor = pooling_metadata.get_pooling_cursor()
        assert not pooling_cursor.is_partial_prefill(), (
            "partial prefill not supported with MEAN pooling"
        )

        prompt_lens = pooling_cursor.prompt_lens_cpu.to(
            hidden_states.device, non_blocking=True
        )

        # Use float32 for torch.cumsum in MeanPool,
        # otherwise precision will be lost significantly.
        cumsum = torch.cumsum(hidden_states, dim=0, dtype=torch.float32)

        start_indices = pooling_cursor.first_token_indices_gpu
        end_indices = pooling_cursor.last_token_indices_gpu
        return (
            cumsum[end_indices] - cumsum[start_indices] + hidden_states[start_indices]
        ) / prompt_lens.unsqueeze(1)


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
        return F.normalize(pooled_data, p=2, dim=-1)


class PoolerMultiLabelClassify(PoolerActivation):
    def forward_chunk(self, pooled_data: torch.Tensor) -> torch.Tensor:
        return F.sigmoid(pooled_data)


class PoolerClassify(PoolerActivation):
    def __init__(self, *, static_num_labels: bool = True) -> None:
        super().__init__()

        if static_num_labels:
            vllm_config = get_current_vllm_config()
            self.num_labels = getattr(
                vllm_config.model_config.hf_config, "num_labels", 0
            )
            if self.num_labels == 0:
                logger.warning(
                    "num_labels should be > 0 for classification"
                    "models, falling back to softmax. "
                    "Please check if the configuration is correct."
                )
        else:
            self.num_labels = None

    def forward_chunk(self, pooled_data: torch.Tensor) -> torch.Tensor:
        num_labels = (
            self.num_labels if self.num_labels is not None else pooled_data.shape[-1]
        )

        if num_labels < 2:
            return F.sigmoid(pooled_data)

        return F.softmax(pooled_data, dim=-1)


class LambdaPoolerActivation(PoolerActivation):
    def __init__(self, fn: Callable[[torch.Tensor], torch.Tensor]):
        super().__init__()

        self.fn = fn

    def forward_chunk(self, pooled_data: torch.Tensor) -> torch.Tensor:
        return self.fn(pooled_data)


class Pooler(nn.Module, ABC):
    """The interface required for all poolers used in pooling models in vLLM."""

    @staticmethod
    def for_token_embed(pooler_config: PoolerConfig):
        head = TokenEmbeddingPoolerHead()

        if pooler_config.pooling_type == "STEP":
            return StepPooler(head=head)

        return AllPooler(head=head)

    @staticmethod
    def for_token_classify(
        pooler_config: PoolerConfig,
        classifier: ClassifierFn | None = None,
        act_fn: PoolerActivation | str | None = None,
    ):
        head = TokenClassifierPoolerHead(classifier=classifier, act_fn=act_fn)

        if pooler_config.pooling_type == "STEP":
            return StepPooler(head=head)

        return AllPooler(head=head)

    @staticmethod
    def for_embed(pooler_config: PoolerConfig):
        resolved_config = ResolvedPoolingConfig.from_config(
            task="embed",
            pooler_config=pooler_config,
        )

        pooling = PoolingMethod.from_pooling_type(resolved_config.pooling_type)
        head = EmbeddingPoolerHead()

        return SimplePooler(pooling=pooling, head=head)

    @staticmethod
    def for_classify(
        pooler_config: PoolerConfig,
        classifier: ClassifierFn | None,
        act_fn: PoolerActivation | str | None = None,
    ):
        resolved_config = ResolvedPoolingConfig.from_config(
            task="classify",
            pooler_config=pooler_config,
        )

        pooling = PoolingMethod.from_pooling_type(resolved_config.pooling_type)

        return ClassifierPooler(
            pooling=pooling,
            classifier=classifier,
            act_fn=act_fn,
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
        hidden_states: torch.Tensor,
        pooling_metadata: PoolingMetadata,
    ) -> PoolerOutput:
        raise NotImplementedError


class DummyPooler(Pooler):
    def get_supported_tasks(self) -> Set[PoolingTask]:
        return {"plugin", "score"}

    def forward(
        self,
        hidden_states: torch.Tensor,
        pooling_metadata: PoolingMetadata,
    ) -> PoolerOutput:
        return hidden_states


class TokenPoolerHead(nn.Module, ABC):
    """Applicable to pooling strategies that output one token."""

    @abstractmethod
    def forward(
        self,
        pooled_data: TokenPoolingMethodOutput,
        pooling_metadata: PoolingMetadata,
    ) -> TokenPoolerHeadOutput:
        raise NotImplementedError


class EmbeddingPoolerHead(TokenPoolerHead):
    def __init__(self) -> None:
        super().__init__()

        # Load ST projector if available
        vllm_config = get_current_vllm_config()
        self.projector = (
            _load_st_projector(vllm_config.model_config) if vllm_config else None
        )
        self.head_dtype = vllm_config.model_config.head_dtype

        self.activation = PoolerNormalize()

    def forward(
        self,
        pooled_data: TokenPoolingMethodOutput,
        pooling_metadata: PoolingMetadata,
    ) -> TokenPoolerHeadOutput:
        if isinstance(pooled_data, list):
            pooled_data = torch.stack(pooled_data)
        # pooled_data shape: [batchsize, hidden_dimension]

        pooled_data = pooled_data.to(self.head_dtype)

        # Apply ST projector
        if self.projector is not None:
            pooled_data = self.projector(pooled_data)
        # pooled_data shape: [batchsize, embedding_dimension]

        pooling_params = pooling_metadata.pooling_params

        # for matryoshka representation
        dimensions_list = [pooling_param.dimensions for pooling_param in pooling_params]
        if any(d is not None for d in dimensions_list):
            # change the output dimension
            assert len(pooled_data) == len(dimensions_list)
            if len(set(dimensions_list)) == 1 and not isinstance(pooled_data, list):
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

        # pooled_data shape: [batchsize, embedding_dimension]
        return pooled_data


class SimplePooler(Pooler):
    """A layer that pools specific information from hidden states.

    This layer does the following:
    1. Extracts specific tokens or aggregates data based on pooling method.
    2. Normalizes output if specified.
    3. Returns structured results as `PoolerOutput`.
    """

    def __init__(self, pooling: PoolingMethod, head: TokenPoolerHead) -> None:
        super().__init__()

        self.pooling = pooling
        self.head = head

    def get_supported_tasks(self) -> Set[PoolingTask]:
        return self.pooling.get_supported_tasks()

    def get_pooling_updates(self, task: PoolingTask) -> PoolingParamsUpdate:
        return self.pooling.get_pooling_updates(task)

    def forward(
        self,
        hidden_states: torch.Tensor,
        pooling_metadata: PoolingMetadata,
    ) -> TokenPoolerHeadOutput:
        pooled_data = self.pooling(hidden_states, pooling_metadata)
        pooled_data = self.head(pooled_data, pooling_metadata)
        return pooled_data


class ClassifierPooler(Pooler):
    """A pooling layer for classification tasks.

    This layer does the following:
    1. Applies a classification layer to the hidden states.
    2. Optionally applies a pooler layer.
    3. Applies an activation function to the output.
    """

    @staticmethod
    def act_fn_for_seq_cls(model_config: ModelConfig):
        return get_classification_activation_function(model_config.hf_config)

    @staticmethod
    def act_fn_for_cross_encoder(model_config: ModelConfig):
        return get_cross_encoder_activation_function(model_config.hf_config)

    @staticmethod
    def resolve_act_fn(
        model_config: ModelConfig,
        static_num_labels: bool = True,
        act_fn: PoolerActivation | str | None = None,
    ):
        if isinstance(act_fn, str):
            if act_fn == "classify":
                return ClassifierPooler.act_fn_for_seq_cls(model_config)
            elif act_fn == "score":
                return ClassifierPooler.act_fn_for_cross_encoder(model_config)
            else:
                raise ValueError(f"act_fn [{act_fn=}] not supported.")
        elif act_fn is None:
            return PoolerClassify(static_num_labels=static_num_labels)
        else:
            assert callable(act_fn)
            return act_fn

    def __init__(
        self,
        pooling: PoolingFn,
        classifier: ClassifierFn | None,
        act_fn: PoolerActivation | str | None = None,
    ) -> None:
        super().__init__()

        vllm_config = get_current_vllm_config()
        self.pooling = pooling
        self.classifier = classifier
        self.act_fn = self.resolve_act_fn(
            vllm_config.model_config, static_num_labels=True, act_fn=act_fn
        )
        self.logit_bias: float | None = (
            vllm_config.model_config.pooler_config.logit_bias
        )
        self.head_dtype = vllm_config.model_config.head_dtype

    def get_supported_tasks(self) -> Set[PoolingTask]:
        return {"classify", "score"}

    def forward(
        self,
        hidden_states: torch.Tensor,
        pooling_metadata: PoolingMetadata,
    ) -> TokenPoolerOutput:
        pooled_data = self.pooling(hidden_states, pooling_metadata)
        if isinstance(pooled_data, list):
            pooled_data = torch.stack(pooled_data)
        # pooled_data shape: [batchsize, hidden_size]

        pooled_data = pooled_data.to(self.head_dtype)

        if self.classifier is not None:
            pooled_data = self.classifier(pooled_data)
        # pooled_data shape: [batchsize, num_labels]

        if self.logit_bias is not None:
            pooled_data -= self.logit_bias

        pooling_params = pooling_metadata.pooling_params
        flags = [p.use_activation for p in pooling_params]

        if len(set(flags)) == 1:
            scores = self.act_fn(pooled_data) if flags[0] else pooled_data
        else:
            scores = [
                self.act_fn(vecs) if f else vecs for vecs, f in zip(pooled_data, flags)
            ]

        # scores shape: [batchsize, num_labels]
        return scores


class TokenwisePoolerHead(nn.Module, ABC):
    """Applicable to pooling strategies that output multiple tokens."""

    @abstractmethod
    def forward(
        self,
        pooled_data: TokenwisePoolingMethodOutputItem,
        pooling_param: PoolingParams,
    ) -> TokenwisePoolerHeadOutput:
        raise NotImplementedError


class TokenEmbeddingPoolerHead(TokenwisePoolerHead):
    def __init__(self) -> None:
        super().__init__()

        # Load ST projector if available
        vllm_config = get_current_vllm_config()
        self.projector = (
            _load_st_projector(vllm_config.model_config) if vllm_config else None
        )
        self.head_dtype = vllm_config.model_config.head_dtype

        self.activation = PoolerNormalize()

    def forward(
        self,
        pooled_data: TokenwisePoolingMethodOutputItem,
        pooling_param: PoolingParams,
    ) -> TokenwisePoolerHeadOutput:
        # for unfinished chunked prefill
        if pooled_data is None:
            return None

        pooled_data = pooled_data.to(self.head_dtype)
        # pooled_data shape: [n_tokens, hidden_dimension]

        # Apply ST projector
        if self.projector is not None:
            pooled_data = self.projector(pooled_data)
        # pooled_data shape: [n_tokens, embedding_dimension]

        # for matryoshka representation
        pooled_data = pooled_data[..., : pooling_param.dimensions]

        # for normalize
        if pooling_param.normalize:
            pooled_data = self.activation(pooled_data)

        # pooled_data shape: [n_tokens, embedding_dimension]
        return pooled_data


class TokenClassifierPoolerHead(TokenwisePoolerHead):
    def __init__(
        self,
        classifier: ClassifierFn | None,
        act_fn: PoolerActivation | str | None = None,
    ) -> None:
        super().__init__()

        vllm_config = get_current_vllm_config()

        self.classifier = classifier
        self.logit_bias: float | None = (
            vllm_config.model_config.pooler_config.logit_bias
        )
        self.head_dtype = vllm_config.model_config.head_dtype

        self.activation = ClassifierPooler.resolve_act_fn(
            vllm_config.model_config, static_num_labels=False, act_fn=act_fn
        )

    def forward(
        self,
        pooled_data: TokenwisePoolingMethodOutputItem,
        pooling_param: PoolingParams,
    ) -> TokenwisePoolerHeadOutput:
        # for unfinished chunked prefill
        if pooled_data is None:
            return None

        pooled_data = pooled_data.to(self.head_dtype)
        # hidden_states shape: [n_token, hidden_size]

        if self.classifier is not None:
            scores = self.classifier(pooled_data)
        else:
            scores = pooled_data
        # scores shape: [n_token, num_labels]

        if self.logit_bias is not None:
            scores -= self.logit_bias

        if pooling_param.use_activation:
            scores = self.activation(scores)

        # scores shape: [n_token, num_labels]
        return scores


class AllPooler(Pooler):
    def __init__(self, head: TokenwisePoolerHead) -> None:
        super().__init__()

        self.pooling = AllPool()
        self.head = head

    def get_supported_tasks(self) -> Set[PoolingTask]:
        return {"token_embed", "token_classify"}

    def forward(
        self,
        hidden_states: torch.Tensor,
        pooling_metadata: PoolingMetadata,
    ) -> TokenwisePoolerOutput:
        pooled_data = self.pooling(hidden_states, pooling_metadata)
        pooling_params = pooling_metadata.pooling_params
        assert len(pooled_data) == len(pooling_params)

        return [self.head(d, p) for d, p in zip(pooled_data, pooling_params)]


class StepPooler(Pooler):
    def __init__(self, head: TokenwisePoolerHead) -> None:
        super().__init__()

        self.pooling = AllPool()
        self.head = head

    def extract_states(
        self,
        hidden_states: torch.Tensor,
        pooling_metadata: PoolingMetadata,
    ) -> list[torch.Tensor | None]:
        pooled_data_lst = self.pooling(hidden_states, pooling_metadata)
        prompt_token_ids = pooling_metadata.get_prompt_token_ids()
        pooling_params = pooling_metadata.pooling_params

        pooled_data = list[torch.Tensor | None]()
        for data, token_id, pooling_param in zip(
            pooled_data_lst, prompt_token_ids, pooling_params
        ):
            # for unfinished chunked prefill
            if data is None:
                pooled_data.append(data)
                continue

            step_tag_id = pooling_param.step_tag_id
            returned_token_ids = pooling_param.returned_token_ids

            if returned_token_ids is not None and len(returned_token_ids) > 0:
                data = data[:, returned_token_ids]

            if step_tag_id is not None:
                data = data[token_id == step_tag_id]
            pooled_data.append(data)

        return pooled_data

    def get_supported_tasks(self) -> Set[PoolingTask]:
        return {"token_embed", "token_classify"}

    def get_pooling_updates(self, task: PoolingTask) -> PoolingParamsUpdate:
        return PoolingParamsUpdate(requires_token_ids=True)

    def forward(
        self,
        hidden_states: torch.Tensor,
        pooling_metadata: PoolingMetadata,
    ) -> TokenwisePoolerOutput:
        pooled_data = self.extract_states(hidden_states, pooling_metadata)
        pooling_params = pooling_metadata.pooling_params
        assert len(pooled_data) == len(pooling_params)

        return [self.head(d, p) for d, p in zip(pooled_data, pooling_params)]


class DispatchPooler(Pooler):
    """Dispatches calls to a sub-pooler based on the pooling task."""

    def __init__(self, poolers_by_task: Mapping[PoolingTask, Pooler]) -> None:
        super().__init__()

        for task, pooler in poolers_by_task.items():
            if task not in pooler.get_supported_tasks():
                raise ValueError(
                    f"{pooler=} does not support {task=}. "
                    f"Supported tasks: {pooler.get_supported_tasks()}"
                )

        self.poolers_by_task = poolers_by_task

    def get_supported_tasks(self) -> Set[PoolingTask]:
        return set(self.poolers_by_task)

    def get_pooling_updates(self, task: PoolingTask) -> PoolingParamsUpdate:
        return self.poolers_by_task[task].get_pooling_updates(task)

    def forward(
        self,
        hidden_states: torch.Tensor,
        pooling_metadata: PoolingMetadata,
    ) -> PoolerOutput:
        poolers_by_task = self.poolers_by_task

        outputs = list[torch.Tensor | None]()
        offset = 0
        for task, group in groupby(pooling_metadata.tasks):
            if not (pooler := poolers_by_task.get(task)):
                raise ValueError(
                    f"Unsupported task: {task} "
                    f"Supported tasks: {self.get_supported_tasks()}"
                )

            num_items = len(list(group))
            group_output: PoolerOutput = pooler(
                hidden_states,
                pooling_metadata[offset : offset + num_items],
            )

            outputs.extend(group_output)
            offset += num_items

        return outputs

    def extra_repr(self) -> str:
        s = f"supported_task={self.get_supported_tasks()}"
        return s
