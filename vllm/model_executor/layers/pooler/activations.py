# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import TypeVar

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PretrainedConfig

from vllm.config import ModelConfig, get_current_vllm_config
from vllm.logger import init_logger
from vllm.utils.import_utils import resolve_obj_by_qualname

logger = init_logger(__name__)


def get_classification_act_fn(
    config: PretrainedConfig,
) -> "PoolerActivation":
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


def get_cross_encoder_act_fn(
    config: PretrainedConfig,
) -> "PoolerActivation":
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


def resolve_classifier_act_fn(
    model_config: ModelConfig,
    static_num_labels: bool = True,
    act_fn: "PoolerActivation | str | None" = None,
):
    if isinstance(act_fn, str):
        if act_fn == "classify":
            return get_classification_act_fn(model_config.hf_config)
        if act_fn == "score":
            return get_cross_encoder_act_fn(model_config.hf_config)

        raise ValueError(f"act_fn [{act_fn=}] not supported.")

    if act_fn is None:
        return PoolerClassify(static_num_labels=static_num_labels)

    assert callable(act_fn)
    return act_fn


_T = TypeVar("_T", torch.Tensor, list[torch.Tensor])


class PoolerActivation(nn.Module, ABC):
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
        # shape:
        # classify (& score) -> (batch_size, num_classes)
        # embed -> (batch_size, embedding_dim) or list(embedding_dim)
        #          (batch_size, dimensions) or list(dimensions) if using MRL
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
            model_config = vllm_config.model_config
            num_labels = getattr(model_config.hf_config, "num_labels", 0)
        else:
            num_labels = None

        if num_labels == 0:
            logger.warning(
                "num_labels should be > 0 for classification "
                "models, falling back to softmax. "
                "Please check if the configuration is correct."
            )

        self.num_labels = num_labels

    def forward_chunk(self, pooled_data: torch.Tensor) -> torch.Tensor:
        num_labels = self.num_labels
        if num_labels is None:
            num_labels = pooled_data.shape[-1]

        if num_labels < 2:
            return F.sigmoid(pooled_data)

        return F.softmax(pooled_data, dim=-1)


class LambdaPoolerActivation(PoolerActivation):
    def __init__(self, fn: Callable[[torch.Tensor], torch.Tensor]):
        super().__init__()

        self.fn = fn

    def forward_chunk(self, pooled_data: torch.Tensor) -> torch.Tensor:
        return self.fn(pooled_data)
