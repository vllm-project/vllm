# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Callable
from functools import wraps
from typing import TYPE_CHECKING, Concatenate, ParamSpec, TypeVar

import torch

from vllm.config import ModelConfig

from .layerwise import (
    finalize_layerwise_reload,
    initialize_layerwise_reload,
)

if TYPE_CHECKING:
    from vllm.model_executor.models.utils import AutoWeightsLoader

__all__ = ["set_torchao_reload_attrs", "support_quantized_model_reload_from_hp_weights"]

_P = ParamSpec("_P")
_R = TypeVar("_R")


def set_torchao_reload_attrs(model: torch.nn.Module, model_config: ModelConfig):
    model._do_torchao_reload = True
    model._model_config = model_config


def support_quantized_model_reload_from_hp_weights(
    original_load_weights: Callable[Concatenate["AutoWeightsLoader", _P], _R],
) -> Callable[Concatenate["AutoWeightsLoader", _P], _R]:
    """Decorator for `load_weights` method for AutoWeightsLoader.load_weights to support
    reloading high precision (bfloat16/float16/float32) weight for an already quantized
    model, this involves restoring the weights to a high precision weights and
    then online quantize the weights.

    Only applies to torchao quantized models. Assumes that all model weights are
    loaded within a single weights iterator (cannot perform batched updates)
    """

    @wraps(original_load_weights)
    def patched_model_load_weights(
        self: "AutoWeightsLoader",
        *args: _P.args,
        **kwargs: _P.kwargs,
    ) -> _R:
        model = self.module

        if not getattr(model, "_do_torchao_reload", False):
            return original_load_weights(self, *args, **kwargs)

        initialize_layerwise_reload(model)
        loaded_weights = original_load_weights(self, *args, **kwargs)
        finalize_layerwise_reload(model, model._model_config)

        return loaded_weights

    return patched_model_load_weights
