# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Iterable
from functools import wraps
from types import FunctionType
from typing import TYPE_CHECKING

import torch

from .layerwise import (
    finalize_layerwise_restore_and_process,
    layerwise_restore_and_process,
)

if TYPE_CHECKING:
    from vllm.model_executor.models.utils import AutoWeightsLoader

__all__ = ["set_torchao_reload_attrs", "support_quantized_model_reload_from_hp_weights"]


def set_torchao_reload_attrs(model: torch.nn.Module):
    """Only called when using torchao quantization"""
    model._do_torchao_reload = True


def support_quantized_model_reload_from_hp_weights(original_load_weights: FunctionType):
    """
    Decorator for `load_weights` method for AutoWeightsLoader.load_weights to support
    reloading high precision (bfloat16/float16/float32) weight for an already quantized
    model, this involves restoring the weights to a high precision weights and
    then online quantize the weights.

    Only applies to torchao quantized models. Assumes that all model weights are
    loaded within a single weights iterator (cannot perform batched updates)
    """

    @wraps(original_load_weights)
    def patched_model_load_weights(
        self: "AutoWeightsLoader",
        weights: Iterable[tuple[str, torch.Tensor]],
        *,
        mapper=None,
    ):
        model = self.module

        if not getattr(model, "_do_torchao_reload", False):
            return original_load_weights(self, weights, mapper=mapper)

        model.apply(layerwise_restore_and_process)
        loaded_weights = original_load_weights(self, weights, mapper=mapper)
        model.apply(finalize_layerwise_restore_and_process)

        return loaded_weights

    return patched_model_load_weights
