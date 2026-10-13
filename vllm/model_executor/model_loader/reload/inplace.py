# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import time
from collections.abc import Iterable
from typing import cast

import torch
import torch.nn as nn

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.model_executor.model_loader import get_model_loader

from .layerwise import finalize_layerwise_reload, initialize_layerwise_reload

logger = init_logger(__name__)

__all__ = ["get_parameter_for_reload", "reload_weights"]


def get_parameter_for_reload(model: nn.Module, name: str) -> nn.Parameter:
    """Resolve checkpoint names without changing the model's module tree."""
    from vllm.lora.layers import BaseLayerWithLoRA

    module_name, _, parameter_name = name.rpartition(".")
    module = model.get_submodule(module_name)
    if isinstance(module, BaseLayerWithLoRA):
        module = module.base_layer
    return module.get_parameter(parameter_name)


def reload_weights(
    vllm_config: VllmConfig,
    model: nn.Module,
    weights_iterator: Iterable[tuple[str, torch.Tensor]] | None,
    weights_path: str | None,
    is_checkpoint_format: bool,
) -> None:
    """Reload a model's weights in place from a weights iterator or from disk.

    Callers are responsible for invalidating any state derived from the old
    weights (e.g. LoRA, encoder and multimodal caches).

    Args:
        vllm_config: config of the model being reloaded. If `weights_path` is
            given, `vllm_config.model_config` is updated to point at it.
        model: model to load weights into
        weights_iterator: weights to load into model
        weights_path: path to load weights from if weights_iterator is not
            provided. Use path of original model if neither is provided.
        is_checkpoint_format: set to False if weights have already been
            processed into kernel format (repacking, renaming, etc.)

    """
    # TODO(@kylesayrs): generalize to all runners and loaders
    # argument validation
    if weights_iterator is None and not is_checkpoint_format:
        logger.warning(
            "Reloading from disk means that weights will be in checkpoint format. "
            "Please use `is_checkpoint_format=True` "
            "to avoid weight reloading errors"
        )

    model_config = vllm_config.model_config
    load_config = vllm_config.load_config
    weights_to_load = {
        name.replace(".base_layer.", ".") if vllm_config.lora_config else name
        for name, _ in model.named_parameters()
    }
    counter_before_reloading = time.perf_counter()

    # load weights from disk if none are provided
    if weights_iterator is None:
        model_loader = get_model_loader(load_config)
        if not hasattr(model_loader, "get_all_weights"):
            raise NotImplementedError(
                f"Model reloading with `{load_config.load_format}` format"
            )

        if weights_path is not None:
            # The revision and any object-storage `model_weights` source
            # belong to the model we are reloading away from, so they must
            # not be carried over to the new path.
            model_config.model = weights_path
            model_config.model_weights = ""
            model_config.revision = None
        weights_iterator = model_loader.get_all_weights(model_config, model)
        weights_iterator = cast(Iterable[tuple[str, torch.Tensor]], weights_iterator)

    # begin loading weights
    logger.info_once("Reloading weights inplace...")
    if is_checkpoint_format:
        # load weights from checkpoint/ original model format
        initialize_layerwise_reload(model)
        loaded_weights = model.load_weights(weights_iterator)
        finalize_layerwise_reload(model, model_config)

    else:
        # load weights from kernel format
        logger.warning_once(
            "Reloading with `is_checkpoint_format=True` requires that "
            "weights be in kernel format and already sharded",
        )
        loaded_weights = set()
        for name, loaded_weight in weights_iterator:
            param = get_parameter_for_reload(model, name)  # TODO: buffers?
            param.copy_(loaded_weight)
            loaded_weights.add(name)

    # logging and validation
    counter_after_reloading = time.perf_counter()
    diff_seconds = counter_after_reloading - counter_before_reloading
    logger.info_once(
        "Reloading and processing weights took %.2f seconds",
        diff_seconds,
    )
    if model_config.quantization is None and loaded_weights is not None:
        weights_not_loaded = weights_to_load - loaded_weights
        if weights_not_loaded:
            logger.warning(
                "Following weights were not loaded from checkpoint: %s",
                weights_not_loaded,
            )
