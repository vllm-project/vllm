# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from abc import ABC, abstractmethod

import torch
import torch.nn as nn

import vllm.envs as envs
from vllm.config import ModelConfig, VllmConfig
from vllm.config.load import LoadConfig
from vllm.logger import init_logger
from vllm.model_executor.model_loader.reload.initial_load import (
    finalize_layerwise_initial_load,
    initialize_layerwise_initial_load,
)
from vllm.model_executor.model_loader.utils import (
    initialize_model,
    process_weights_after_loading,
)
from vllm.platforms import current_platform
from vllm.utils.mem_utils import format_gib
from vllm.utils.torch_utils import set_default_torch_dtype

logger = init_logger(__name__)


class BaseModelLoader(ABC):
    """Base class for model loaders."""

    def __init__(self, load_config: LoadConfig):
        self.load_config = load_config

    @abstractmethod
    def download_model(self, model_config: ModelConfig) -> None:
        """Download a model so that it can be immediately loaded."""
        raise NotImplementedError

    @abstractmethod
    def load_weights(self, model: nn.Module, model_config: ModelConfig) -> None:
        """Load weights into a model. This standalone API allows
        inplace weights loading for an already-initialized model"""
        raise NotImplementedError

    def load_model(
        self, vllm_config: VllmConfig, model_config: ModelConfig, prefix: str = ""
    ) -> nn.Module:
        """Load a model with the given configurations."""
        device_config = vllm_config.device_config
        load_config = vllm_config.load_config
        load_device = (
            device_config.device if load_config.device is None else load_config.device
        )
        target_device = torch.device(load_device)
        with set_default_torch_dtype(model_config.dtype):
            with target_device:
                model = initialize_model(
                    vllm_config=vllm_config, model_config=model_config, prefix=prefix
                )

            log_model_inspection(model)

            logger.debug("Loading weights on %s ...", load_device)

            is_online_quant = _is_online_quant(vllm_config, model_config)
            if not is_online_quant:
                # Regular path, `process_weights_after_loading` is called
                # after all weights are loaded.

                # Quantization does not happen in `load_weights` but after it
                self.load_weights(model, model_config)
                process_weights_after_loading(model, model_config, target_device)

            else:
                # Online quantization can take the layerwise loading path
                # where `process_weights_after_loading` is done just-in-time
                # after all of a layer's weights are loaded.

                # set up weight loaders for layerwise loading with
                # streaming post-processing
                initialize_layerwise_initial_load(model, target_device)

                # load the weights, layerwise loading infra will call
                # each layer's `process_weights_after_loading` function
                # as soon as every weight of that layer is loaded
                self.load_weights(model, model_config)

                # finalize layerwise reloading (call any post-processing
                # that did not happen in real time)
                finalize_layerwise_initial_load(model, model_config)

                # Log peak GPU memory after loading weights. This is needed
                # to have test coverage on peak memory for online quantization.
                if current_platform.is_cuda():
                    peak_memory = torch.cuda.max_memory_allocated()
                    logger.debug_once(
                        "Peak GPU memory after loading weights: %s GiB",
                        format_gib(peak_memory),
                        scope="local",
                    )

        return model.eval()


def log_model_inspection(model: nn.Module) -> None:
    """Log model structure if VLLM_LOG_MODEL_INSPECTION=1."""
    if not envs.VLLM_LOG_MODEL_INSPECTION:
        return

    from vllm.model_inspection import format_model_inspection

    logger.info("vLLM model structure:\n%s", format_model_inspection(model))


def _is_online_quant(vllm_config: VllmConfig, model_config: ModelConfig) -> bool:
    quant_config = vllm_config.quant_config
    return (
        # TODO(future): add other online quant paths here
        model_config.quantization == "fp8"
        and quant_config is not None
        and hasattr(quant_config, "is_checkpoint_fp8_serialized")
        and not quant_config.is_checkpoint_fp8_serialized
    )
