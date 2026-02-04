# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from abc import ABC, abstractmethod

import torch
import torch.nn as nn

import vllm.envs as envs
from vllm.config import ModelConfig, VllmConfig
from vllm.config.load import LoadConfig
from vllm.logger import init_logger
from vllm.model_executor.model_loader.reload import (
    finalize_layerwise_reload,
    initialize_layerwise_reload,
)
from vllm.model_executor.model_loader.utils import (
    initialize_model,
    process_weights_after_loading,
)
from vllm.platforms import current_platform
from vllm.tracing import instrument
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

    @instrument(span_name="Load model")
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

            use_layerwise_loading = _get_use_layerwise_loading(model, self)

            if use_layerwise_loading:
                # set up layer loading
                initialize_layerwise_reload(
                    model, is_reload=False, target_device=load_device
                )
                # load weights, quantization via each layer's
                # `process_weights_after_loading` will happen for each layer
                # as soon as all of that layer's weights are loaded
                self.load_weights(model, model_config)
                # finalize layer reloading
                finalize_layerwise_reload(model, model_config, is_reload=False)

            else:
                # Load weights to model format
                self.load_weights(model, model_config)
                # For layers with quantization, convert to kernel format
                with target_device:
                    process_weights_after_loading(model, model_config, target_device)

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


def _get_use_layerwise_loading(
    model: torch.nn.Module,
    model_loader: BaseModelLoader,
) -> bool:
    from vllm.model_executor.model_loader.dummy_loader import DummyModelLoader
    from vllm.model_executor.model_loader.utils import (
        model_has_any_online_quant_with_device_meta,
    )

    has_online_quant = model_has_any_online_quant_with_device_meta(model)

    is_dummy_loader = isinstance(model_loader, DummyModelLoader)
    return has_online_quant and not is_dummy_loader


def log_model_inspection(model: nn.Module) -> None:
    """Log model structure if VLLM_LOG_MODEL_INSPECTION=1."""
    if not envs.VLLM_LOG_MODEL_INSPECTION:
        return

    from vllm.model_inspection import format_model_inspection

    logger.info("vLLM model structure:\n%s", format_model_inspection(model))
