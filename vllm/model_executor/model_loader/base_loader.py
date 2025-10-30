# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import enum
from abc import ABC, abstractmethod

import huggingface_hub
import torch
import torch.nn as nn

from vllm.config import ModelConfig, VllmConfig
from vllm.config.load import LoadConfig
from vllm.logger import init_logger
from vllm.model_executor.model_loader.utils import (
    initialize_model,
    process_weights_after_loading,
)
from vllm.utils.torch_utils import set_default_torch_dtype
from vllm.validation.plugins import ModelType, ModelValidationPluginRegistry

logger = init_logger(__name__)


class DownloadType(int, enum.Enum):
    HUGGINGFACE_HUB = 1
    LOCAL_FILE = 2
    S3 = 3  # not currently supported
    UNKNOWN = 4


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

    def get_download_type(self, model_name_or_path: str) -> DownloadType | None:
        """Subclass must override this and return the download type it needs"""
        return None

    def download_all_files(
        self, model: nn.Module, model_config: ModelConfig, load_config: LoadConfig
    ) -> str | None:
        """Download all files. Ask the subclass for what type of download
        it does; Huggingface is used so often, so download all files here."""
        dt = self.get_download_type(model_config.model)
        if dt == DownloadType.HUGGINGFACE_HUB:
            return huggingface_hub.snapshot_download(
                model_config.model,
                allow_patterns=["*"],
                cache_dir=self.load_config.download_dir,
                revision=model_config.revision,
                local_files_only=huggingface_hub.constants.HF_HUB_OFFLINE,
            )
        elif dt == DownloadType.LOCAL_FILE:
            return model_config.model
        return None

    def validate_model(
        self, model: nn.Module, model_config: ModelConfig, load_config: LoadConfig
    ) -> None:
        """If needed, validate the model after downloading _all_ its files."""
        if ModelValidationPluginRegistry.model_validation_needed(
            ModelType.MODEL_TYPE_AI_MODEL, model_config.model
        ):
            folder = self.download_all_files(model, model_config, load_config)
            if folder is None:
                raise RuntimeError(
                    "Model validation could not be done due to "
                    "an unsupported download method."
                )
            ModelValidationPluginRegistry.validate_model(
                ModelType.MODEL_TYPE_AI_MODEL, folder, model_config.model
            )

    def load_model(
        self, vllm_config: VllmConfig, model_config: ModelConfig
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
                    vllm_config=vllm_config, model_config=model_config
                )

            logger.debug("Loading weights on %s ...", load_device)
            self.validate_model(model, model_config, vllm_config.load_config)
            # Quantization does not happen in `load_weights` but after it
            self.load_weights(model, model_config)
            process_weights_after_loading(model, model_config, target_device)
        return model.eval()
