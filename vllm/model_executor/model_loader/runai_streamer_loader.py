# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# ruff: noqa: SIM117
import os
from collections.abc import Generator
from typing import Optional

import torch
from torch import nn
from transformers.utils import SAFE_WEIGHTS_INDEX_NAME

from vllm.config import ModelConfig
from vllm.config.load import LoadConfig
from vllm.model_executor.model_loader.base_loader import BaseModelLoader
from vllm.model_executor.model_loader.weight_utils import (
    download_safetensors_index_file_from_hf, download_weights_from_hf,
    runai_safetensors_weights_iterator)
from vllm.transformers_utils.runai_utils import (is_runai_obj_uri,
                                                 list_safetensors)


class RunaiModelStreamerLoader(BaseModelLoader):
    """
        Model loader that can load safetensors
        files from local FS or S3 bucket.
    """

    def __init__(self, load_config: LoadConfig):
        super().__init__(load_config)
        if load_config.model_loader_extra_config:
            extra_config = load_config.model_loader_extra_config

            if ("concurrency" in extra_config
                    and isinstance(extra_config.get("concurrency"), int)):
                os.environ["RUNAI_STREAMER_CONCURRENCY"] = str(
                    extra_config.get("concurrency"))

            if ("memory_limit" in extra_config
                    and isinstance(extra_config.get("memory_limit"), int)):
                os.environ["RUNAI_STREAMER_MEMORY_LIMIT"] = str(
                    extra_config.get("memory_limit"))

            runai_streamer_s3_endpoint = os.getenv(
                'RUNAI_STREAMER_S3_ENDPOINT')
            aws_endpoint_url = os.getenv('AWS_ENDPOINT_URL')
            if (runai_streamer_s3_endpoint is None
                    and aws_endpoint_url is not None):
                os.environ["RUNAI_STREAMER_S3_ENDPOINT"] = aws_endpoint_url

    def _prepare_weights(self, model_name_or_path: str,
                         revision: Optional[str]) -> list[str]:
        """Prepare weights for the model.

        If the model is not local, it will be downloaded."""

        is_object_storage_path = is_runai_obj_uri(model_name_or_path)
        is_local = os.path.isdir(model_name_or_path)
        safetensors_pattern = "*.safetensors"
        index_file = SAFE_WEIGHTS_INDEX_NAME

        hf_folder = (model_name_or_path if (is_local or is_object_storage_path)
                     else download_weights_from_hf(
                         model_name_or_path,
                         self.load_config.download_dir,
                         [safetensors_pattern],
                         revision,
                         ignore_patterns=self.load_config.ignore_patterns,
                     ))
        hf_weights_files = list_safetensors(path=hf_folder)

        if not is_local and not is_object_storage_path:
            download_safetensors_index_file_from_hf(
                model_name_or_path, index_file, self.load_config.download_dir,
                revision)

        if not hf_weights_files:
            raise RuntimeError(
                f"Cannot find any safetensors model weights with "
                f"`{model_name_or_path}`")

        return hf_weights_files

    def _get_weights_iterator(
            self, model_or_path: str,
            revision: str) -> Generator[tuple[str, torch.Tensor], None, None]:
        """Get an iterator for the model weights based on the load format."""
        hf_weights_files = self._prepare_weights(model_or_path, revision)
        return runai_safetensors_weights_iterator(
            hf_weights_files,
            self.load_config.use_tqdm_on_load,
        )

    def download_model(self, model_config: ModelConfig) -> None:
        """Download model if necessary"""
        self._prepare_weights(model_config.model, model_config.revision)

    def load_weights(self, model: nn.Module,
                     model_config: ModelConfig) -> None:
        """Load weights into a model."""
        model_weights = model_config.model
        if hasattr(model_config, "model_weights"):
            model_weights = model_config.model_weights
        model.load_weights(
            self._get_weights_iterator(model_weights, model_config.revision))
