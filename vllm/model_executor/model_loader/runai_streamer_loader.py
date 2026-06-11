# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import fnmatch
import os
from collections.abc import Generator

import torch
from torch import nn
from transformers.utils import SAFE_WEIGHTS_INDEX_NAME

from vllm.config import ModelConfig
from vllm.config.load import LoadConfig
from vllm.model_executor.model_loader.base_loader import BaseModelLoader
from vllm.model_executor.model_loader.weight_utils import (
    download_safetensors_index_file_from_hf,
    download_weights_from_hf,
    filter_duplicate_safetensors_files,
    runai_safetensors_weights_iterator,
)
from vllm.transformers_utils.repo_utils import is_mistral_model_repo
from vllm.transformers_utils.runai_utils import is_runai_obj_uri, list_safetensors

MISTRAL_SAFETENSORS_PATTERN = "consolidated*.safetensors"
MISTRAL_SAFETENSORS_INDEX_NAME = "consolidated.safetensors.index.json"


class RunaiModelStreamerLoader(BaseModelLoader):
    """
    Model loader that can load safetensors
    files from local FS, S3, GCS, or Azure Blob Storage.
    """

    def __init__(self, load_config: LoadConfig):
        super().__init__(load_config)

        self._is_distributed: bool = False
        if load_config.model_loader_extra_config:
            extra_config = load_config.model_loader_extra_config

            if isinstance(distributed := extra_config.get("distributed"), bool):
                self._is_distributed = distributed
            if isinstance(concurrency := extra_config.get("concurrency"), int):
                os.environ["RUNAI_STREAMER_CONCURRENCY"] = str(concurrency)
            if isinstance(memory_limit := extra_config.get("memory_limit"), int):
                os.environ["RUNAI_STREAMER_MEMORY_LIMIT"] = str(memory_limit)

            runai_streamer_s3_endpoint = os.getenv("RUNAI_STREAMER_S3_ENDPOINT")
            aws_endpoint_url = os.getenv("AWS_ENDPOINT_URL")
            if runai_streamer_s3_endpoint is None and aws_endpoint_url is not None:
                os.environ["RUNAI_STREAMER_S3_ENDPOINT"] = aws_endpoint_url

    @staticmethod
    def _matches_any_pattern(file_name: str, allow_patterns: list[str]) -> bool:
        return any(
            fnmatch.fnmatch(os.path.basename(file_name), pattern)
            for pattern in allow_patterns
        )

    def _uses_mistral_weight_format(
        self,
        model_name_or_path: str,
        model_config: ModelConfig,
        safetensors_files: list[str] | None = None,
    ) -> bool:
        if model_config.config_format == "mistral":
            return True

        if is_runai_obj_uri(model_name_or_path):
            if safetensors_files is None:
                safetensors_files = list_safetensors(path=model_name_or_path)
            return any(
                self._matches_any_pattern(weight_file, [MISTRAL_SAFETENSORS_PATTERN])
                for weight_file in safetensors_files
            )

        return is_mistral_model_repo(
            model_name_or_path,
            revision=model_config.revision,
        )

    def _get_safetensors_patterns(
        self,
        model_name_or_path: str,
        model_config: ModelConfig,
        safetensors_files: list[str] | None = None,
    ) -> tuple[list[str], str]:
        if self._uses_mistral_weight_format(
            model_name_or_path,
            model_config,
            safetensors_files,
        ):
            return [MISTRAL_SAFETENSORS_PATTERN], MISTRAL_SAFETENSORS_INDEX_NAME

        return ["*.safetensors"], SAFE_WEIGHTS_INDEX_NAME

    def _prepare_weights(
        self, model_name_or_path: str, model_config: ModelConfig
    ) -> list[str]:
        """Prepare weights for the model.

        If the model is not local, it will be downloaded."""

        revision = model_config.revision
        is_object_storage_path = is_runai_obj_uri(model_name_or_path)
        is_local = os.path.isdir(model_name_or_path)
        safetensors_files = (
            list_safetensors(path=model_name_or_path)
            if is_object_storage_path
            else None
        )
        allow_patterns, index_file = self._get_safetensors_patterns(
            model_name_or_path, model_config, safetensors_files
        )

        hf_folder = (
            model_name_or_path
            if (is_local or is_object_storage_path)
            else download_weights_from_hf(
                model_name_or_path,
                self.load_config.download_dir,
                allow_patterns,
                revision,
                ignore_patterns=self.load_config.ignore_patterns,
            )
        )
        if safetensors_files is None:
            safetensors_files = list_safetensors(path=hf_folder)

        hf_weights_files = [
            weight_file
            for weight_file in safetensors_files
            if self._matches_any_pattern(weight_file, allow_patterns)
        ]

        if not is_local and not is_object_storage_path:
            download_safetensors_index_file_from_hf(
                model_name_or_path, index_file, self.load_config.download_dir, revision
            )

        if not is_object_storage_path:
            hf_weights_files = filter_duplicate_safetensors_files(
                hf_weights_files, hf_folder, index_file
            )

        if not hf_weights_files:
            raise RuntimeError(
                f"Cannot find any safetensors model weights with `{model_name_or_path}`"
            )

        return hf_weights_files

    def _get_weights_iterator(
        self, model_or_path: str, model_config: ModelConfig
    ) -> Generator[tuple[str, torch.Tensor], None, None]:
        """Get an iterator for the model weights based on the load format."""
        hf_weights_files = self._prepare_weights(model_or_path, model_config)
        return runai_safetensors_weights_iterator(
            hf_weights_files, self.load_config.use_tqdm_on_load, self._is_distributed
        )

    def download_model(self, model_config: ModelConfig) -> None:
        """Download model if necessary"""
        model_weights = model_config.model_weights or model_config.model
        self._prepare_weights(model_weights, model_config)

    def load_weights(self, model: nn.Module, model_config: ModelConfig) -> None:
        """Load weights into a model."""
        model_weights = model_config.model
        if model_weights_override := model_config.model_weights:
            model_weights = model_weights_override
        model.load_weights(self._get_weights_iterator(model_weights, model_config))
