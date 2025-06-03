# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: SIM117
import copy
from collections.abc import Generator
from typing import Union

import torch
from torch import nn

from vllm.config import LoadConfig, ModelConfig, ParallelConfig, VllmConfig
from vllm.logger import init_logger
from vllm.model_executor.model_loader.base_loader import BaseModelLoader
from vllm.model_executor.model_loader.tensorizer import (
    TensorizerConfig, deserialize_tensorizer_model, init_tensorizer_model,
    is_vllm_tensorized, serialize_vllm_model, tensorizer_weights_iterator)
from vllm.model_executor.model_loader.utils import (get_model_architecture,
                                                    initialize_model,
                                                    set_default_torch_dtype)

logger = init_logger(__name__)


class TensorizerLoader(BaseModelLoader):
    """Model loader using CoreWeave's tensorizer library."""

    def __init__(self, load_config: LoadConfig):
        super().__init__(load_config)
        if isinstance(load_config.model_loader_extra_config, TensorizerConfig):
            self.tensorizer_config = load_config.model_loader_extra_config
        else:
            self.tensorizer_config = TensorizerConfig(
                **load_config.model_loader_extra_config)

    def _verify_config(self, model_config: ModelConfig,
                       parallel_config: ParallelConfig):
        self.tensorizer_config.verify_with_model_config(model_config)
        self.tensorizer_config.verify_with_parallel_config(parallel_config)

    def _get_weights_iterator(
        self, ) -> Generator[tuple[str, torch.Tensor], None, None]:
        tensorizer_args = self.tensorizer_config._construct_tensorizer_args()
        return tensorizer_weights_iterator(tensorizer_args)

    def _load_model_serialized_cpu(
        self,
        vllm_config: VllmConfig,
    ) -> nn.Module:
        """Load a serialized model with tensorizer to the CPU.

        This is only necessary when the model isn't vLLM-tensorized (see
        examples/others/tensorize_vllm_model.py) This should still
        be faster than default HuggingFace loading, but will be slower than
        loading a vLLM-tensorized model.
        """
        device_config = vllm_config.device_config
        model_config = vllm_config.model_config
        with set_default_torch_dtype(model_config.dtype):
            with torch.device(device_config.device):
                model = initialize_model(vllm_config=vllm_config)

            model.load_weights(self._get_weights_iterator())
        return model.eval()

    def download_model(self, model_config: ModelConfig) -> None:
        self.tensorizer_config.verify_with_model_config(model_config)

        with self.tensorizer_config.open_stream():
            pass

    def _patch_tensorizer_config(
            self, model_config: ModelConfig) -> TensorizerConfig:
        model_class = get_model_architecture(model_config)[0]
        tensorizer_config = copy.copy(self.tensorizer_config)
        tensorizer_config.model_class = model_class
        tensorizer_config.hf_config = model_config.hf_config
        tensorizer_config.dtype = model_config.dtype
        return tensorizer_config

    def load_weights(self, model: nn.Module,
                     model_config: ModelConfig) -> None:
        """Load serialized model weights with tensorizer.

        Expects a vLLM-tensorized model. See the
        examples/others/tensorize_vllm_model.py example script
        for serializing vLLM models."""
        if is_vllm_tensorized(self.tensorizer_config):
            tensorizer_config = self._patch_tensorizer_config(model_config)
            deserialize_tensorizer_model(model, tensorizer_config)
        else:
            model.load_weights(self._get_weights_iterator())

    def load_model(self, vllm_config: VllmConfig,
                   model_config: ModelConfig) -> nn.Module:
        parallel_config = vllm_config.parallel_config
        self._verify_config(model_config, parallel_config)

        if parallel_config.tensor_parallel_size > 1:
            from vllm.distributed import get_tensor_model_parallel_rank

            self.tensorizer_config.tensorizer_uri = (
                self.tensorizer_config.tensorizer_uri %
                get_tensor_model_parallel_rank())

        if is_vllm_tensorized(self.tensorizer_config):
            tensorizer_config = self._patch_tensorizer_config(model_config)
            model = init_tensorizer_model(tensorizer_config=tensorizer_config,
                                          vllm_config=vllm_config)
            self.load_weights(model, model_config)
            return model
        return self._load_model_serialized_cpu(vllm_config=vllm_config)

    @staticmethod
    def save_model(
        model: torch.nn.Module,
        tensorizer_config: Union[TensorizerConfig, dict],
    ) -> None:
        if isinstance(tensorizer_config, dict):
            tensorizer_config = TensorizerConfig(**tensorizer_config)
        serialize_vllm_model(
            model=model,
            tensorizer_config=tensorizer_config,
        )
