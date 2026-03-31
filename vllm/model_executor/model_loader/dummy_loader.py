# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch
import torch.nn as nn

from vllm.config import ModelConfig
from vllm.config.load import LoadConfig
from vllm.model_executor.model_loader.base_loader import BaseModelLoader
from vllm.model_executor.model_loader.reload.meta import materialize_meta_tensor
from vllm.model_executor.model_loader.reload.utils import get_layer_tensors
from vllm.model_executor.model_loader.weight_utils import initialize_dummy_weights


class DummyModelLoader(BaseModelLoader):
    """Model loader that will set model weights to random values."""

    def __init__(self, load_config: LoadConfig):
        super().__init__(load_config)
        if load_config.model_loader_extra_config:
            raise ValueError(
                f"Model loader extra config is not supported for "
                f"load format {load_config.load_format}"
            )

    def download_model(self, model_config: ModelConfig) -> None:
        pass  # Nothing to download

    def load_weights(self, model: nn.Module, model_config: ModelConfig) -> None:
        # materialize meta tensors as part of online quantization lifecycle
        for layer in model.modules():
            for name, param in get_layer_tensors(layer).items():
                if param.device == torch.device("meta"):
                    setattr(layer, name, materialize_meta_tensor(param))

        # NOTE(woosuk): For accurate performance evaluation, we assign
        # random values to the weights.
        initialize_dummy_weights(model, model_config)

        # Some models build derived weights from loaded parameters instead of
        # storing them in checkpoints. Rebuild those tensors for dummy load.
        for layer in model.modules():
            fuse_indexer_weights = getattr(layer, "fuse_indexer_weights", None)
            if callable(fuse_indexer_weights):
                fuse_indexer_weights()
