# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch.nn as nn

from vllm.config import ModelConfig
from vllm.config.load import LoadConfig
from vllm.model_executor.model_loader.base_loader import BaseModelLoader
from vllm.model_executor.model_loader.reload.online import (
    fill_online_layer,
    has_online_buffers,
)
from vllm.model_executor.model_loader.reload.utils import get_layer_tensors
from vllm.model_executor.model_loader.weight_utils import (
    initialize_single_dummy_weight,
)


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
        for layer in model.modules():
            # NOTE(woosuk): For accurate performance evaluation, we assign
            # random values to the weights.
            # Initialize only local tensors: state_dict() would recursively
            # initialize descendants that model.modules() also visits.
            # Exclude nonpersistent buffers, matching state_dict().
            def fill(layer: nn.Module = layer) -> None:
                for name, tensor in get_layer_tensors(layer).items():
                    if name not in layer._non_persistent_buffers_set:
                        initialize_single_dummy_weight(tensor, name=name)

            if has_online_buffers(layer):
                # online quantization: dummy checkpoint weights, then quantize
                fill_online_layer(layer, fill)
            else:
                fill()
