# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch
import torch.nn as nn

from vllm.config import ModelConfig
from vllm.config.load import LoadConfig
from vllm.model_executor.layers.quantization.base_config import QuantizeMethodBase
from vllm.model_executor.model_loader.base_loader import BaseModelLoader
from vllm.model_executor.model_loader.reload.layerwise import (
    _get_original_loader,
    get_layerwise_info,
)
from vllm.model_executor.model_loader.reload.meta import materialize_layer
from vllm.model_executor.model_loader.reload.types import LayerReloadingInfo
from vllm.model_executor.model_loader.reload.utils import get_layer_tensors
from vllm.model_executor.model_loader.weight_utils import (
    initialize_single_dummy_weight,
)


class DummyModelLoader(BaseModelLoader):
    """Model loader that will set model weights to random values."""

    def __init__(self, load_config: LoadConfig):
        super().__init__(load_config)
        extra_config = dict(load_config.model_loader_extra_config)
        # Half-width of the uniform init for 16/32-bit float weights. The
        # default (1e-3) keeps activations so small that outputs barely depend
        # on the input; tests that compare outputs can widen it.
        self.dummy_weight_range: float | None = extra_config.pop(
            "dummy_weight_range", None
        )
        if extra_config:
            raise ValueError(
                f"Unsupported model loader extra config {sorted(extra_config)} "
                f"for load format {load_config.load_format}; only "
                "'dummy_weight_range' is supported"
            )

    def _initialize(self, tensor: torch.Tensor, name: str) -> None:
        bound = self.dummy_weight_range
        if (
            bound is not None
            and tensor.is_floating_point()
            and torch.finfo(tensor.dtype).bits >= 16
        ):
            initialize_single_dummy_weight(tensor, low=-bound, high=bound, name=name)
        else:
            initialize_single_dummy_weight(tensor, name=name)

    def download_model(self, model_config: ModelConfig) -> None:
        pass  # Nothing to download

    def load_weights(self, model: nn.Module, model_config: ModelConfig) -> None:
        for layer in model.modules():
            info = get_layerwise_info(layer)
            if info.can_load():
                self._process_online_quant_layer(layer, info)
            else:
                # NOTE(woosuk): For accurate performance evaluation, we assign
                # random values to the weights.
                # Initialize only local tensors: state_dict() would recursively
                # initialize descendants that model.modules() also visits.
                # Exclude nonpersistent buffers, matching state_dict().
                for name, tensor in get_layer_tensors(layer).items():
                    if name not in layer._non_persistent_buffers_set:
                        self._initialize(tensor, name)

    def _process_online_quant_layer(
        self,
        layer: nn.Module,
        info: LayerReloadingInfo,
    ) -> None:
        """Materialize, apply dummy weights, and run quantization processing."""
        materialize_layer(layer, info)

        for name, tensor in get_layer_tensors(layer).items():
            self._initialize(tensor, name)

        for param in get_layer_tensors(layer).values():
            param.weight_loader = _get_original_loader(param)

        quant_method = getattr(layer, "quant_method", None)
        if isinstance(quant_method, QuantizeMethodBase):
            quant_method.process_weights_after_loading(layer)

        info.reset()
