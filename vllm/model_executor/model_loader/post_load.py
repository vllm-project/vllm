# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The lifecycle shared by initial weight loading and checkpoint reloads."""

import torch
from torch import nn

from vllm.config import ModelConfig
from vllm.model_executor.layers.quantization.base_config import (
    QuantizeMethodBase,
)
from vllm.utils.mem_utils import release_device_memory_under_pressure


class WeightLoadSession:
    """Prepare, load, then finish one checkpoint-format weight update."""

    def __init__(
        self,
        model: nn.Module,
        model_config: ModelConfig,
        target_device: torch.device | None = None,
    ) -> None:
        self.model = model
        self.model_config = model_config
        self.target_device = target_device
        self._processed_quant: set[nn.Module] = set()
        self._processed_attention: set[nn.Module] = set()
        self._uses_layerwise_loading = target_device is None or any(
            getattr(getattr(module, "quant_method", None), "uses_meta_device", False)
            for module in model.modules()
        )

    def prepare(self) -> None:
        if not self._uses_layerwise_loading:
            return

        from vllm.model_executor.model_loader.reload.layerwise import (
            _prepare_layerwise_loading,
            get_layerwise_info,
        )

        for module in self.model.modules():
            get_layerwise_info(module).load_session = self
        if self.target_device is None:
            _prepare_layerwise_loading(self.model)

    def finish(self) -> None:
        if self._uses_layerwise_loading:
            from vllm.model_executor.model_loader.reload.layerwise import (
                _finish_layerwise_loading,
            )

            _finish_layerwise_loading(self.model, self)

        if self.target_device is not None:
            for module in self.model.modules():
                self.process_quant(module)
            for module in self.model.modules():
                self.process_attention(module)

            if self.model_config.quantization == "torchao":
                from vllm.model_executor.model_loader.reload import (
                    set_torchao_reload_attrs,
                )

                set_torchao_reload_attrs(self.model, self.model_config)

        from vllm.model_executor.layers.hpc import HpcModule

        for module in self.model.modules():
            if isinstance(module, HpcModule):
                module.process_weights_after_loading(self.model)

    def process_quant(self, module: nn.Module) -> None:
        if module in self._processed_quant:
            return
        quant_method = getattr(module, "quant_method", None)
        if not isinstance(quant_method, QuantizeMethodBase):
            return

        if self.target_device is None:
            quant_method.process_weights_after_loading(module)
        else:
            from vllm.model_executor.model_loader.utils import (
                device_loading_context,
            )

            with device_loading_context(module, self.target_device):
                quant_method.process_weights_after_loading(module)
            release_device_memory_under_pressure(self.target_device)
        self._processed_quant.add(module)

    def process_attention(self, module: nn.Module) -> None:
        if module in self._processed_attention:
            return

        from vllm.model_executor.layers.attention import (
            Attention,
            MLAAttention,
            MMEncoderAttention,
        )

        if not isinstance(module, (Attention, MLAAttention, MMEncoderAttention)):
            return
        if self.target_device is None:
            module.process_weights_after_loading(self.model_config.dtype)
        else:
            from vllm.model_executor.model_loader.utils import (
                device_loading_context,
            )

            with device_loading_context(module, self.target_device):
                module.process_weights_after_loading(self.model_config.dtype)
        self._processed_attention.add(module)
