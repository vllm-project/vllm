# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The lifecycle shared by initial weight loading and checkpoint reloads."""

from weakref import WeakSet, ref

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
        initial_load_device: torch.device | None = None,
    ) -> None:
        self._model_ref = ref(model)
        self.initial_load_device = initial_load_device
        self._processed_quant: WeakSet[nn.Module] = WeakSet()
        self._processed_attention: WeakSet[nn.Module] = WeakSet()
        self._uses_layerwise_loading = initial_load_device is None or any(
            getattr(getattr(module, "quant_method", None), "uses_meta_device", False)
            for module in model.modules()
        )

    @property
    def model(self) -> nn.Module:
        model = self._model_ref()
        if model is None:
            raise RuntimeError("weight load model no longer exists")
        return model

    def prepare(self) -> None:
        model = self.model
        active_session = get_active_weight_load_session(model)
        if active_session is not None:
            raise RuntimeError("a weight load session is already active")
        model._vllm_weight_load_session = self

        try:
            if not self._uses_layerwise_loading:
                return

            from vllm.model_executor.model_loader.reload.layerwise import (
                _prepare_layerwise_loading,
                get_layerwise_info,
            )

            for module in model.modules():
                get_layerwise_info(module).load_session = self
            if self.initial_load_device is None:
                _prepare_layerwise_loading(model)
        except BaseException:
            self.abort()
            raise

    def finish(
        self,
        model_config: ModelConfig | None,
    ) -> None:
        model = self.model
        if get_active_weight_load_session(model) is not self:
            raise RuntimeError("weight load session was not prepared")
        try:
            if self._uses_layerwise_loading:
                from vllm.model_executor.model_loader.reload.layerwise import (
                    _finish_layerwise_loading,
                )

                act_dtype = None if model_config is None else model_config.dtype
                _finish_layerwise_loading(model, self, act_dtype)

            if self.initial_load_device is not None:
                if model_config is None:
                    raise ValueError("model_config is required for initial loading")
                from vllm.model_executor.model_loader.utils import (
                    process_weights_after_loading,
                )

                process_weights_after_loading(
                    model,
                    model_config,
                    self.initial_load_device,
                )
            else:
                self._process_hpc_modules()
        except BaseException:
            self.abort()
            raise
        else:
            self._unbind()

    def abort(self) -> None:
        """Discard layerwise loading state so a full update can be retried."""
        model = self._model_ref()
        if model is None or get_active_weight_load_session(model) is not self:
            return

        try:
            if self._uses_layerwise_loading:
                from vllm.model_executor.model_loader.reload.layerwise import (
                    _abort_layerwise_loading,
                )

                _abort_layerwise_loading(model)
        finally:
            self._unbind()

    def process_weights_after_loading(self, model_config: ModelConfig) -> None:
        """Process initial-load weights in dependency order."""
        for module in self.model.modules():
            self.process_quant(module)
        for module in self.model.modules():
            self.process_attention(module, model_config.dtype)
        self._process_hpc_modules()

        if model_config.quantization == "torchao":
            from vllm.model_executor.model_loader.reload import (
                set_torchao_reload_attrs,
            )

            set_torchao_reload_attrs(self.model, model_config)

    def _process_hpc_modules(self) -> None:
        from vllm.model_executor.layers.hpc import HpcModule

        for module in self.model.modules():
            if isinstance(module, HpcModule):
                module.process_weights_after_loading(self.model)

    def process_quant(self, module: nn.Module) -> None:
        if module in self._processed_quant:
            return
        if _process_quant_method(module, self.initial_load_device):
            self._processed_quant.add(module)

    def process_attention(self, module: nn.Module, act_dtype: torch.dtype) -> None:
        if module in self._processed_attention:
            return

        from vllm.model_executor.layers.attention import (
            Attention,
            MLAAttention,
            MMEncoderAttention,
        )

        if not isinstance(module, (Attention, MLAAttention, MMEncoderAttention)):
            return
        if self.initial_load_device is None:
            module.process_weights_after_loading(act_dtype)
        else:
            from vllm.model_executor.model_loader.utils import (
                device_loading_context,
            )

            with device_loading_context(module, self.initial_load_device):
                module.process_weights_after_loading(act_dtype)
        self._processed_attention.add(module)

    def _unbind(self) -> None:
        model = self._model_ref()
        if model is None:
            return
        if get_active_weight_load_session(model) is self:
            delattr(model, "_vllm_weight_load_session")


def get_active_weight_load_session(model: nn.Module) -> WeightLoadSession | None:
    return getattr(model, "_vllm_weight_load_session", None)


def _process_quant_method(
    module: nn.Module,
    target_device: torch.device | None = None,
) -> bool:
    quant_method = getattr(module, "quant_method", None)
    if not isinstance(quant_method, QuantizeMethodBase):
        return False

    if target_device is None:
        quant_method.process_weights_after_loading(module)
    else:
        from vllm.model_executor.model_loader.utils import device_loading_context

        with device_loading_context(module, target_device):
            quant_method.process_weights_after_loading(module)
        release_device_memory_under_pressure(target_device)
    return True
