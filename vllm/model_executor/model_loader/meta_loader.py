# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch
import torch.nn as nn
from torch.overrides import TorchFunctionMode

from vllm.config import ModelConfig, VllmConfig
from vllm.model_executor.layers.quantization.kv_cache import KVCacheScaleParameter
from vllm.model_executor.model_loader.dummy_loader import DummyModelLoader
from vllm.model_executor.model_loader.reload.layerwise import get_layerwise_info


class _AcceleratorToMetaMode(TorchFunctionMode):
    """Redirect explicit accelerator placements to meta during construction.

    Explicit ``device=current_platform.device_type`` bypasses the default-device
    context; host (``cpu``) placements are left untouched.
    """

    def __torch_function__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        device = kwargs.get("device")
        if device is not None and torch.device(device).type not in ("cpu", "meta"):
            kwargs["device"] = torch.device("meta")
        return func(*args, **kwargs)


class MetaModelLoader(DummyModelLoader):
    """Model loader that builds the model on the meta device without weights.

    Parameters only carry shape and dtype (zero memory); nothing is downloaded
    or initialized. Meant for functional validation (e.g. ``VLLM_META_MODE``),
    where model outputs are meaningless.
    """

    def get_load_device(self, vllm_config: VllmConfig) -> torch.device:
        return torch.device("meta")

    def get_processing_device(self, target_device: torch.device) -> torch.device:
        # Restored scalar sentinels must remain readable during post-processing.
        return torch.device("cpu")

    def load_model(
        self, vllm_config: VllmConfig, model_config: ModelConfig, prefix: str = ""
    ) -> nn.Module:
        with _AcceleratorToMetaMode():
            return super().load_model(vllm_config, model_config, prefix)

    def load_weights(self, model: nn.Module, model_config: ModelConfig) -> None:
        from vllm import meta_mode

        # Model construction imports and registers model-specific custom ops.
        meta_mode.register_meta_impls()
        self._restore_kv_cache_scales(model)

        # Meta tensors hold no data, so only online-quant layers need work:
        # materialize them (back onto meta) and run quantization processing.
        for layer in model.modules():
            info = get_layerwise_info(layer)
            if info.can_load():
                self._process_online_quant_layer(layer, info)

    @staticmethod
    def _restore_kv_cache_scales(model: nn.Module) -> None:
        """Restore the readable sentinel used when a checkpoint has no scale."""
        for module in model.modules():
            for name, param in list(module.named_parameters(recurse=False)):
                if isinstance(param, KVCacheScaleParameter) and param.is_meta:
                    setattr(module, name, KVCacheScaleParameter())
