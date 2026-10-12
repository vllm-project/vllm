# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch
import torch.nn as nn

from vllm.config import ModelConfig, VllmConfig
from vllm.config.load import LoadConfig
from vllm.model_executor.layers.quantization.kv_cache import KVCacheScaleParameter
from vllm.model_executor.model_loader.base_loader import BaseModelLoader
from vllm.model_executor.model_loader.utils import process_weights_after_loading
from vllm.utils.torch_utils import set_default_torch_dtype

META_DEVICE = torch.device("meta")


class MetaModelLoader(BaseModelLoader):
    """Model loader that builds the model on the `meta` device.

    Every parameter and buffer stays unmaterialized, so the model can be
    constructed from an HF config alone: no checkpoint is read and no
    accelerator memory is allocated. Weight loading is a no-op, which leaves
    parameter *values* undefined -- the model is only usable for shape
    propagation, e.g. the operator capture in `vllm.profiler.op_capture`.
    """

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
        pass  # Meta parameters are never materialized

    def load_model(
        self, vllm_config: VllmConfig, model_config: ModelConfig, prefix: str = ""
    ) -> nn.Module:
        with set_default_torch_dtype(model_config.dtype):
            with META_DEVICE:
                model = self.create_model(
                    vllm_config=vllm_config,
                    model_config=model_config,
                    prefix=prefix,
                )
            _restore_kv_cache_scales(model)
            # CPU, so the restored scales are not moved back to meta.
            process_weights_after_loading(model, model_config, torch.device("cpu"))
        return model.eval()


def _restore_kv_cache_scales(model: nn.Module) -> None:
    """Give KV-cache scales their "not in checkpoint" value back.

    Post-processing branches on these scalars, and on `meta` the -1.0
    sentinel is lost. Restoring it takes the same path as a checkpoint without
    KV-cache scales.
    """
    for module in model.modules():
        for name, param in list(module.named_parameters(recurse=False)):
            if isinstance(param, KVCacheScaleParameter) and param.is_meta:
                setattr(module, name, KVCacheScaleParameter())
