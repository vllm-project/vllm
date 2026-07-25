# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import importlib

from torch import nn

from vllm.config import ModelConfig, VllmConfig
from vllm.config.load import LoadConfig
from vllm.model_executor.model_loader.base_loader import BaseModelLoader
from vllm.model_executor.model_loader.reload.layerwise import (
    finalize_load_recording,
    record_external_tensor_manifest,
    record_load_consumption,
)
from vllm.tracing import instrument

_MODELEXPRESS_LOADER_MODULE = "modelexpress.engines.vllm.loader"
_MISSING_MODELEXPRESS_MODULES = frozenset(
    {
        "modelexpress",
        "modelexpress.engines",
        "modelexpress.engines.vllm",
        _MODELEXPRESS_LOADER_MODULE,
    }
)


def _missing_modelexpress_error() -> ImportError:
    return ImportError(
        "The 'modelexpress' load format requires the ModelExpress Python package. "
        "Install it with `pip install modelexpress`."
    )


class ModelExpressModelLoader(BaseModelLoader):
    """Thin vLLM loader wrapper for ModelExpress."""

    def __init__(self, load_config: LoadConfig):
        super().__init__(load_config)
        self._loader = self._load_modelexpress_loader(load_config)

    @staticmethod
    def _load_modelexpress_loader(load_config: LoadConfig) -> BaseModelLoader:
        try:
            module = importlib.import_module(_MODELEXPRESS_LOADER_MODULE)
        except ModuleNotFoundError as exc:
            if exc.name not in _MISSING_MODELEXPRESS_MODULES:
                raise
            raise _missing_modelexpress_error() from exc

        ModelExpressVllmLoader = module.MxModelLoader
        return ModelExpressVllmLoader(load_config)

    def download_model(self, model_config: ModelConfig) -> None:
        self._loader.download_model(model_config)

    def load_weights(self, model: nn.Module, model_config: ModelConfig) -> None:
        record_load_consumption(model)
        try:
            self._loader.load_weights(model, model_config)
        finally:
            finalize_load_recording(model)

    @instrument(span_name="Load model")
    def load_model(
        self,
        vllm_config: VllmConfig,
        model_config: ModelConfig,
        prefix: str = "",
    ) -> nn.Module:
        model = self._loader.load_model(
            vllm_config=vllm_config,
            model_config=model_config,
            prefix=prefix,
        )
        tensors = getattr(self._loader, "tensors", {})
        has_parameters = any(True for _ in model.parameters())
        if not tensors and has_parameters:
            # ModelExpress skips its registry walk when no metadata/NIXL
            # publication path is configured. Manifest construction still
            # needs the same final post-PWAL tensor set.
            from modelexpress.tensor_utils import collect_module_tensors

            tensors = collect_module_tensors(model)
        if tensors:
            record_external_tensor_manifest(
                model,
                tensors,
                source_scheme="modelexpress",
            )
        elif has_parameters:
            raise RuntimeError(
                "ModelExpress returned a parameterized model without its "
                "published tensor registry; a complete load manifest cannot "
                "be established."
            )
        return model.eval()
