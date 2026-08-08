# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import glob
import os

import torch
import torch.nn as nn
from transformers.utils import SAFE_WEIGHTS_INDEX_NAME

from vllm.config import ModelConfig
from vllm.config.load import LoadConfig
from vllm.logger import init_logger
from vllm.model_executor.layers.quantization.base_config import QuantizeMethodBase
from vllm.model_executor.model_loader.base_loader import BaseModelLoader
from vllm.model_executor.model_loader.reload.layerwise import (
    _get_original_loader,
    get_layerwise_info,
)
from vllm.model_executor.model_loader.reload.meta import materialize_layer
from vllm.model_executor.model_loader.reload.probe import (
    probe_model_load,
    safetensors_meta_weights_from_files,
    validate_probe_plan_coverage,
)
from vllm.model_executor.model_loader.reload.types import LayerReloadingInfo
from vllm.model_executor.model_loader.reload.utils import get_layer_tensors
from vllm.model_executor.model_loader.weight_utils import (
    download_safetensors_index_file_from_hf,
    download_weights_from_hf,
    filter_duplicate_safetensors_files,
    initialize_dummy_weights,
    initialize_single_dummy_weight,
    maybe_download_from_modelscope,
)

logger = init_logger(__name__)


class DummyModelLoader(BaseModelLoader):
    """Model loader that will set model weights to random values."""

    def __init__(self, load_config: LoadConfig):
        super().__init__(load_config)
        extra_config = load_config.model_loader_extra_config
        if not isinstance(extra_config, dict):
            raise ValueError(
                "model_loader_extra_config must be a dict for load format "
                f"{load_config.load_format}, got {type(extra_config).__name__}"
            )
        unexpected_keys = set(extra_config) - {"enable_load_probe"}
        if unexpected_keys:
            raise ValueError(
                "Unexpected extra config keys for load format "
                f"{load_config.load_format}: {unexpected_keys}"
            )
        enable_load_probe = extra_config.get("enable_load_probe", False)
        if not isinstance(enable_load_probe, bool):
            raise ValueError(
                "enable_load_probe must be a bool, got "
                f"{type(enable_load_probe).__name__}"
            )
        self.enable_load_probe = enable_load_probe
        self._probe_weights_cache: list[tuple[str, torch.Tensor]] | None = None

    def download_model(self, model_config: ModelConfig) -> None:
        if self.enable_load_probe:
            self._probe_meta_weights(model_config)

    def load_weights(self, model: nn.Module, model_config: ModelConfig) -> None:
        supports_metadata_probe = True
        for layer in model.modules():
            info = get_layerwise_info(layer)
            if info.can_load():
                # Online quantization has already transformed the layer away
                # from the checkpoint layout. Its exact baseline must be
                # learned from the first real transfer instead.
                supports_metadata_probe = False
                self._process_online_quant_layer(layer, info)
            else:
                # NOTE(woosuk): For accurate performance evaluation, we assign
                # random values to the weights.
                initialize_dummy_weights(layer, model_config)

        weights = (
            self._probe_meta_weights(model_config)
            if self.enable_load_probe and supports_metadata_probe
            else []
        )
        if weights:
            report = probe_model_load(model, weights)
            report.raise_for_error()
            validate_probe_plan_coverage(model, report)
            logger.info(
                "Probed %d checkpoint sources through dummy model loaders "
                "without materializing weight data",
                len(weights),
            )

    def _probe_meta_weights(
        self,
        model_config: ModelConfig,
    ) -> list[tuple[str, torch.Tensor]]:
        if self._probe_weights_cache is not None:
            return self._probe_weights_cache
        model_name_or_path = (
            maybe_download_from_modelscope(
                model_config.model,
                revision=model_config.revision,
                download_dir=self.load_config.download_dir,
                ignore_patterns=self.load_config.ignore_patterns,
                allow_patterns=["*.safetensors"],
            )
            or model_config.model
        )
        if not os.path.isdir(model_name_or_path):
            hf_folder = download_weights_from_hf(
                model_name_or_path,
                self.load_config.download_dir,
                ["*.safetensors"],
                model_config.revision,
                ignore_patterns=self.load_config.ignore_patterns,
            )
            download_safetensors_index_file_from_hf(
                model_name_or_path,
                SAFE_WEIGHTS_INDEX_NAME,
                self.load_config.download_dir,
                revision=model_config.revision,
            )
        else:
            hf_folder = model_name_or_path

        filenames = sorted(glob.glob(os.path.join(hf_folder, "*.safetensors")))
        filenames = filter_duplicate_safetensors_files(
            filenames,
            hf_folder,
            SAFE_WEIGHTS_INDEX_NAME,
        )
        self._probe_weights_cache = safetensors_meta_weights_from_files(filenames)
        return self._probe_weights_cache

    def _process_online_quant_layer(
        self,
        layer: nn.Module,
        info: LayerReloadingInfo,
    ) -> None:
        """Materialize, apply dummy weights, and run quantization processing."""
        materialize_layer(layer, info)

        for tensor in get_layer_tensors(layer).values():
            initialize_single_dummy_weight(tensor)

        for param in get_layer_tensors(layer).values():
            param.weight_loader = _get_original_loader(param)

        quant_method = getattr(layer, "quant_method", None)
        if isinstance(quant_method, QuantizeMethodBase):
            quant_method.process_weights_after_loading(layer)

        info.reset()
