# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch.nn as nn

from vllm.config import ModelConfig
from vllm.config.load import LoadConfig
from vllm.logger import init_logger
from vllm.model_executor.layers.quantization.base_config import QuantizeMethodBase
from vllm.model_executor.model_loader.base_loader import BaseModelLoader
from vllm.model_executor.model_loader.reload.layerwise import (
    _get_original_loader,
    get_layerwise_info,
    record_dummy_load_manifest,
)
from vllm.model_executor.model_loader.reload.meta import materialize_layer
from vllm.model_executor.model_loader.reload.probe import (
    probe_model_load,
    safetensors_meta_weights,
    validate_probe_receipt_coverage,
)
from vllm.model_executor.model_loader.reload.types import LayerReloadingInfo
from vllm.model_executor.model_loader.reload.utils import get_layer_tensors
from vllm.model_executor.model_loader.weight_utils import (
    initialize_dummy_weights,
    initialize_single_dummy_weight,
)

logger = init_logger(__name__)


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
            safetensors_meta_weights(model_config.model)
            if supports_metadata_probe
            else []
        )
        if weights:
            report = probe_model_load(model, weights)
            report.raise_for_error()
            validate_probe_receipt_coverage(model, report)
            logger.info(
                "Probed %d checkpoint sources through dummy model loaders "
                "without materializing weight data",
                len(weights),
            )

    def finalize_load_manifest(self, model: nn.Module) -> None:
        exact_events = sum(
            len(get_layerwise_info(layer).required_keys or ())
            for layer in model.modules()
        )
        if exact_events:
            return
        record_dummy_load_manifest(model)

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
