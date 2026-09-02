# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Generator, Iterable
from typing import TYPE_CHECKING, cast

import torch
import torch.nn as nn
from safetensors.torch import _TYPES as _SAFETENSORS_TO_TORCH_DTYPE

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
from vllm.model_executor.model_loader.reload.types import LayerReloadingInfo
from vllm.model_executor.model_loader.reload.utils import get_layer_tensors
from vllm.model_executor.model_loader.utils import validate_weights_loading
from vllm.model_executor.model_loader.weight_utils import (
    initialize_dummy_weights,
    initialize_single_dummy_weight,
)
from vllm.transformers_utils.config import get_safetensors_params_metadata

if TYPE_CHECKING:
    from .default_loader import DefaultModelLoader

logger = init_logger(__name__)


class _SkipValidation(Exception):
    def __init__(self, reason: str) -> None:
        super().__init__(reason)

        self.reason = reason


class DummyModelLoader(BaseModelLoader):
    """Model loader that will set model weights to random values."""

    def __init__(self, load_config: LoadConfig):
        super().__init__(load_config)

        extra_config = load_config.model_loader_extra_config
        if not isinstance(extra_config, dict):
            raise ValueError(
                f"model_loader_extra_config must be a dict for load format "
                f"{load_config.load_format}, got {type(extra_config).__name__}"
            )

        self.enable_weights_track: bool | None = extra_config.get(
            "enable_weights_track", None
        )

    def download_model(self, model_config: ModelConfig) -> None:
        pass  # Nothing to download

    def _get_weights_iterator(
        self, source: "DefaultModelLoader.Source"
    ) -> Generator[tuple[str, torch.Tensor], None, None]:
        metadata = get_safetensors_params_metadata(
            source.model_or_path,
            revision=source.revision,
        )
        if not metadata:
            raise _SkipValidation("Missing safetensors metadata")

        for name, info in metadata.items():
            if "dtype" not in info:
                raise _SkipValidation(f"Missing safetensors dtype metadata for {name=}")

            dtype = info["dtype"]
            if dtype not in _SAFETENSORS_TO_TORCH_DTYPE:
                raise _SkipValidation(
                    f"Unrecognized safetensors dtype metadata for {name=}: {dtype=}"
                )

            yield (
                name,
                torch.empty(
                    info["shape"],
                    dtype=_SAFETENSORS_TO_TORCH_DTYPE[dtype],
                    device="meta",
                ),
            )

    def get_all_weights(
        self,
        model_config: ModelConfig,
        model: nn.Module,
    ) -> Generator[tuple[str, torch.Tensor], None, None]:
        from .default_loader import DefaultModelLoader

        primary_weights = DefaultModelLoader.Source(
            model_config.model,
            model_config.revision,
            prefix="",
            fall_back_to_pt=getattr(model, "fall_back_to_pt_during_load", True),
            allow_patterns_overrides=getattr(model, "allow_patterns_overrides", None),
        )
        yield from self._get_weights_iterator(primary_weights)

        secondary_weights = cast(
            Iterable[DefaultModelLoader.Source],
            getattr(model, "secondary_weights", ()),
        )
        for source in secondary_weights:
            yield from self._get_weights_iterator(source)

    def validate_weights(self, model: nn.Module, model_config: ModelConfig) -> None:
        """
        Imitate `DefaultModelLoader.load_weights` so we can use dummy weights
        to validate the weight mapping.
        """
        loaded_weights = model.load_weights(self.get_all_weights(model_config, model))

        default_enable_weights_track = (
            model_config.quantization is None and loaded_weights is not None
        )
        enable_weights_track = (
            self.enable_weights_track
            if self.enable_weights_track is not None
            else default_enable_weights_track
        )
        if enable_weights_track:
            validate_weights_loading(model, loaded_weights)

    def load_weights(self, model: nn.Module, model_config: ModelConfig) -> None:
        try:
            self.validate_weights(model, model_config)
        except _SkipValidation as e:
            logger.info(
                "Skipping validation when loading dummy weights for %s. Reason: %s",
                model_config.model,
                e.reason,
            )

        for layer in model.modules():
            info = get_layerwise_info(layer)
            if info.can_load():
                self._process_online_quant_layer(layer, info)
            else:
                # NOTE(woosuk): For accurate performance evaluation, we assign
                # random values to the weights.
                initialize_dummy_weights(layer, model_config)

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
