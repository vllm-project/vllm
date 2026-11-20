# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Generator
from contextlib import contextmanager
from contextvars import ContextVar

import torch

from vllm.model_executor.layers.quantization.base_config import QuantizeMethodBase
from vllm.model_executor.layers.quantization.utils.quant_utils import QuantKey
from vllm.model_executor.model_loader.reload.layerwise import (
    initialize_online_processing,
)
from vllm.model_executor.model_loader.reload.meta import to_meta_tensor

_requantization_source: ContextVar[bool] = ContextVar(
    "requantization_source", default=False
)


@contextmanager
def skip_checkpoint_backend_selection() -> Generator[None]:
    """Skip backend selection for checkpoint quantization methods in this scope.

    Covers checkpoint methods used as requantization sources (their weights are
    only dequantized) and unquantized checkpoint methods discarded when an online
    target replaces them, so oracles need not probe a compute backend.
    """
    token = _requantization_source.set(True)
    try:
        yield
    finally:
        _requantization_source.reset(token)


def is_requantization_source() -> bool:
    return _requantization_source.get()


class OnlineRequantizationMixin:
    """Shared lifecycle for converting serialized weights before quantization."""

    # Only the MXFP4 and FP8 per-channel (PTPC) online linear methods honor an
    # activation override for now.
    accepts_activation_override: bool = False

    @classmethod
    def resolve_activation_quant_key(
        cls, activation_quant_key: QuantKey | None, explicit: bool
    ) -> QuantKey | None:
        """Final activation key for a requested (or default) key."""
        # TODO: this should be removed and handled by oracles once oracles are
        # refactored.
        if explicit and not cls.accepts_activation_override:
            raise NotImplementedError(
                f"activation override is not supported for online {cls.__name__}"
            )
        if activation_quant_key is not None and activation_quant_key.scale.static:
            # Online quantization has no calibrated static activation scales.
            raise NotImplementedError(
                f"online {cls.__name__} cannot honour "
                f"activation={activation_quant_key}: static activation scales are "
                "not supported. Use a dynamic activation key, or 'activation: null'."
            )

        return activation_quant_key

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.requantization_source: QuantizeMethodBase | None = None
        self.requantization_source_parameters: dict[str, torch.nn.Parameter] = {}

    def set_requantization_source(self, source_method: QuantizeMethodBase) -> None:
        """Configure serialized-weight conversion before online quantization."""
        self.requantization_source = source_method

    def create_requantization_source_weights(
        self, layer: torch.nn.Module, *weight_args, **extra_weight_attrs
    ) -> bool:
        """Delegate weight creation to the source and track its parameters."""
        if self.requantization_source is None:
            return False

        existing_parameter_names = set(layer._parameters)
        # Schemes may select kernels in `create_weights`: skip them for sources.
        with skip_checkpoint_backend_selection():
            self.requantization_source.create_weights(
                layer, *weight_args, **extra_weight_attrs
            )
        self.requantization_source_parameters = {
            name: parameter
            for name, parameter in layer._parameters.items()
            if name not in existing_parameter_names and parameter is not None
        }
        for name, parameter in self.requantization_source_parameters.items():
            # `replace_parameter` then registers new params over these.
            parameter.requantization_source = True
            # The serialized weights live on the meta device and are materialized
            # one layer at a time by the layerwise processing loader: a layer is
            # requantized as soon as it is loaded, so the peak memory is the
            # quantized model plus one layer of serialized weights.
            meta_parameter = to_meta_tensor(parameter)
            layer.register_parameter(name, meta_parameter)
            self.requantization_source_parameters[name] = meta_parameter
        initialize_online_processing(layer)
        return True

    def release_requantization_source_weights(self, layer: torch.nn.Module) -> None:
        """Release serialized parameters after successful requantization."""
        for name in self.requantization_source_parameters:
            # Materialization re-creates the parameter, keeping its attributes.
            parameter = layer._parameters.get(name)
            if parameter is not None and getattr(
                parameter, "requantization_source", False
            ):
                delattr(layer, name)
        self.requantization_source_parameters.clear()
