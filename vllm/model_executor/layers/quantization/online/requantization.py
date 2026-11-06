# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar

import torch

from vllm.model_executor.layers.quantization.base_config import QuantizeMethodBase
from vllm.model_executor.layers.quantization.utils.quant_utils import QuantKey

_requantization_source: ContextVar[bool] = ContextVar(
    "requantization_source", default=False
)


@contextmanager
def requantization_source_scope() -> Iterator[None]:
    """Mark quantization methods created in this scope as requantization sources.

    Their weights are only dequantized, so oracles can skip backend selection.
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

    # Only the MXFP4 online methods honor an activation override for now.
    accepts_activation_override: bool = False

    # finish clears the guard that keeps it from requantizing converted bytes
    supports_direct_reload = False
    direct_reload_unsupported_reason: str | None = "online quantization"

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
        self.uses_meta_device = False

    def create_requantization_source_weights(
        self, layer: torch.nn.Module, *weight_args, **extra_weight_attrs
    ) -> bool:
        """Delegate weight creation to the source and track its parameters."""
        if self.requantization_source is None:
            return False

        existing_parameter_names = set(layer._parameters)
        self.requantization_source.create_weights(
            layer, *weight_args, **extra_weight_attrs
        )
        self.requantization_source_parameters = {
            name: parameter
            for name, parameter in layer._parameters.items()
            if name not in existing_parameter_names and parameter is not None
        }
        for parameter in self.requantization_source_parameters.values():
            # `replace_parameter` then registers new params over these.
            parameter.requantization_source = True
        return True

    def release_requantization_source_weights(self, layer: torch.nn.Module) -> None:
        """Release serialized parameters after successful requantization."""
        for name, source_parameter in self.requantization_source_parameters.items():
            if layer._parameters.get(name) is source_parameter:
                delattr(layer, name)
        self.requantization_source_parameters.clear()
