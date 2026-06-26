# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Registry for post-load weight processing module types."""

import importlib
from collections.abc import Callable, Iterable

from torch import nn


class WeightProcessingFactory:
    """Registry for modules that need deferred post-load weight processing."""

    _registry: dict[str, Callable[[], type[nn.Module]]] = {}

    @classmethod
    def register_module(
        cls,
        name: str,
        module_path: str,
        class_name: str,
    ) -> None:
        """Register a module type with a lazy-loading module and class name."""
        if name in cls._registry:
            raise ValueError(f"Module type '{name}' is already registered.")

        def loader() -> type[nn.Module]:
            module = importlib.import_module(module_path)
            module_type = getattr(module, class_name)
            if not issubclass(module_type, nn.Module):
                raise TypeError(
                    "Registered module type must be a subclass of torch.nn.Module, "
                    f"got {module_type!r}."
                )
            return module_type

        cls._registry[name] = loader

    @classmethod
    def get_module_types(cls) -> tuple[type[nn.Module], ...]:
        return tuple(loader() for loader in cls._registry.values())

    @classmethod
    def is_registered_module(cls, module: object) -> bool:
        return isinstance(module, cls.get_module_types())

    @classmethod
    def iter_registered_module_types(cls) -> Iterable[type[nn.Module]]:
        return tuple(loader() for loader in cls._registry.values())


WeightProcessingFactory.register_module(
    "Attention",
    "vllm.model_executor.layers.attention.attention",
    "Attention",
)

WeightProcessingFactory.register_module(
    "MLAAttention",
    "vllm.model_executor.layers.attention.mla_attention",
    "MLAAttention",
)

WeightProcessingFactory.register_module(
    "MMEncoderAttention",
    "vllm.model_executor.layers.attention.mm_encoder_attention",
    "MMEncoderAttention",
)
