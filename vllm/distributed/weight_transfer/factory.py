# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Factory for weight transfer engines with lazy loading."""

import importlib
from collections.abc import Callable
from typing import TYPE_CHECKING

from vllm.distributed.weight_transfer.base import WeightTransferEngine
from vllm.logger import init_logger

if TYPE_CHECKING:
    from vllm.config.parallel import ParallelConfig
    from vllm.config.weight_transfer import WeightTransferConfig

logger = init_logger(__name__)


class WeightTransferEngineFactory:
    """Factory for creating weight transfer engines with lazy loading.

    This factory implements a registry pattern that supports:
    - Lazy loading: Engine modules are only imported when actually needed
    - Extensibility: Custom engines can be registered at runtime
    - Centralized registration: All built-in engines registered in one place
    """

    _registry: dict[str, Callable[[], type[WeightTransferEngine]]] = {}

    @classmethod
    def register_engine(
        cls,
        name: str,
        module_path_or_cls: str | type[WeightTransferEngine],
        class_name: str | None = None,
    ) -> None:
        """Register an engine with lazy-loading or direct class reference.

        Supports two calling conventions:
        1. Lazy loading: register_engine(name, module_path, class_name)
        2. Direct class: register_engine(name, engine_cls)

        Args:
            name: The name to register the engine under (e.g., "nccl")
            module_path_or_cls: Either a module path string for lazy loading,
                or the engine class directly
            class_name: Name of the engine class (required if module_path is string)

        Raises:
            ValueError: If an engine with the same name is already registered
        """
        if name in cls._registry:
            raise ValueError(f"Weight transfer engine '{name}' is already registered.")

        if isinstance(module_path_or_cls, str):
            # Lazy loading path
            module_path = module_path_or_cls
            if class_name is None:
                raise ValueError(
                    "class_name is required when registering with module path"
                )

            def loader() -> type[WeightTransferEngine]:
                module = importlib.import_module(module_path)
                return getattr(module, class_name)

            cls._registry[name] = loader
        else:
            # Direct class registration
            engine_cls = module_path_or_cls
            cls._registry[name] = lambda: engine_cls

    @classmethod
    def create_engine(
        cls,
        config: "WeightTransferConfig",
        parallel_config: "ParallelConfig",
    ) -> WeightTransferEngine:
        """Create a weight transfer engine instance.

        Args:
            config: Weight transfer configuration containing the backend name
            parallel_config: Parallel configuration for the engine

        Returns:
            An initialized weight transfer engine instance

        Raises:
            ValueError: If the backend is not registered
        """
        backend = config.backend
        if backend not in cls._registry:
            available = list(cls._registry.keys())
            raise ValueError(
                f"Invalid weight transfer backend: {backend}. "
                f"Available engines: {available}"
            )
        engine_cls = cls._registry[backend]()

        logger.info(
            "Creating weight transfer engine: %s",
            engine_cls.__name__,
        )

        return engine_cls(config, parallel_config)


# Register built-in weight transfer engines here.
# Registration should be centralized to ensure lazy loading -
# engine modules are only imported when actually used.

WeightTransferEngineFactory.register_engine(
    "nccl",
    "vllm.distributed.weight_transfer.nccl_engine",
    "NCCLWeightTransferEngine",
)
