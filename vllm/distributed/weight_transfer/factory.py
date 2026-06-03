# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Factory for weight transfer engines with lazy loading."""

import importlib
from collections.abc import Callable
from typing import TYPE_CHECKING

from vllm.distributed.weight_transfer.base import (
    TrainerWeightTransferEngine,
    WeightTransferEngine,
)
from vllm.logger import init_logger

if TYPE_CHECKING:
    import torch

    from vllm.config import VllmConfig
    from vllm.config.weight_transfer import WeightTransferConfig
    from vllm.distributed.weight_transfer.base import (
        VLLMWeightSyncClient,
        WeightIterator,
        WeightTransferInitInfo,
    )

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
        vllm_config: "VllmConfig",
        device: "torch.device",
        model: "torch.nn.Module",
    ) -> WeightTransferEngine:
        """Create a weight transfer engine instance.

        Args:
            config: Weight transfer configuration containing the backend name
            vllm_config: The full vLLM config (provides parallel/model config)
            device: The device this worker's model lives on
            model: The local model instance which will receive the weights

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

        return engine_cls(config, vllm_config, device, model)


class WeightTransferTrainerFactory:
    """Factory for creating trainer-side weight transfer engines.

    Parallel to `WeightTransferEngineFactory`, with its own lazy-import
    registry. The trainer-side and worker-side registries are kept separate:
    they share backend names by convention, but the trainer process never
    instantiates a worker engine and vice versa, so unifying them would only
    couple the import graphs.
    """

    _registry: dict[str, Callable[[], type[TrainerWeightTransferEngine]]] = {}

    @classmethod
    def register_engine(
        cls,
        name: str,
        module_path_or_cls: "str | type[TrainerWeightTransferEngine]",
        class_name: str | None = None,
    ) -> None:
        """Register a trainer engine. Same conventions as
        `WeightTransferEngineFactory.register_engine`."""
        if name in cls._registry:
            raise ValueError(
                f"Weight transfer trainer engine '{name}' is already registered."
            )

        if isinstance(module_path_or_cls, str):
            module_path = module_path_or_cls
            if class_name is None:
                raise ValueError(
                    "class_name is required when registering with module path"
                )

            def loader() -> type[TrainerWeightTransferEngine]:
                module = importlib.import_module(module_path)
                return getattr(module, class_name)

            cls._registry[name] = loader
        else:
            engine_cls = module_path_or_cls
            cls._registry[name] = lambda: engine_cls

    @classmethod
    def trainer_init(
        cls,
        backend: str,
        config: "WeightTransferConfig",
        init_info: "WeightTransferInitInfo",
        *,
        client: "VLLMWeightSyncClient",
        weight_iterator: "WeightIterator | None" = None,
    ) -> TrainerWeightTransferEngine:
        """Build and rendezvous a ready-to-send trainer engine.

        Args:
            backend: Backend name (must be registered).
            config: Backend-specific weight transfer config.
            init_info: Backend-specific trainer init info.
            client: Inference-side control-plane client.
            weight_iterator: Default (name, tensor) iterator factory.

        Raises:
            ValueError: If the backend is not registered.
        """
        if backend not in cls._registry:
            available = list(cls._registry.keys())
            raise ValueError(
                f"Invalid weight transfer backend: {backend}. "
                f"Available trainer engines: {available}"
            )
        engine_cls = cls._registry[backend]()

        logger.info(
            "Creating weight transfer trainer engine: %s",
            engine_cls.__name__,
        )

        return engine_cls.trainer_init(
            config=config,
            init_info=init_info,
            client=client,
            weight_iterator=weight_iterator,
        )


# Register built-in weight transfer engines here.
# Registration should be centralized to ensure lazy loading -
# engine modules are only imported when actually used.

WeightTransferEngineFactory.register_engine(
    "nccl",
    "vllm.distributed.weight_transfer.nccl_engine",
    "NCCLWeightTransferEngine",
)

WeightTransferEngineFactory.register_engine(
    "ipc",
    "vllm.distributed.weight_transfer.ipc_engine",
    "IPCWeightTransferEngine",
)

WeightTransferEngineFactory.register_engine(
    "sparse_nccl",
    "vllm.distributed.weight_transfer.sparse_nccl_engine",
    "SparseNCCLWeightTransferEngine",
)

# Trainer-side engines (no sparse: the sparse backend keeps its own static
# trainer path for now).
WeightTransferTrainerFactory.register_engine(
    "nccl",
    "vllm.distributed.weight_transfer.nccl_engine",
    "NCCLTrainerWeightTransferEngine",
)

WeightTransferTrainerFactory.register_engine(
    "ipc",
    "vllm.distributed.weight_transfer.ipc_engine",
    "IPCTrainerWeightTransferEngine",
)
