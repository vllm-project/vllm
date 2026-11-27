# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Factory for weight transfer engines with lazy loading."""

import importlib
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, ClassVar, Generic, TypeVar

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
        TrainerInitInfo,
        VLLMWeightSyncClient,
        WeightSource,
    )

logger = init_logger(__name__)

_EngineT = TypeVar("_EngineT")


class _LazyEngineFactory(Generic[_EngineT]):
    """Registry-backed factory that imports engine classes lazily on demand."""

    _registry: dict[str, Callable[[], type[Any]]]
    _engine_kind: ClassVar[str]

    @classmethod
    def register_engine(
        cls,
        name: str,
        module_path_or_cls: str | type[_EngineT],
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
            raise ValueError(
                f"Weight transfer {cls._engine_kind} '{name}' is already registered."
            )

        if isinstance(module_path_or_cls, str):
            # Lazy loading path
            module_path = module_path_or_cls
            if class_name is None:
                raise ValueError(
                    "class_name is required when registering with module path"
                )
            target_class = class_name

            def loader() -> type[_EngineT]:
                module = importlib.import_module(module_path)
                return getattr(module, target_class)

            cls._registry[name] = loader
        else:
            # Direct class registration
            engine_cls = module_path_or_cls
            cls._registry[name] = lambda: engine_cls

    @classmethod
    def _resolve_engine(cls, backend: str) -> type[_EngineT]:
        if backend not in cls._registry:
            available = list(cls._registry.keys())
            raise ValueError(
                f"Invalid weight transfer backend: {backend}. "
                f"Available {cls._engine_kind}s: {available}"
            )
        engine_cls = cls._registry[backend]()

        logger.info(
            "Creating weight transfer %s: %s",
            cls._engine_kind,
            engine_cls.__name__,
        )
        return engine_cls


class WeightTransferEngineFactory(_LazyEngineFactory[WeightTransferEngine]):
    """Factory for creating weight transfer engines with lazy loading.

    This factory implements a registry pattern that supports:
    - Lazy loading: Engine modules are only imported when actually needed
    - Extensibility: Custom engines can be registered at runtime
    - Centralized registration: All built-in engines registered in one place
    """

    _registry: dict[str, Callable[[], type[WeightTransferEngine]]] = {}
    _engine_kind: ClassVar[str] = "engine"

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
        engine_cls = cls._resolve_engine(config.backend)
        return engine_cls(config, vllm_config, device, model)


class WeightTransferTrainerFactory(_LazyEngineFactory[TrainerWeightTransferEngine]):
    """Factory for creating trainer-side weight transfer engines.

    Parallel to `WeightTransferEngineFactory`, with its own lazy-import
    registry. The trainer-side and worker-side registries are kept separate:
    they share backend names by convention, but the trainer process never
    instantiates a worker engine and vice versa, so unifying them would only
    couple the import graphs.
    """

    _registry: dict[str, Callable[[], type[TrainerWeightTransferEngine]]] = {}
    _engine_kind: ClassVar[str] = "trainer engine"

    @classmethod
    def trainer_init(
        cls,
        init_info: "TrainerInitInfo",
        *,
        client: "VLLMWeightSyncClient",
        source: "WeightSource | None" = None,
    ) -> TrainerWeightTransferEngine:
        """Build and rendezvous a ready-to-send trainer engine.

        Called on every trainer rank (multi-rank trainers construct on all
        ranks; the sender is resolved inside the engine's `trainer_init`).

        The trainer side takes no `WeightTransferConfig` and no separate
        `backend` argument: the backend is read from `init_info.backend` (a
        `ClassVar` on each `TrainerInitInfo` subclass), and the static wire
        params ride `init_info`.

        Args:
            init_info: Backend-specific trainer init info. Its `backend`
                selects the engine; it also carries the wire params (e.g.
                `packed`).
            client: Inference-side control-plane client.
            source: `WeightSource` of `(name, tensor)` pairs to send each round,
                for full-resync backends (NCCL, IPC). Sparse backend
                omits it and passes its per-round payload to `send_weights`.

        Raises:
            ValueError: If `init_info.backend` is not registered.

        """
        engine_cls = cls._resolve_engine(init_info.backend)
        return engine_cls.trainer_init(
            init_info=init_info,
            client=client,
            source=source,
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

WeightTransferEngineFactory.register_engine(
    "sparse_disk",
    "vllm.distributed.weight_transfer.sparse_disk_engine",
    "SparseDiskWeightTransferEngine",
)

WeightTransferEngineFactory.register_engine(
    "sharded_rdt",
    "vllm.distributed.weight_transfer.sharded_rdt_engine",
    "ShardedRDTWeightTransferEngine",
)

WeightTransferEngineFactory.register_engine(
    "nccl_m2n",
    "vllm.distributed.weight_transfer.m2n_engine",
    "M2NWeightTransferEngine",
)


WeightTransferEngineFactory.register_engine(
    "modelexpress",
    "vllm.distributed.weight_transfer.modelexpress_engine",
    "ModelExpressWeightTransferEngine",
)

WeightTransferEngineFactory.register_engine(
    "connex",
    "vllm.distributed.weight_transfer.connex_engine",
    "ConnexWeightTransferEngine",
)


# Trainer-side engines, parallel to the worker registry above.
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

WeightTransferTrainerFactory.register_engine(
    "sparse_nccl",
    "vllm.distributed.weight_transfer.sparse_nccl_engine",
    "SparseNCCLTrainerWeightTransferEngine",
)

WeightTransferTrainerFactory.register_engine(
    "sparse_disk",
    "vllm.distributed.weight_transfer.sparse_disk_engine",
    "SparseDiskTrainerWeightTransferEngine",
)

WeightTransferTrainerFactory.register_engine(
    "sharded_rdt",
    "vllm.distributed.weight_transfer.sharded_rdt_trainer",
    "ShardedRDTTrainerWeightTransferEngine",
)

WeightTransferTrainerFactory.register_engine(
    "nccl_m2n",
    "vllm.distributed.weight_transfer.m2n_trainer",
    "M2NTrainerWeightTransferEngine",
)

WeightTransferTrainerFactory.register_engine(
    "connex",
    "vllm.distributed.weight_transfer.connex_engine",
    "ConnexTrainerWeightTransferEngine",
)
