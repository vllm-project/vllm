# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import importlib
from collections.abc import Callable

from vllm.distributed.kv_transfer.kv_connector.v1.explicit_offloading.storage.abstract import (  # noqa: E501
    ExOffloadingStorage,
)
from vllm.logger import init_logger

logger = init_logger(__name__)


class ExOffloadingStorageFactory:
    _registry: dict[str, Callable[[], type[ExOffloadingStorage]]] = {}

    @classmethod
    def register_storage(cls, name: str, module_path: str, class_name: str) -> None:
        """Register a storage with a lazy-loading module and class name."""
        if name in cls._registry:
            raise ValueError(f"Storage '{name}' is already registered.")

        def loader() -> type[ExOffloadingStorage]:
            module = importlib.import_module(module_path)
            return getattr(module, class_name)

        cls._registry[name] = loader

    @classmethod
    def get_storage_class(cls, name: str) -> type[ExOffloadingStorage]:
        if name not in cls._registry:
            available_backends = list(cls._registry.keys())
            raise ValueError(
                f"Unknown storage '{name}'. Registered storages: {available_backends}. "
            )

        return cls._registry[name]()


ExOffloadingStorageFactory.register_storage(
    "file",
    "vllm.distributed.kv_transfer.kv_connector.v1.explicit_offloading.storage.file",
    "FileStorage",
)

ExOffloadingStorageFactory.register_storage(
    "gd2fs",
    "vllm.distributed.kv_transfer.kv_connector.v1.explicit_offloading.storage.gd2fs",
    "GD2FSStorage",
)
