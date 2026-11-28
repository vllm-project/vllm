# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import importlib
from collections.abc import Callable, Mapping
from typing import Any

import torch

from vllm.logger import init_logger
from vllm.v1.kv_offload.base import CanonicalKVCaches
from vllm.v1.kv_offload.file_mapper import FileMapper
from vllm.v1.kv_offload.fs.worker import FSOffloadingWorker

logger = init_logger(__name__)


class FSWorkerFactory:
    """Registry for FSOffloadingWorker backends, resolved by fs_backend.

    Mirrors OffloadingSpecFactory (vllm/v1/kv_offload/factory.py): built-in
    backends are pre-registered below. External backends can either
    register_backend() a friendly short name up front, or skip registration
    and set fs_backend_module_path in the config -- see get_worker_cls.

    Every backend is constructed as
    cls(file_mapper=, blocks_per_file=, device=, kv_caches=, root_dir=,
    extra_config=).
    """

    _registry: dict[str, Callable[[], type[FSOffloadingWorker]]] = {}

    @classmethod
    def register_backend(cls, name: str, module_path: str, class_name: str) -> None:
        """Register a backend with a lazy-loading module and class name."""
        if name in cls._registry:
            raise ValueError(f"FS backend '{name}' is already registered.")

        def loader() -> type[FSOffloadingWorker]:
            module = importlib.import_module(module_path)
            return getattr(module, class_name)

        cls._registry[name] = loader

    @classmethod
    def get_worker_cls(
        cls, extra_config: Mapping[str, Any]
    ) -> type[FSOffloadingWorker]:
        """Get the worker class named by extra_config["fs_backend"].

        Args:
            extra_config: kv_connector_extra_config. fs_backend (default
                "gds") is checked against the registry first; if it is not
                registered and fs_backend_module_path is set, fs_backend is
                imported from there instead.

        Returns:
            The FSOffloadingWorker subclass.

        Raises:
            ValueError: If fs_backend is neither registered nor resolvable via
                fs_backend_module_path.

        """
        name = extra_config.get("fs_backend", "gds")
        if name in cls._registry:
            worker_cls = cls._registry[name]()
        else:
            module_path = extra_config.get("fs_backend_module_path")
            if module_path is None:
                raise ValueError(
                    f"Unknown fs_backend: {name!r}. Supported: {list(cls._registry)}. "
                    "For an out-of-tree backend, also set fs_backend_module_path."
                )
            logger.warning_once(
                "Loading out-of-tree FS offloading backend. This API is "
                "experimental and subject to change in the future "
                "as we iterate the design."
            )
            logger.info(
                "Loading out-of-tree FS offloading backend '%s' from '%s'.",
                name,
                module_path,
            )
            module = importlib.import_module(module_path)
            worker_cls = getattr(module, name)
        assert issubclass(worker_cls, FSOffloadingWorker)
        return worker_cls

    @classmethod
    def create_worker(
        cls,
        extra_config: Mapping[str, Any],
        file_mapper: FileMapper,
        blocks_per_file: int,
        device: torch.device,
        kv_caches: CanonicalKVCaches,
        root_dir: str,
    ) -> FSOffloadingWorker:
        worker_cls = cls.get_worker_cls(extra_config)
        logger.info("Creating FS offloading worker: %s", worker_cls.__name__)
        return worker_cls(  # type: ignore[call-arg]
            file_mapper=file_mapper,
            blocks_per_file=blocks_per_file,
            device=device,
            kv_caches=kv_caches,
            root_dir=root_dir,
            extra_config=extra_config,
        )


FSWorkerFactory.register_backend(
    "gds", "vllm.v1.kv_offload.fs.gds.worker", "GDSOffloadingWorker"
)
