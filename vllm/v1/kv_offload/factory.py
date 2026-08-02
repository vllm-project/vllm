# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import importlib
from collections.abc import Callable, Mapping
from typing import Any

from vllm.logger import init_logger
from vllm.v1.kv_offload.base import OffloadingSpec
from vllm.v1.kv_offload.config import OffloadingConfig

logger = init_logger(__name__)


class OffloadingSpecFactory:
    _registry: dict[str, Callable[[], type[OffloadingSpec]]] = {}

    @classmethod
    def register_spec(cls, name: str, module_path: str, class_name: str) -> None:
        """Register a spec with a lazy-loading module and class name."""
        if name in cls._registry:
            raise ValueError(f"Connector '{name}' is already registered.")

        def loader() -> type[OffloadingSpec]:
            module = importlib.import_module(module_path)
            return getattr(module, class_name)

        cls._registry[name] = loader

    @classmethod
    def get_spec_cls(cls, extra_config: Mapping[str, Any]) -> type[OffloadingSpec]:
        spec_name = extra_config.get("spec_name", "CPUOffloadingSpec")
        if spec_name in cls._registry:
            spec_cls = cls._registry[spec_name]()
        else:
            spec_module_path = extra_config.get("spec_module_path")
            if spec_module_path is None:
                raise ValueError(f"Unsupported spec type: {spec_name}")
            spec_module = importlib.import_module(spec_module_path)
            spec_cls = getattr(spec_module, spec_name)
        assert issubclass(spec_cls, OffloadingSpec)
        return spec_cls

    @classmethod
    def create_spec(cls, config: OffloadingConfig) -> OffloadingSpec:
        spec_name = config.extra_config.get("spec_name", "CPUOffloadingSpec")
        spec_cls = cls.get_spec_cls(config.extra_config)
        logger.info("Creating offloading spec with name: %s", spec_name)
        return spec_cls(config)


# Register various specs here.
OffloadingSpecFactory.register_spec(
    "CPUOffloadingSpec", "vllm.v1.kv_offload.cpu.spec", "CPUOffloadingSpec"
)
OffloadingSpecFactory.register_spec(
    "TieringOffloadingSpec",
    "vllm.v1.kv_offload.tiering.spec",
    "TieringOffloadingSpec",
)
