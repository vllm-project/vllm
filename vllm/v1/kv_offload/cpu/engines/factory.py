# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Factory selecting the copy mechanism for the CPU offload worker."""

import importlib
from collections.abc import Callable
from typing import Any

from vllm.v1.kv_offload.cpu.engines.base import CopyEngine

EngineCtor = Callable[..., CopyEngine]


class CopyEngineFactory:
    """Registry of copy mechanisms by name.

    Registrations hold an import path rather than a class, so a mechanism is
    only imported when it is selected.
    """

    _registry: dict[str, Callable[[], EngineCtor]] = {}

    @classmethod
    def register(cls, name: str, module_path: str, class_name: str) -> None:
        if name in cls._registry:
            raise ValueError(f"Copy mechanism {name!r} is already registered.")

        def load() -> EngineCtor:
            module = importlib.import_module(module_path)
            return getattr(module, class_name)

        cls._registry[name] = load

    @classmethod
    def get_engine(cls, name: str) -> EngineCtor:
        load = cls._registry.get(name)
        if load is None:
            raise ValueError(f"Unknown copy mechanism: {name!r}")
        return load()

    @classmethod
    def create(cls, name: str, **kwargs: Any) -> CopyEngine:
        return cls.get_engine(name)(**kwargs)


CopyEngineFactory.register(
    "torch",
    "vllm.v1.kv_offload.cpu.engines.torch",
    "TorchCopyEngine",
)
