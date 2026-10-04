# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Runtime configuration and adapter factory for UMBP."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Any

from vllm.config import VllmConfig

from .base import IUMBPRuntime


@dataclass(frozen=True)
class UMBPRuntimeConfig:
    """Shared connector options passed to the selected runtime adapter."""

    mode: str
    options: dict[str, Any]
    rank_count: int = 1

    @classmethod
    def from_vllm(cls, vllm_config: VllmConfig) -> UMBPRuntimeConfig:
        transfer_config = vllm_config.kv_transfer_config
        if transfer_config is None:
            raise ValueError("UMBP requires kv_transfer_config")
        options = dict(transfer_config.kv_connector_extra_config)
        mode = options.get("mode", "embedded")
        if mode not in {"embedded", "standalone", "distributed"}:
            raise ValueError(f"unknown UMBP mode: {mode!r}")
        namespace = options.get("key_namespace", "auto")
        if namespace != "auto" and (not isinstance(namespace, str) or not namespace):
            raise ValueError("key_namespace must be a non-empty string or 'auto'")
        return cls(mode=mode, options=options)

    def resolve_for_rank_count(self, rank_count: int) -> UMBPRuntimeConfig:
        """Pass topology to the adapter without interpreting its capacity options."""
        if rank_count <= 0:
            raise ValueError("rank_count must be positive")
        return replace(self, rank_count=rank_count)


RuntimeBuilder = Callable[[UMBPRuntimeConfig], IUMBPRuntime]


class UMBPRuntimeFactory:
    """Registry used by UMBP adapters without importing MORI in vLLM."""

    _builders: dict[str, RuntimeBuilder] = {}

    @classmethod
    def register(cls, mode: str, builder: RuntimeBuilder) -> None:
        if mode not in {"embedded", "standalone", "distributed"}:
            raise ValueError(f"unknown UMBP mode: {mode!r}")
        if mode in cls._builders:
            raise ValueError(f"UMBP runtime mode {mode!r} is already registered")
        cls._builders[mode] = builder

    @classmethod
    def build(cls, config: UMBPRuntimeConfig) -> IUMBPRuntime:
        try:
            builder = cls._builders[config.mode]
        except KeyError as exc:
            raise RuntimeError(
                f"no UMBP runtime adapter is registered for mode {config.mode!r}"
            ) from exc
        runtime = builder(config)
        capabilities = runtime.capabilities
        missing = [
            name
            for name in ("lookup", "load", "store", "publish")
            if not getattr(capabilities, name, False)
        ]
        if missing:
            raise RuntimeError(
                f"UMBP runtime {config.mode!r} lacks required capabilities: "
                + ", ".join(missing)
            )
        return runtime
