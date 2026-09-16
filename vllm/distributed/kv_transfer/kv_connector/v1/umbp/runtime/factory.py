# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Runtime configuration and adapter factory for UMBP."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from vllm.config import VllmConfig

from .base import IUMBPRuntime


@dataclass(frozen=True)
class UMBPRuntimeConfig:
    """Validated mode-specific configuration shared by connector roles."""

    mode: str
    options: dict[str, Any]

    @classmethod
    def from_vllm(cls, vllm_config: VllmConfig) -> "UMBPRuntimeConfig":
        transfer_config = vllm_config.kv_transfer_config
        if transfer_config is None:
            raise ValueError("UMBP requires kv_transfer_config")
        options = dict(transfer_config.kv_connector_extra_config)
        mode = options.get("mode", "embedded")
        if mode not in {"embedded", "standalone", "distributed"}:
            raise ValueError(f"unknown UMBP mode: {mode!r}")
        namespace = options.get("key_namespace", "auto")
        if namespace != "auto" and (
            not isinstance(namespace, str) or not namespace
        ):
            raise ValueError("key_namespace must be a non-empty string or 'auto'")
        for name in ("capacity_bytes", "ranged_scratch_size"):
            if name in options and (
                type(options[name]) is not int or options[name] <= 0
            ):
                raise ValueError(f"{name} must be a positive integer")
        if mode == "standalone" and not options.get("endpoint"):
            raise ValueError("standalone UMBP requires endpoint")
        if mode == "distributed":
            missing = [
                name
                for name in ("master_address", "node_address", "io_engine_host")
                if not options.get(name)
            ]
            if missing:
                raise ValueError(
                    "distributed UMBP requires " + ", ".join(missing)
                )
            if (
                "peer_service_port" in options
                and (
                    type(options["peer_service_port"]) is not int
                    or options["peer_service_port"] <= 0
                )
            ):
                raise ValueError("peer_service_port must be a positive integer")
        local_only = {
            "master_address",
            "node_address",
            "io_engine_host",
            "peer_service_port",
        }
        if mode != "distributed" and local_only.intersection(options):
            raise ValueError(
                f"{mode} UMBP cannot configure distributed-only options"
            )
        return cls(mode=mode, options=options)


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
