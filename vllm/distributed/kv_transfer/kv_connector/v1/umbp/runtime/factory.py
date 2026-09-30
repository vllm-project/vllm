# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Runtime configuration and adapter factory for UMBP."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Any

from vllm.config import VllmConfig

from .base import IUMBPRuntime

_EMBEDDED_DRAM_BOOL_OPTIONS = {
    "dram_use_shared_memory",
    "dram_use_hugepages",
    "dram_prefault",
}
_EMBEDDED_DRAM_POSITIVE_INT_OPTIONS = {
    "capacity_bytes",
    "dram_hugepage_size",
}
_EMBEDDED_DRAM_FLOAT_OPTIONS = {
    "dram_high_watermark",
    "dram_low_watermark",
}


@dataclass(frozen=True)
class UMBPRuntimeConfig:
    """Validated mode-specific configuration shared by connector roles."""

    mode: str
    options: dict[str, Any]

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
        for name in ("capacity_bytes", "ranged_scratch_size"):
            if name in options and (
                type(options[name]) is not int or options[name] <= 0
            ):
                raise ValueError(f"{name} must be a positive integer")
        if mode == "embedded":
            cls._validate_embedded_dram_options(options)
        if mode == "standalone" and not options.get("endpoint"):
            raise ValueError("standalone UMBP requires endpoint")
        if mode == "distributed":
            missing = [
                name
                for name in ("master_address", "node_address", "io_engine_host")
                if not options.get(name)
            ]
            if missing:
                raise ValueError("distributed UMBP requires " + ", ".join(missing))
            if "peer_service_port" in options and (
                type(options["peer_service_port"]) is not int
                or options["peer_service_port"] <= 0
            ):
                raise ValueError("peer_service_port must be a positive integer")
        local_only = {
            "master_address",
            "node_address",
            "io_engine_host",
            "peer_service_port",
        }
        if mode != "distributed" and local_only.intersection(options):
            raise ValueError(f"{mode} UMBP cannot configure distributed-only options")
        return cls(mode=mode, options=options)

    def resolve_for_rank_count(self, rank_count: int) -> UMBPRuntimeConfig:
        """Resolve embedded total DRAM capacity into a per-rank capacity."""
        if rank_count <= 0:
            raise ValueError("rank_count must be positive")
        if self.mode != "embedded" or "total_capacity_bytes" not in self.options:
            return self
        total = self.options["total_capacity_bytes"]
        if total < rank_count:
            raise ValueError(
                "total_capacity_bytes must provide at least one byte per rank"
            )
        options = dict(self.options)
        options["capacity_bytes"] = total // rank_count
        options["_configured_total_capacity_bytes"] = total
        return replace(self, options=options)

    @staticmethod
    def _validate_embedded_dram_options(options: dict[str, Any]) -> None:
        if "capacity_bytes" in options and "total_capacity_bytes" in options:
            raise ValueError(
                "capacity_bytes and total_capacity_bytes are mutually exclusive"
            )
        if "total_capacity_bytes" in options and (
            type(options["total_capacity_bytes"]) is not int
            or options["total_capacity_bytes"] <= 0
        ):
            raise ValueError("total_capacity_bytes must be a positive integer")
        for name in _EMBEDDED_DRAM_BOOL_OPTIONS:
            if name in options and type(options[name]) is not bool:
                raise ValueError(f"{name} must be a boolean")
        for name in _EMBEDDED_DRAM_POSITIVE_INT_OPTIONS - {"capacity_bytes"}:
            if name in options and (
                type(options[name]) is not int or options[name] <= 0
            ):
                raise ValueError(f"{name} must be a positive integer")
        for name in _EMBEDDED_DRAM_FLOAT_OPTIONS:
            if name in options and (
                type(options[name]) not in (int, float) or not 0 < options[name] <= 1
            ):
                raise ValueError(f"{name} must be in (0, 1]")
        low = options.get("dram_low_watermark")
        high = options.get("dram_high_watermark")
        if low is not None and high is not None and low > high:
            raise ValueError("dram_low_watermark must not exceed dram_high_watermark")
        if "dram_numa_node" in options and (
            type(options["dram_numa_node"]) is not int or options["dram_numa_node"] < -1
        ):
            raise ValueError("dram_numa_node must be an integer >= -1")
        if "dram_shm_name" in options and (
            not isinstance(options["dram_shm_name"], str)
            or not options["dram_shm_name"]
        ):
            raise ValueError("dram_shm_name must be a non-empty string")


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
