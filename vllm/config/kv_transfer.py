# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import enum
import uuid
from dataclasses import field
from typing import Any, Literal, get_args

from vllm.config.utils import CompileFactors, config

KVProducer = Literal["kv_producer", "kv_both"]
KVConsumer = Literal["kv_consumer", "kv_both"]
KVRole = Literal[KVProducer, KVConsumer]


def hisparse_host_pool_gib(
    kv_transfer_config: "KVTransferConfig | None",
) -> float | None:
    if kv_transfer_config is None:
        return None
    if kv_transfer_config.kv_connector == "MultiConnector":
        connectors = kv_transfer_config.kv_connector_extra_config.get("connectors", [])
    else:
        connectors = [
            {
                "kv_connector": kv_transfer_config.kv_connector,
                "kv_connector_extra_config": (
                    kv_transfer_config.kv_connector_extra_config
                ),
            }
        ]
    entries = [
        connector.get("kv_connector_extra_config", {})
        for connector in connectors
        if connector.get("kv_connector") == "HiSparseConnector"
    ]
    if len(entries) > 1:
        raise ValueError("Only one HiSparseConnector may be configured")
    if not entries:
        return None
    host_pool_gib = entries[0].get("host_pool_gib")
    if host_pool_gib is None:
        raise ValueError("HiSparseConnector requires host_pool_gib")
    host_pool_gib = float(host_pool_gib)
    if host_pool_gib <= 0:
        raise ValueError("HiSparseConnector host_pool_gib must be positive")
    return host_pool_gib


class KVCacheAccess(enum.IntFlag):
    """How a KV connector accesses the device KV cache, as flags of what it
    requires from the allocation; MultiConnector combines its children's."""

    IN_PROCESS = 0
    """In-process only (torch ops, copy kernels): any allocation works."""

    REGISTERS = enum.auto()
    """Registers it with UCX or RDMA: one physical allocation per segment."""

    EXPORTS_FD = enum.auto()
    """Another process imports it through POSIX file descriptors: one physical
    allocation per segment, exportable (not yet implemented)."""

    EXPORTS_FABRIC = enum.auto()
    """Another process on the host imports it through FABRIC handles (UCX
    cuda_ipc): one physical allocation per segment, and an IMEX channel."""

    LEGACY_IPC = enum.auto()
    """Shares it through legacy CUDA IPC: the extensible KV cache is unsupported."""

    def needs_single_chunk(self) -> bool:
        """Whether each segment must be backed by one physical allocation."""
        exports = KVCacheAccess.EXPORTS_FD | KVCacheAccess.EXPORTS_FABRIC
        return bool(self & (KVCacheAccess.REGISTERS | exports))


def kv_buffer_device_default_factory() -> str:
    from vllm.platforms import current_platform

    return current_platform.device_type


@config
class KVTransferConfig:
    """Configuration for distributed KV cache transfer."""

    kv_connector: str | None = None
    """The KV connector for vLLM to transmit KV caches between vLLM instances.
    """

    engine_id: str | None = None
    """The engine id for KV transfers."""

    kv_buffer_device: str = field(default_factory=kv_buffer_device_default_factory)
    """The device used by kv connector to buffer the KV cache. Choices are
    'cuda', 'cpu' and 'xpu'."""

    kv_role: KVRole | None = None
    """Whether this vLLM instance produces, consumes KV cache, or both. Choices
    are 'kv_producer', 'kv_consumer', and 'kv_both'. With MultiConnector, the
    roles of its connectors decide this instead."""

    kv_port: int = 14579
    """The KV connector port, used to build distributed connection."""

    kv_connector_extra_config: dict[str, Any] = field(default_factory=dict)
    """any extra config that the connector may need."""

    kv_connector_module_path: str | None = None
    """The Python module path to dynamically load the KV connector from.
    Only supported in V1."""

    enable_permute_local_kv: bool = False
    """Experiment feature flag to enable HND to NHD KV Transfer"""

    kv_load_failure_policy: Literal["recompute", "fail"] = "fail"
    """Policy for handling KV cache load failures.
    'recompute': reschedule the request to recompute failed blocks
    'fail': immediately fail the request with an error finish reason (default)"""

    def compile_factors(self) -> CompileFactors:
        """This config does not affect the compiled graph.

        If a field starts affecting it, return
        `get_compile_factors(self, ignored_factors)` with every other field in
        `ignored_factors`.
        """
        return {}

    def __post_init__(self) -> None:
        if self.engine_id is None:
            self.engine_id = str(uuid.uuid4())

        if self.kv_role is not None and self.kv_role not in get_args(KVRole):
            raise ValueError(
                f"Unsupported kv_role: {self.kv_role}. "
                f"Supported roles are {get_args(KVRole)}"
            )

        if self.kv_connector is not None and self.kv_role is None:
            raise ValueError(
                "Please specify kv_role when kv_connector "
                f"is set, supported roles are {get_args(KVRole)}"
            )

    @property
    def is_kv_transfer_instance(self) -> bool:
        return self.kv_connector is not None and self.kv_role in get_args(KVRole)

    @property
    def is_kv_producer(self) -> bool:
        return any(role in get_args(KVProducer) for role in self._connector_roles())

    @property
    def is_kv_consumer(self) -> bool:
        return any(role in get_args(KVConsumer) for role in self._connector_roles())

    def _connector_roles(self) -> list[str | None]:
        """The kv_role of each connector. MultiConnector does not pass its own
        kv_role to its connectors, so only theirs count."""
        if self.kv_connector is None:
            return []
        if self.kv_connector == "MultiConnector":
            return [
                child.get("kv_role")
                for child in self.kv_connector_extra_config.get("connectors", [])
            ]
        return [self.kv_role]

    def get_from_extra_config(self, key, default) -> Any:
        return self.kv_connector_extra_config.get(key, default)

    def uses_custom_mem_pool(self) -> bool:
        """Whether a connector allocates the KV cache from its own memory pool
        (custom_mem_pool), directly or in MultiConnector."""
        configs = [self.kv_connector_extra_config]
        if self.kv_connector == "MultiConnector":
            configs += [
                child.get("kv_connector_extra_config", {})
                for child in self.kv_connector_extra_config.get("connectors", [])
            ]
        return any(config.get("custom_mem_pool") for config in configs)

    def _connector_children(self) -> list[dict[str, Any]]:
        if self.kv_connector == "MultiConnector":
            return self.kv_connector_extra_config.get("connectors", [])
        return [{"kv_connector": self.kv_connector}]

    def kv_cache_access(self) -> KVCacheAccess:
        """How the configured connectors reach the device KV cache, directly or
        in MultiConnector. Imports the connector class to ask it, through
        `KVConnectorBase_V1.kv_cache_access`."""
        if self.kv_connector is None:
            return KVCacheAccess.IN_PROCESS
        from vllm.distributed.kv_transfer.kv_connector.factory import (
            KVConnectorFactory,
        )

        return KVConnectorFactory.kv_cache_access(self)

    def registers_kv_cache_memory(self) -> bool:
        """Whether a connector registers the device KV cache with UCX or RDMA,
        or another process imports it, directly or in MultiConnector: each
        segment must then be backed by one physical allocation."""
        return self.kv_cache_access().needs_single_chunk()

    def has_connector(self, connector_name: str) -> bool:
        """Whether `connector_name` is configured, directly or in MultiConnector."""
        if self.kv_connector == connector_name:
            return True
        return self.kv_connector == "MultiConnector" and any(
            child.get("kv_connector") == connector_name
            for child in self.kv_connector_extra_config.get("connectors", [])
        )
