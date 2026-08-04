# SPDX-License-Identifier: Apache-2.0
"""Wire schema for all usage-telemetry messages — the single source of truth.

Every message that LMCache can phone home is defined in this module,
protobuf-style: one dataclass per message, whose

- **class name** is the wire ``message_type`` discriminator,
- ``ENDPOINT`` class attribute names the stats-server endpoint it POSTs to,
- fields are the payload keys (flat and JSON-serializable; join lists into
  delimited strings, since the stats backend stores flat key-value pairs).

:data:`USAGE_SCHEMA_VERSION` is stamped on every payload; bump it only when
an *existing* field changes meaning — adding a message or field is not a
schema bump.

Identity fields (``session_id``, ``machine_id``) and ``deployment_mode``
are stamped by :func:`lmcache.usage_telemetry.transport.build_usage_payload`,
not declared here. See ``README.md`` for the full schema contract and
extension guide.
"""

# Future
from __future__ import annotations

# Standard
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, ClassVar

# Third Party
import torch

# First Party
from lmcache import torch_device_type

if TYPE_CHECKING:
    # First Party
    from lmcache.v1.config import LMCacheEngineConfig
    from lmcache.v1.distributed.config import StorageManagerConfig
    from lmcache.v1.metadata import LMCacheMetadata
    from lmcache.v1.multiprocess.config import MPServerConfig


USAGE_SCHEMA_VERSION = 1
"""Version stamped on every usage payload; bump when fields change meaning."""


class DeploymentMode(Enum):
    """Which LMCache deployment mode produced a payload.

    Stamped on every payload as the ``deployment_mode`` key.
    """

    SINGLE_PROCESS = "single_process"
    """LMCache runs inside the serving engine (vLLM/SGLang/TRT-LLM
    integrations)."""

    MP_SERVER = "mp_server"
    """LMCache runs as a standalone multiprocess cache server."""


@dataclass
class UsageMessage:
    """Base class of every wire message.

    Subclasses set :attr:`ENDPOINT`; the wire ``message_type`` is the
    subclass name.
    """

    ENDPOINT: ClassVar[str] = "context"
    """Stats-server endpoint this message POSTs to."""


@dataclass
class EnvMessage(UsageMessage):
    """Hardware and platform environment of the reporting process.

    Sent once at startup by every reporter. Integer fields use ``0`` for
    "unknown" (e.g. CPU count unavailable).
    """

    provider: str
    num_cpu: int
    cpu_type: str
    cpu_family_model_stepping: str
    total_memory: int
    architecture: tuple[str, str]
    platforms: str
    gpu_count: int
    gpu_type: str
    gpu_memory_per_device: int
    source: str


@dataclass
class EngineMessage(UsageMessage):
    """Configuration snapshot of a single-process LMCacheEngine.

    Sent once at engine startup (single-process integrations only).
    """

    chunksize: int
    local_device: str
    max_local_cache_size: int
    remote_url: str | None
    remote_serde: str | None
    pipelined_backend: bool
    save_decode_cache: bool
    enable_blending: bool
    blend_recompute_ratio: float
    blend_min_tokens: int
    model_name: str
    world_size: int
    worker_id: int
    kv_dtype: torch.dtype
    kv_shape: tuple[int, int, int, int, int]

    @classmethod
    def from_config(
        cls, config: LMCacheEngineConfig, metadata: LMCacheMetadata
    ) -> EngineMessage:
        """Build the message from the engine configuration and metadata.

        Args:
            config: The engine configuration to snapshot.
            metadata: The engine metadata (model, world size, kv layout).

        Returns:
            The populated message.
        """
        return cls(
            chunksize=config.chunk_size,
            local_device="cpu" if config.local_cpu else torch_device_type,
            max_local_cache_size=int(config.max_local_cpu_size),
            remote_url=config.remote_url,
            remote_serde=config.remote_serde,
            pipelined_backend=False,
            save_decode_cache=config.save_decode_cache,
            enable_blending=config.enable_blending,
            blend_recompute_ratio=0.15,
            blend_min_tokens=config.blend_min_tokens,
            model_name=metadata.model_name,
            world_size=metadata.world_size,
            worker_id=metadata.worker_id,
            kv_dtype=metadata.kv_dtype,
            kv_shape=metadata.kv_shape,
        )


@dataclass
class MetadataMessage(UsageMessage):
    """Process start time and uptime at send time.

    Sent once at engine startup (single-process integrations only).
    """

    start_time: str
    duration: float


@dataclass
class MPServerMessage(UsageMessage):
    """Configuration snapshot of a multiprocess (MP) cache server.

    Sent once at MP server startup.

    Note:
        The server's operator-facing ``instance_id`` is deliberately not
        included: it can be operator-chosen and therefore identifying.
        Usage messages are correlated through
        :class:`lmcache.usage_telemetry.identity.UsageIdentity` only.
    """

    lmcache_version: str
    chunk_size: int
    hash_algorithm: str
    engine_type: str
    enable_segmented_prefix: bool
    supported_transfer_mode: str
    separate_object_groups: bool
    max_gpu_workers: int
    max_cpu_workers: int
    p2p_enabled: bool
    l1_size_bytes: int
    l1_medium: str
    l1_shm_enabled: bool
    eviction_policy: str
    l2_adapter_types: str
    l2_serde_types: str
    l2_store_policy: str
    l2_prefetch_policy: str

    @classmethod
    def from_configs(
        cls,
        mp_config: MPServerConfig,
        storage_manager_config: StorageManagerConfig,
    ) -> MPServerMessage:
        """Build the message from the MP server and storage configurations.

        Args:
            mp_config: The MP server configuration.
            storage_manager_config: The storage manager configuration.

        Returns:
            The populated message. ``l1_medium`` is one of ``"dram"``,
            ``"gds"``, or ``"dram+devdax"``; ``l2_adapter_types`` is a
            comma-joined list of adapter type names (empty when no L2 is
            configured); ``l2_serde_types`` is comma-joined and aligned
            with ``l2_adapter_types``, with an empty entry for adapters
            that have no serde configured.
        """
        # Import here to avoid pulling the distributed config stack into
        # every usage_telemetry import (single-process deployments never
        # load it).
        # First Party
        from lmcache import __version__
        from lmcache.v1.distributed.l2_adapters.config import (
            get_type_name_for_config,
        )

        l1_config = storage_manager_config.l1_manager_config
        memory_config = l1_config.memory_config
        if l1_config.gds_l1_config is not None:
            l1_size_bytes = l1_config.gds_l1_config.size_in_bytes
            l1_medium = "gds"
        else:
            l1_size_bytes = (
                memory_config.size_in_bytes + memory_config.devdax_size_in_bytes
            )
            l1_medium = "dram+devdax" if memory_config.devdax_path else "dram"
        adapter_configs = storage_manager_config.l2_adapter_config.adapters
        l2_adapter_types = ",".join(
            get_type_name_for_config(adapter_config)
            for adapter_config in adapter_configs
        )
        l2_serde_types = ",".join(
            adapter_config.serde_config.type
            if adapter_config.serde_config is not None
            else ""
            for adapter_config in adapter_configs
        )
        return cls(
            lmcache_version=__version__,
            chunk_size=mp_config.chunk_size,
            hash_algorithm=mp_config.hash_algorithm,
            engine_type=mp_config.engine_type,
            enable_segmented_prefix=mp_config.enable_segmented_prefix,
            supported_transfer_mode=mp_config.supported_transfer_mode,
            separate_object_groups=mp_config.separate_object_groups,
            max_gpu_workers=mp_config.max_gpu_workers,
            max_cpu_workers=mp_config.max_cpu_workers,
            p2p_enabled=mp_config.p2p_config.enabled,
            l1_size_bytes=l1_size_bytes,
            l1_medium=l1_medium,
            l1_shm_enabled=bool(memory_config.shm_name),
            eviction_policy=storage_manager_config.eviction_config.eviction_policy,
            l2_adapter_types=l2_adapter_types,
            l2_serde_types=l2_serde_types,
            l2_store_policy=storage_manager_config.store_policy,
            l2_prefetch_policy=storage_manager_config.prefetch_policy,
        )


@dataclass
class ContinuousContextMessage(UsageMessage):
    """Interval counters flushed periodically by the continuous reporters.

    Sent by both deployment modes; distinguish by the ``deployment_mode``
    header. ``interval_num_hit_tokens`` counts tokens actually retrieved
    (served) from LMCache in the interval; ``interval_stored_kv_size`` is
    exact bytes in MP mode and a kv-bytes-per-token estimate in
    single-process mode.
    """

    ENDPOINT: ClassVar[str] = "cache-usage"

    interval_num_stored_tokens: int
    interval_num_hit_tokens: int
    interval_stored_kv_size: int
    sequence_number: int
    uptime_seconds: float
    """Seconds since the reporting process started."""


@dataclass
class CacheLifespanMessage(UsageMessage):
    """Cache-entry lifespan histogram flushed with each continuous report.

    Single-process mode only; measures the store-to-reuse gap per reused
    chunk.
    """

    ENDPOINT: ClassVar[str] = "cache-lifespan"

    cache_lifespan_histogram: dict[float, int]
    """Bucket lower bound (minutes) to sample count. Numeric keys become
    strings on the wire (JSON object keys)."""
    sequence_number: int
    uptime_seconds: float
    """Seconds since the reporting process started."""
