# SPDX-License-Identifier: Apache-2.0

"""
Configuration for distributed storage manager
"""

# Standard
from dataclasses import dataclass, field
from typing import Any, Literal, cast
import argparse
import os

# First Party
from lmcache.logging import init_logger
from lmcache.v1.distributed.l2_adapters.config import (
    L2AdapterConfigBase,
    L2AdaptersConfig,
    add_l2_adapters_args,
    get_type_name_for_config,
    parse_args_to_l2_adapters_config,
)
from lmcache.v1.platform import current_device_spec

logger = init_logger(__name__)


_HYBRID_L1_SINGLE_REGION_L2_ADAPTERS = {
    "nixl_store",
    "nixl_store_dynamic",
}


def _requires_single_l1_memory_region(
    adapter_config: L2AdapterConfigBase,
) -> str | None:
    type_name = get_type_name_for_config(adapter_config)
    if type_name in _HYBRID_L1_SINGLE_REGION_L2_ADAPTERS:
        return type_name
    if (
        type_name == "mooncake_store"
        and getattr(adapter_config, "setup_config", {}).get("protocol") == "rdma"
    ):
        return type_name
    return None


def _infer_l1_devdax_overflow_from_dax_adapter(
    memory_config: "L1MemoryManagerConfig",
    l2_adapter_config: L2AdaptersConfig,
) -> None:
    if not memory_config.devdax_path or memory_config.devdax_size_in_bytes:
        return

    l1_devdax_path = memory_config.devdax_path
    remaining_adapters: list[L2AdapterConfigBase] = []
    matched_dax_device: Any | None = None
    for adapter_config in l2_adapter_config.adapters:
        if get_type_name_for_config(adapter_config) != "dax":
            remaining_adapters.append(adapter_config)
            continue

        dax_adapter = cast(Any, adapter_config)
        devices = list(getattr(dax_adapter, "devices", []))
        matched_devices = [
            device
            for device in devices
            if getattr(device, "device_path", None) == l1_devdax_path
        ]
        if not matched_devices:
            remaining_adapters.append(adapter_config)
            continue
        if matched_dax_device is not None or len(matched_devices) > 1:
            raise ValueError(
                "Only one DAX device can match l1-devdax-path for hybrid L1"
            )

        matched_dax_device = matched_devices[0]
        remaining_devices = [
            device
            for device in devices
            if getattr(device, "device_path", None) != l1_devdax_path
        ]
        if remaining_devices:
            remaining_dax_adapter = type(dax_adapter)(
                devices=remaining_devices,
                hotplug_enabled=dax_adapter.hotplug_enabled,
                slot_bytes=dax_adapter.slot_bytes,
                num_store_workers=dax_adapter.num_store_workers,
                num_lookup_workers=dax_adapter.num_lookup_workers,
                num_load_workers=dax_adapter.num_load_workers,
            )
            remaining_dax_adapter.eviction_config = dax_adapter.eviction_config
            remaining_dax_adapter.persist_config = dax_adapter.persist_config
            remaining_dax_adapter.serde_config = dax_adapter.serde_config
            remaining_adapters.append(remaining_dax_adapter)

    if matched_dax_device is None:
        return
    max_dax_size_gb = matched_dax_device.max_dax_size_gb
    memory_config.devdax_size_in_bytes = int(max_dax_size_gb * (1 << 30))
    l2_adapter_config.adapters = remaining_adapters


@dataclass
class L1MemoryManagerConfig:
    """
    The configuration for L1 memory manager.
    """

    size_in_bytes: int
    """ The size of L1 memory in bytes. """

    use_lazy: bool
    """ Whether to use lazy initialization for L1 memory. """

    init_size_in_bytes: int = field(default=20 << 30)
    """ The initial size when using lazy allocation. Default is 20GB. """

    align_bytes: int = field(default=0x1000)
    """ The alignment size in bytes. Default is 4KB. """

    shm_name: str = field(default_factory=lambda: f"lmcache_l1_pool_{os.getpid()}")
    """ POSIX shared-memory segment name for L1 pool. Empty disables SHM. """

    devdax_path: str | None = None
    """ Optional Device-DAX path to use as the L1 backing arena. """

    devdax_size_in_bytes: int = 0
    """ Optional Device-DAX overflow size for hybrid DRAM + DAX L1. """

    def __post_init__(self):
        self.init_size_in_bytes = min(self.init_size_in_bytes, self.size_in_bytes)

        if self.devdax_path is not None:
            self.devdax_path = self.devdax_path.strip()

        if self.devdax_size_in_bytes < 0:
            raise ValueError("devdax_size_in_bytes must be >= 0")
        if self.devdax_size_in_bytes and not self.devdax_path:
            raise ValueError("devdax_size_in_bytes requires devdax_path")

        if self.devdax_path and self.use_lazy:
            raise ValueError(
                "l1-devdax-path requires lazy allocation to be disabled. "
                "Please set --no-l1-use-lazy."
            )
        if self.devdax_path and self.shm_name:
            raise ValueError(
                'l1-devdax-path requires SHM to be disabled. Please set --shm-name "".'
            )

        # LazyMemoryAllocator requires pinned memory support.
        # Auto-disable on platforms that don't support it to avoid a RuntimeError.
        if self.use_lazy and not current_device_spec.is_pin_supported:
            logger.warning(
                "LazyMemoryAllocator requires memory pinning which is not "
                "supported on the current backend. Disabling l1-use-lazy."
            )
            self.use_lazy = False


@dataclass
class GdsL1Config:
    """Configuration for the GDS slab-file L1 tier.

    When present on :class:`L1ManagerConfig`, the L1 medium becomes an NVMe
    slab file accessed via GPUDirect Storage DMA (cuFile on NVIDIA, hipFile on
    AMD ROCm) instead of pinned DRAM (mutually exclusive with the pinned-DRAM
    tier in ``memory_config``). Carries the slab location, capacity, and DMA
    mode.
    """

    file_location: str
    """Directory for the slab file (one shared slab per process, used by all
    GPU instances)."""

    size_in_bytes: int
    """Slab capacity in bytes (from ``--l1-size-gb``). Sizes both the
    preallocated slab file and the GDS tier's address space."""

    use_direct_io: bool = True
    """Open the slab with ``O_DIRECT`` (required for the GDS DMA fast path)."""

    align_bytes: int = 4096
    """Allocation alignment; cuFile/hipFile and O_DIRECT require 4 KiB."""


@dataclass
class L1ManagerConfig:
    """
    Special config for the L1 Object/Key manager
    """

    memory_config: L1MemoryManagerConfig
    """ The memory manager configuration for L1 cache. """

    gds_l1_config: "GdsL1Config | None" = None
    """ Optional GDS L1 tier. When set, the GDS slab is the L1 medium
    (mutually exclusive with the pinned-DRAM tier in ``memory_config``). """

    write_ttl_seconds: int = field(default=600)
    """ Time to live for each object's write lock. Default is 600s (10 minutes). """

    read_ttl_seconds: int = field(default=300)
    """ Time to live for each object's read lock. Default is 300s (5 minutes). """


@dataclass
class EvictionConfig:
    """
    The configuration for eviction policies (L1 and optionally L2).
    """

    eviction_policy: Literal["LRU", "IsolatedLRU", "noop"]
    """ The eviction policy to use. """

    trigger_watermark: float = field(default=0.8)
    """ The memory usage watermark to trigger eviction (0.0 to 1.0). """

    eviction_ratio: float = field(default=0.2)
    """ The fraction of *allocated* memory to evict when triggered (0.0 to 1.0). """

    extra_logging_enabled: bool = field(default=False)
    """ Whether the eviction loop periodically logs L1 memory usage at INFO. """

    extra_logging_interval: float = field(default=10.0)
    """ Seconds between L1 memory usage log lines when extra logging is enabled. """


@dataclass
class StorageManagerConfig:
    """
    The configuration for the distributed storage manager.
    """

    l1_manager_config: L1ManagerConfig
    """ The configuration for the L1 manager. """

    eviction_config: EvictionConfig
    """ The configuration for eviction policies. """

    l2_adapter_config: L2AdaptersConfig = field(
        default_factory=lambda: L2AdaptersConfig([])
    )
    """ The configuration for L2 adapters. """

    store_policy: str = "default"
    """ The L2 store policy name. """

    prefetch_policy: str = "default"
    """ The L2 prefetch policy name. """

    prefetch_max_in_flight: int = 8
    """ Maximum number of concurrent prefetch requests. """

    periodic_notifier_interval_ms: int = 5
    """ Interval (ms) for the periodic event notifier heartbeat. """

    def __post_init__(self) -> None:
        normalize_storage_manager_config(self)
        validate_storage_manager_config(self)


def normalize_storage_manager_config(config: StorageManagerConfig) -> None:
    """Normalize storage manager configuration.

    This consumes a matching DAX adapter as hybrid L1 Device-DAX overflow
    capacity.

    Args:
        config: Storage manager configuration to normalize in place.

    Returns:
        None.

    Raises:
        ValueError: If more than one DAX device matches ``l1-devdax-path``.
    """
    memory_config = config.l1_manager_config.memory_config
    _infer_l1_devdax_overflow_from_dax_adapter(memory_config, config.l2_adapter_config)


def validate_storage_manager_config(config: StorageManagerConfig) -> None:
    """Validate storage manager configuration.

    This rejects L2 adapters that require a single contiguous L1 memory
    descriptor when hybrid L1 Device-DAX overflow is enabled.

    Args:
        config: Storage manager configuration to validate.

    Returns:
        None.

    Raises:
        ValueError: If mutually exclusive L1 tiers are both configured, or
            hybrid L1 is paired with incompatible L2 adapters.
    """
    if (
        config.l1_manager_config.gds_l1_config is not None
        and config.l1_manager_config.memory_config.devdax_path
    ):
        raise ValueError("gds-l1-path cannot be used with l1-devdax-path")

    memory_config = config.l1_manager_config.memory_config
    if not (memory_config.devdax_path and memory_config.devdax_size_in_bytes):
        return

    incompatible_adapters = [
        adapter_name
        for adapter_config in config.l2_adapter_config.adapters
        if (adapter_name := _requires_single_l1_memory_region(adapter_config))
        is not None
    ]
    if incompatible_adapters:
        raise ValueError(
            "Hybrid DRAM + Device-DAX L1 cannot be used with L2 adapters "
            "that register a single L1 memory region: "
            f"{', '.join(incompatible_adapters)}"
        )


def l1_exposes_single_memory_region(config: StorageManagerConfig) -> bool:
    """Whether L1 is a single memory region a transfer channel can register.

    Args:
        config: Storage manager configuration to inspect.

    Returns:
        ``True`` if L1 is a single registerable memory region, ``False`` for
        GDS L1 or Device-DAX L1.
    """
    l1_config = config.l1_manager_config
    if l1_config.gds_l1_config is not None:
        return False
    if l1_config.memory_config.devdax_path:
        return False
    return True


def add_storage_manager_args(
    parser: argparse.ArgumentParser,
) -> argparse.ArgumentParser:
    """
    Add storage manager configuration arguments to an existing parser.

    This function allows other modules to integrate storage manager arguments
    into their own argument parsers. Arguments are organized into groups to
    avoid naming conflicts with other modules.

    Args:
        parser: The argument parser to add arguments to.

    Returns:
        argparse.ArgumentParser: The same parser with storage manager
            arguments added.

    Example:
        >>> # In another module that needs its own arguments
        >>> parser = argparse.ArgumentParser(description="My Application")
        >>> parser.add_argument("--my-arg", type=str)
        >>> add_storage_manager_args(parser)
        >>> args = parser.parse_args()
        >>> config = parse_args_to_config(args)
    """
    # L1 Memory Manager Config
    memory_group = parser.add_argument_group(
        "L1 Memory Manager", "Configuration for L1 memory manager"
    )
    memory_group.add_argument(
        "--l1-size-gb",
        type=float,
        required=True,
        help="The size of L1 memory in GB.",
    )
    memory_group.add_argument(
        "--l1-use-lazy",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether to use lazy loading for L1 memory. (Default is True)",
    )
    memory_group.add_argument(
        "--l1-init-size-gb",
        type=int,
        default=20,
        help="The initial size (GB) when using lazy allocation. Default is 20.",
    )
    memory_group.add_argument(
        "--l1-align-bytes",
        type=int,
        default=4096,
        help="The alignment size in bytes. Default is 4KB (4096 bytes).",
    )
    memory_group.add_argument(
        "--l1-devdax-path",
        type=str,
        default=None,
        help=(
            "Optional /dev/dax device or mmap-able file to use as the L1 "
            "backing arena. When set, L1 lazy allocation and SHM transfer "
            "advertising must be disabled because the L1 bytes live in the DAX "
            'map. Set --no-l1-use-lazy and --shm-name "". '
            "If a DAX L2 adapter with the same device_path is registered, "
            "that adapter's max_dax_size_gb is used as L1 overflow size."
        ),
    )

    # GDS L1 tier (optional, opt-in via --gds-l1-path)
    gds_group = parser.add_argument_group(
        "GDS L1 tier",
        "Configuration for the GDS slab-file L1 tier. Setting --gds-l1-path "
        "makes the L1 medium an NVMe slab accessed via GPUDirect Storage DMA "
        "(cuFile on NVIDIA, hipFile on AMD ROCm) instead of pinned DRAM; "
        "--l1-size-gb then sizes the slab. Disable byte-array L2 adapters when "
        "this is on.",
    )
    gds_group.add_argument(
        "--gds-l1-path",
        type=str,
        default=None,
        help="NVMe directory path for the GDS L1 slab. Setting this enables GDS L1.",
    )
    gds_group.add_argument(
        "--gds-l1-use-direct-io",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Open the slab file with O_DIRECT (required for the GDS DMA fast "
        "path on ext4). Default True.",
    )
    # L1 Manager Config (TTL settings)
    ttl_group = parser.add_argument_group(
        "L1 Manager TTL", "TTL configuration for L1 manager locks"
    )
    ttl_group.add_argument(
        "--l1-write-ttl-seconds",
        type=int,
        default=600,
        help="Time to live for each object's write lock. Default is 600s.",
    )
    ttl_group.add_argument(
        "--l1-read-ttl-seconds",
        type=int,
        default=300,
        help="Time to live for each object's read lock. Default is 300s.",
    )

    # Eviction Config
    eviction_group = parser.add_argument_group(
        "Eviction Policy", "Configuration for eviction policies"
    )
    eviction_group.add_argument(
        "--eviction-policy",
        type=str,
        choices=["LRU", "IsolatedLRU", "noop"],
        required=True,
        help="The eviction policy to use ('LRU', 'IsolatedLRU', or 'noop'). "
        "'IsolatedLRU' maintains one LRU list per cache_salt and requires "
        "quotas keyed by cache_salt to be configured via the HTTP API.",
    )
    eviction_group.add_argument(
        "--eviction-trigger-watermark",
        type=float,
        default=0.8,
        help="The memory usage watermark to trigger eviction (0.0 to 1.0). "
        "Default is 0.8.",
    )
    eviction_group.add_argument(
        "--eviction-ratio",
        type=float,
        default=0.2,
        help="The fraction of memory to evict when triggered (0.0 to 1.0). "
        "Default is 0.2.",
    )

    # L2 Policies
    # Import here to break circular dependency:
    # config.py <-> storage_controllers (via eviction_controller)
    # Safe because config.py is fully initialized by the time this
    # function is called.
    # First Party
    from lmcache.v1.distributed.storage_controllers.prefetch_policy import (
        get_registered_prefetch_policies,
    )
    from lmcache.v1.distributed.storage_controllers.store_policy import (
        get_registered_store_policies,
    )
    import lmcache.v1.distributed.storage_controllers  # noqa: F401

    policy_group = parser.add_argument_group(
        "L2 Policies", "Store and prefetch policy selection for L2 adapters"
    )
    policy_group.add_argument(
        "--l2-store-policy",
        type=str,
        choices=get_registered_store_policies(),
        default="default",
        help="L2 store policy. Determines which adapters receive each key "
        "and whether keys are deleted from L1 after L2 store. "
        "Default is 'default' (store all keys to all adapters, keep L1).",
    )
    policy_group.add_argument(
        "--l2-prefetch-policy",
        type=str,
        choices=get_registered_prefetch_policies(),
        default="default",
        help="L2 prefetch policy. Determines which adapter loads each key "
        "when multiple adapters have it. "
        "Default is 'default' (pick the first adapter by index).",
    )
    policy_group.add_argument(
        "--l2-prefetch-max-in-flight",
        type=int,
        default=8,
        help="Maximum number of concurrent prefetch requests. Default is 8.",
    )
    policy_group.add_argument(
        "--periodic-notifier-interval-ms",
        type=int,
        default=5,
        help="Interval in ms for the periodic event notifier heartbeat. Default is 5.",
    )

    # Adapter config
    add_l2_adapters_args(parser)
    return parser


def get_arg_parser() -> argparse.ArgumentParser:
    """
    Get a standalone argument parser for storage manager configuration.

    This creates a new parser with only storage manager arguments.
    For integrating with other modules' parsers, use add_storage_manager_args()
    instead.

    Returns:
        argparse.ArgumentParser: The argument parser with all storage manager
            configuration options.
    """
    parser = argparse.ArgumentParser(
        description="Distributed Storage Manager Configuration"
    )
    return add_storage_manager_args(parser)


def parse_args_to_config(
    args: argparse.Namespace,
) -> StorageManagerConfig:
    """
    Convert parsed command line arguments to a StorageManagerConfig.

    Args:
        args: Parsed arguments from the argument parser.

    Returns:
        StorageManagerConfig: The configuration object.
    """
    shm_name = getattr(args, "shm_name", None)
    if shm_name is None:
        memory_config = L1MemoryManagerConfig(
            size_in_bytes=int(args.l1_size_gb * (1 << 30)),
            use_lazy=args.l1_use_lazy,
            init_size_in_bytes=int(args.l1_init_size_gb * (1 << 30)),
            align_bytes=args.l1_align_bytes,
            devdax_path=args.l1_devdax_path,
        )
    else:
        memory_config = L1MemoryManagerConfig(
            size_in_bytes=int(args.l1_size_gb * (1 << 30)),
            use_lazy=args.l1_use_lazy,
            init_size_in_bytes=int(args.l1_init_size_gb * (1 << 30)),
            align_bytes=args.l1_align_bytes,
            shm_name=shm_name,
            devdax_path=args.l1_devdax_path,
        )

    gds_l1_config: GdsL1Config | None = None
    if getattr(args, "gds_l1_path", None):
        # --l1-size-gb is the single L1 size flag; under GDS it sizes the slab.
        gds_l1_config = GdsL1Config(
            file_location=args.gds_l1_path,
            size_in_bytes=int(args.l1_size_gb * (1 << 30)),
            use_direct_io=args.gds_l1_use_direct_io,
        )

    l1_manager_config = L1ManagerConfig(
        memory_config=memory_config,
        gds_l1_config=gds_l1_config,
        write_ttl_seconds=args.l1_write_ttl_seconds,
        read_ttl_seconds=args.l1_read_ttl_seconds,
    )

    eviction_config = EvictionConfig(
        eviction_policy=args.eviction_policy,
        trigger_watermark=args.eviction_trigger_watermark,
        eviction_ratio=args.eviction_ratio,
        extra_logging_enabled=getattr(args, "enable_extra_logging", False),
        extra_logging_interval=getattr(args, "extra_logging_interval", 10.0),
    )

    l2_adapter_config = parse_args_to_l2_adapters_config(args)

    config = StorageManagerConfig(
        l1_manager_config=l1_manager_config,
        eviction_config=eviction_config,
        l2_adapter_config=l2_adapter_config,
        store_policy=args.l2_store_policy,
        prefetch_policy=args.l2_prefetch_policy,
        prefetch_max_in_flight=args.l2_prefetch_max_in_flight,
        periodic_notifier_interval_ms=args.periodic_notifier_interval_ms,
    )
    return config


def parse_args(args: list[str] | None = None) -> StorageManagerConfig:
    """
    Parse command line arguments and return a StorageManagerConfig.

    This is a convenience function that combines get_arg_parser() and
    parse_args_to_config().

    Args:
        args: Optional list of arguments to parse. If None, uses sys.argv.

    Returns:
        StorageManagerConfig: The configuration object.
    """
    parser = get_arg_parser()
    parsed_args = parser.parse_args(args)
    return parse_args_to_config(parsed_args)
