from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Mapping

import torch

from vllm.config import VllmConfig
from vllm.platforms import current_platform
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.kv_offload.abstract import LoadStoreSpec, OffloadingManager
from vllm.v1.kv_offload.mediums import (
    CXLLoadStoreSpec,
    GPULoadStoreSpec,
)
from vllm.v1.kv_offload.spec import OffloadingSpec

from .backends.cxl_backend import LoomCXLBackend
from .handlers.gpu_cxl import LoomGPUCxlOffloadingHandlers
from .manager import LoomManager
from ..logger import get_loom_logger

logger = get_loom_logger(__name__)


def _coerce_int(name: str, value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a number, got {type(value).__name__}")
    int_value = int(value)
    if int_value < 0:
        raise ValueError(f"{name} must be >= 0, got {int_value}")
    return int_value


def _coerce_float(name: str, value: Any) -> float:
    if value is None:
        raise TypeError(f"{name} must be a number, got None")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a number, got {type(value).__name__}")
    return float(value)


def _coerce_fraction(name: str, value: Any) -> float:
    f = _coerce_float(name, value)
    if not (0.0 <= f <= 1.0):
        raise ValueError(f"{name} must be within [0, 1], got {f}")
    return f


def _coerce_bool(name: str, value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in ("1", "true", "t", "yes", "y", "on"):
            return True
        if normalized in ("0", "false", "f", "no", "n", "off"):
            return False
    raise TypeError(f"{name} must be a bool, got {type(value).__name__}")


def _coerce_str(name: str, value: Any) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a str, got {type(value).__name__}")
    return value


def _parse_ratio_or_auto(name: str, value: Any) -> float | Literal["auto"]:
    if isinstance(value, str) and value.strip().lower() == "auto":
        return "auto"
    return _coerce_fraction(name, value)


@dataclass
class LoomOffloadingConfig:
    seed_pool_size_GB: int = 0
    cxl_kvcache_size_GB: int = 0
    # Loom (MVP-0): request-level recompute baseline knobs.
    loom_recompute_ratio: float | Literal["auto"] = 0.0
    loom_disable_store_for_recompute: bool = False
    loom_recompute_log_every_steps: int = 50
    loom_hybrid_head_tokens: int | None = None
    loom_hybrid_head_ratio: float | None = None
    cxl_numa_node: int | None = 1
    eviction_policy: Literal["lru", "arc"] = "lru"

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "LoomOffloadingConfig":
        defaults = cls()

        if "seed_pool_size_GB" in raw:
            seed_pool_size_GB = _coerce_int("seed_pool_size_GB", raw["seed_pool_size_GB"])
        else:
            seed_pool_size_GB = defaults.seed_pool_size_GB

        if "cxl_kvcache_size_GB" in raw:
            cxl_kvcache_size_GB = _coerce_int(
                "cxl_kvcache_size_GB", raw["cxl_kvcache_size_GB"]
            )
        else:
            cxl_kvcache_size_GB = defaults.cxl_kvcache_size_GB

        loom_recompute_ratio = defaults.loom_recompute_ratio
        if "loom_recompute_ratio" in raw:
            loom_recompute_ratio = _parse_ratio_or_auto(
                "loom_recompute_ratio", raw["loom_recompute_ratio"]
            )

        loom_disable_store_for_recompute = defaults.loom_disable_store_for_recompute
        if "loom_disable_store_for_recompute" in raw:
            loom_disable_store_for_recompute = _coerce_bool(
                "loom_disable_store_for_recompute",
                raw["loom_disable_store_for_recompute"],
            )

        loom_recompute_log_every_steps = defaults.loom_recompute_log_every_steps
        if "loom_recompute_log_every_steps" in raw:
            loom_recompute_log_every_steps = _coerce_int(
                "loom_recompute_log_every_steps", raw["loom_recompute_log_every_steps"]
            )

        loom_hybrid_head_tokens = defaults.loom_hybrid_head_tokens
        if "loom_hybrid_head_tokens" in raw:
            loom_hybrid_head_tokens = _coerce_int(
                "loom_hybrid_head_tokens", raw["loom_hybrid_head_tokens"]
            )
            if loom_hybrid_head_tokens < 0:
                raise ValueError("loom_hybrid_head_tokens must be >= 0")

        loom_hybrid_head_ratio = defaults.loom_hybrid_head_ratio
        if "loom_hybrid_head_ratio" in raw:
            loom_hybrid_head_ratio = _coerce_fraction(
                "loom_hybrid_head_ratio", raw["loom_hybrid_head_ratio"]
            )

        cxl_numa_node = defaults.cxl_numa_node
        if "cxl_numa_node" in raw:
            if raw["cxl_numa_node"] is None:
                cxl_numa_node = None
            else:
                cxl_numa_node = _coerce_int("cxl_numa_node", raw["cxl_numa_node"])

        eviction_policy = defaults.eviction_policy
        if "eviction_policy" in raw:
            eviction_policy = _coerce_str("eviction_policy", raw["eviction_policy"])
            if eviction_policy not in ("lru", "arc"):
                raise ValueError("eviction_policy must be one of: lru, arc")

        return cls(
            seed_pool_size_GB=seed_pool_size_GB,
            cxl_kvcache_size_GB=cxl_kvcache_size_GB,
            loom_recompute_ratio=loom_recompute_ratio,
            loom_disable_store_for_recompute=loom_disable_store_for_recompute,
            loom_recompute_log_every_steps=loom_recompute_log_every_steps,
            loom_hybrid_head_tokens=loom_hybrid_head_tokens,
            loom_hybrid_head_ratio=loom_hybrid_head_ratio,
            cxl_numa_node=cxl_numa_node,
            eviction_policy=eviction_policy,
        )


class LoomOffloadingSpec(OffloadingSpec):
    def __init__(self, config: VllmConfig, kv_cache_config: KVCacheConfig | None):
        super().__init__(config, kv_cache_config)
        extra = self.extra_config
        if isinstance(extra, LoomOffloadingConfig):
            self.loom_config = extra
        elif isinstance(extra, Mapping):
            try:
                self.loom_config = LoomOffloadingConfig.from_dict(extra)
            except KeyError as e:
                raise ValueError(
                    "loom offloading config is missing required key: "
                    f"{e.args[0]!r}"
                ) from e
        else:
            raise TypeError(
                "loom offloading config must be a dict-like mapping or "
                f"LoomOffloadingConfig, got {type(extra).__name__}"
            )

        # Scheduler-side
        self._manager: OffloadingManager | None = None

        # Worker-side
        self._cxl_handlers: LoomGPUCxlOffloadingHandlers | None = None

        # Optional tuning knobs
        self.eviction_policy = self.loom_config.eviction_policy

    def _get_kv_bytes_per_offloaded_block(self) -> int:
        if self.kv_cache_config is None:
            raise ValueError("kv_cache_config must be provided for LoomOffloadingSpec")

        page_sizes = {
            kv_cache_group.kv_cache_spec.page_size_bytes
            for kv_cache_group in self.kv_cache_config.kv_cache_groups
        }
        if len(page_sizes) != 1:
            raise ValueError(
                "Expected a single page size across kv_cache_groups, got: "
                f"{sorted(page_sizes)}"
            )
        page_size_bytes = next(iter(page_sizes))

        kv_bytes_per_block = (
            page_size_bytes
            * len(self.kv_cache_config.kv_cache_tensors)
            * self.vllm_config.parallel_config.world_size
        )
        return kv_bytes_per_block * (self.offloaded_block_size // self.gpu_block_size)

    def get_manager(self) -> OffloadingManager:
        if self._manager is None:
            kv_bytes_per_offloaded_block = self._get_kv_bytes_per_offloaded_block()
            if kv_bytes_per_offloaded_block <= 0:
                num_cxl_blocks = 0
            else:
                cxl_bytes_to_use = self.loom_config.cxl_kvcache_size_GB * (1024**3)
                num_cxl_blocks = (
                    cxl_bytes_to_use // kv_bytes_per_offloaded_block
                )

            kv_events_config = self.vllm_config.kv_events_config
            enable_events = (
                kv_events_config is not None and kv_events_config.enable_kv_cache_events
            )

            cxl_backend = LoomCXLBackend(
                block_size=self.offloaded_block_size,
                num_blocks=num_cxl_blocks,
                numa_node=self.loom_config.cxl_numa_node,
            )

            self._manager = LoomManager(
                cxl_backend=cxl_backend,
                enable_events=enable_events,
            )

        return self._manager

    def get_handlers(
        self,
        kv_caches: dict[str, torch.Tensor],
        attn_backends: dict[str, type],
    ):
        # Keep signature compatible with OffloadingSpec: Iterator[(src_type, dst_type, handler)]
        if not current_platform.is_cuda_alike():
            raise RuntimeError(
                "LoomOffloadingSpec (CPU-backed) is currently only supported on CUDA-alike GPUs"
            )

        if self._cxl_handlers is None:
            kv_bytes_per_offloaded_block = self._get_kv_bytes_per_offloaded_block()
            if kv_bytes_per_offloaded_block <= 0:
                num_cxl_blocks = 0
            else:
                cxl_bytes_to_use = self.loom_config.cxl_kvcache_size_GB * (1024**3)
                num_cxl_blocks = (
                    cxl_bytes_to_use // kv_bytes_per_offloaded_block
                )

            self._cxl_handlers = LoomGPUCxlOffloadingHandlers(
                attn_backends=attn_backends,
                gpu_block_size=self.gpu_block_size,
                cxl_block_size=self.offloaded_block_size,
                num_cxl_blocks=num_cxl_blocks,
                gpu_caches=kv_caches,
                numa_node=self.loom_config.cxl_numa_node,
            )

        assert self._cxl_handlers is not None

        yield GPULoadStoreSpec, CXLLoadStoreSpec, self._cxl_handlers.gpu_to_cxl_handler
        yield CXLLoadStoreSpec, GPULoadStoreSpec, self._cxl_handlers.cxl_to_gpu_handler