from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, Literal, Mapping

import torch

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.kv_offload.abstract import LoadStoreSpec, OffloadingManager
from vllm.v1.kv_offload.arc_manager import ARCOffloadingManager
from vllm.v1.kv_offload.lru_manager import LRUOffloadingManager
from vllm.v1.kv_offload.mediums import (
    CXLLoadStoreSpec,
    DRAMLoadStoreSpec,
    GPULoadStoreSpec,
)
from vllm.v1.kv_offload.spec import OffloadingSpec
from vllm.v1.kv_offload.worker.worker import OffloadingHandler

from .cxl_backend import WeaveCXLBackend
from .dram_cxl import WeaveDramCxlOffloadingHandler
from .cpu_gpu import WeaveGPUDramOffloadingHandlers
from .dram_backend import WeaveDRAMBackend
from .numa import numa_membind
from .two_tier_manager import TwoTierOffloadingManager

logger = init_logger(__name__)




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


def _parse_ratio_or_auto(name: str, value: Any) -> float | Literal["auto"]:
    if isinstance(value, str) and value.strip().lower() == "auto":
        return "auto"
    return _coerce_fraction(name, value)


def _parse_mode(value: Any) -> WeaveOffloadingMode:
    if isinstance(value, WeaveOffloadingMode):
        return value
    if isinstance(value, str):
        normalized = value.strip().upper().replace("-", "_")
        # allow a couple of common aliases
        aliases = {
            "DRAM": "DRAM_ONLY",
            "CXL": "CXL_ONLY",
        }
        normalized = aliases.get(normalized, normalized)
        try:
            return WeaveOffloadingMode[normalized]
        except KeyError as e:
            valid = ", ".join(m.name for m in WeaveOffloadingMode)
            raise ValueError(f"mode must be one of: {valid}; got {value!r}") from e
    raise TypeError(
        f"mode must be a str or WeaveOffloadingMode, got {type(value).__name__}"
    )


class WeaveOffloadingMode(Enum):
    DRAM_ONLY = auto()
    CXL_ONLY = auto()
    DEFAULT = auto()


def _coerce_bytes(name: str, value: Any) -> int:
    if value is None:
        raise TypeError(f"{name} must be a number, got None")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a number, got {type(value).__name__}")
    int_value = int(value)
    if int_value < 0:
        raise ValueError(f"{name} must be >= 0, got {int_value}")
    return int_value


def _gb_to_bytes(gb: Any) -> int:
    # Accept int/float; treat as GiB for consistency with most tooling.
    gb_int = _coerce_int("*_pool_size_gb", gb)
    return gb_int * (1024**3)


@dataclass
class WeaveOffloadingConfig:
    dram_bytes_to_use: int = 0
    cxl_bytes_to_use: int = 0
    mode: WeaveOffloadingMode = WeaveOffloadingMode.DEFAULT
    
    # 水位控制
    dram_high_watermark: float = 0.8
    dram_low_watermark:  float = 0.6
    
    # 写入配置
    kv_prefill_dram_ratio: float | Literal["auto"] = 0.67
    flush_batch_size_MB: int = 64
    flush_budget_MBps: int = 256 

    # decode 
    kv_hot_window_tokens: int = 512   
    kv_prefetch_blocks: int = 2
    promotion_budget_MBps: int = 256  
    decode_allow_sync_cxl_read: bool = True

    cxl_numa_node: int | None = None
    

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "WeaveOffloadingConfig":
        # Accept a few aliases so users can start with minimal configs.
        # Preferred keys:
        # - dram_bytes_to_use / cxl_bytes_to_use
        # - dram_pool_size_gb / cxl_pool_size_gb
        # Back-compat with vLLM CPUOffloadingSpec:
        # - cpu_bytes_to_use -> dram_bytes_to_use

        if "dram_bytes_to_use" in raw:
            dram_bytes_to_use = _coerce_bytes("dram_bytes_to_use", raw["dram_bytes_to_use"])
        elif "cpu_bytes_to_use" in raw:
            dram_bytes_to_use = _coerce_bytes("cpu_bytes_to_use", raw["cpu_bytes_to_use"])
        elif "dram_pool_size_gb" in raw:
            dram_bytes_to_use = _gb_to_bytes(raw["dram_pool_size_gb"])
        else:
            raise ValueError(
                "weave offloading config must specify one of: "
                "dram_bytes_to_use, cpu_bytes_to_use, dram_pool_size_gb"
            )

        if "cxl_bytes_to_use" in raw:
            cxl_bytes_to_use = _coerce_bytes("cxl_bytes_to_use", raw["cxl_bytes_to_use"])
        elif "cxl_pool_size_gb" in raw:
            cxl_bytes_to_use = _gb_to_bytes(raw["cxl_pool_size_gb"])
        else:
            cxl_bytes_to_use = 0

        mode = _parse_mode(raw.get("mode", WeaveOffloadingMode.DEFAULT))

        defaults = cls()

        dram_high_watermark = defaults.dram_high_watermark
        if "dram_high_watermark" in raw:
            dram_high_watermark = _coerce_fraction(
                "dram_high_watermark", raw["dram_high_watermark"]
            )

        dram_low_watermark = defaults.dram_low_watermark
        if "dram_low_watermark" in raw:
            dram_low_watermark = _coerce_fraction(
                "dram_low_watermark", raw["dram_low_watermark"]
            )

        if dram_low_watermark > dram_high_watermark:
            raise ValueError(
                "dram_low_watermark must be <= dram_high_watermark, got: "
                f"{dram_low_watermark} > {dram_high_watermark}"
            )

        kv_prefill_dram_ratio = defaults.kv_prefill_dram_ratio
        if "kv_prefill_dram_ratio" in raw:
            kv_prefill_dram_ratio = _parse_ratio_or_auto(
                "kv_prefill_dram_ratio", raw["kv_prefill_dram_ratio"]
            )

        flush_batch_size_MB = defaults.flush_batch_size_MB
        if "flush_batch_size_MB" in raw:
            flush_batch_size_MB = _coerce_int(
                "flush_batch_size_MB", raw["flush_batch_size_MB"]
            )

        flush_budget_MBps = defaults.flush_budget_MBps
        if "flush_budget_MBps" in raw:
            flush_budget_MBps = _coerce_int("flush_budget_MBps", raw["flush_budget_MBps"])

        kv_hot_window_tokens = defaults.kv_hot_window_tokens
        if "kv_hot_window_tokens" in raw:
            kv_hot_window_tokens = _coerce_int(
                "kv_hot_window_tokens", raw["kv_hot_window_tokens"]
            )

        kv_prefetch_blocks = defaults.kv_prefetch_blocks
        if "kv_prefetch_blocks" in raw:
            kv_prefetch_blocks = _coerce_int(
                "kv_prefetch_blocks", raw["kv_prefetch_blocks"]
            )

        promotion_budget_MBps = defaults.promotion_budget_MBps
        if "promotion_budget_MBps" in raw:
            promotion_budget_MBps = _coerce_int(
                "promotion_budget_MBps", raw["promotion_budget_MBps"]
            )

        decode_allow_sync_cxl_read = defaults.decode_allow_sync_cxl_read
        if "decode_allow_sync_cxl_read" in raw:
            decode_allow_sync_cxl_read = _coerce_bool(
                "decode_allow_sync_cxl_read", raw["decode_allow_sync_cxl_read"]
            )

        cxl_numa_node = defaults.cxl_numa_node
        if "cxl_numa_node" in raw and raw["cxl_numa_node"] is not None:
            cxl_numa_node = _coerce_int("cxl_numa_node", raw["cxl_numa_node"])
        return cls(
            dram_bytes_to_use=dram_bytes_to_use,
            cxl_bytes_to_use=cxl_bytes_to_use,
            mode=mode,
            dram_high_watermark=dram_high_watermark,
            dram_low_watermark=dram_low_watermark,
            kv_prefill_dram_ratio=kv_prefill_dram_ratio,
            flush_batch_size_MB=flush_batch_size_MB,
            flush_budget_MBps=flush_budget_MBps,
            kv_hot_window_tokens=kv_hot_window_tokens,
            kv_prefetch_blocks=kv_prefetch_blocks,
            promotion_budget_MBps=promotion_budget_MBps,
            decode_allow_sync_cxl_read=decode_allow_sync_cxl_read,
            cxl_numa_node=cxl_numa_node,
        )

class WeaveOffloadingSpec(OffloadingSpec):
    def __init__(self, config: VllmConfig, kv_cache_config: KVCacheConfig | None):
        super().__init__(config, kv_cache_config)
        extra = self.extra_config
        if isinstance(extra, WeaveOffloadingConfig):
            self.weave_config = extra
        elif isinstance(extra, Mapping):
            try:
                self.weave_config = WeaveOffloadingConfig.from_dict(extra)
            except KeyError as e:
                raise ValueError(
                    "weave offloading config is missing required key: "
                    f"{e.args[0]!r}"
                ) from e
        else:
            raise TypeError(
                "weave offloading config must be a dict-like mapping or "
                f"WeaveOffloadingConfig, got {type(extra).__name__}"
            )

        # Scheduler-side
        self._manager: OffloadingManager | None = None

        # Worker-side
        self._dram_handlers: WeaveGPUDramOffloadingHandlers | None = None
        self._cxl_handlers: WeaveGPUDramOffloadingHandlers | None = None
        self._dram_cxl_dram_to_cxl: WeaveDramCxlOffloadingHandler | None = None
        self._dram_cxl_cxl_to_dram: WeaveDramCxlOffloadingHandler | None = None

        # Optional tuning knobs
        self.eviction_policy: str = (
            extra.get("eviction_policy", "lru") if isinstance(extra, Mapping) else "lru"
        )

    def _get_kv_bytes_per_offloaded_block(self) -> int:
        if self.kv_cache_config is None:
            raise ValueError("kv_cache_config must be provided for WeaveOffloadingSpec")

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
                num_dram_blocks = 0
                num_cxl_blocks = 0
            else:
                num_dram_blocks = (
                    self.weave_config.dram_bytes_to_use // kv_bytes_per_offloaded_block
                )
                num_cxl_blocks = (
                    self.weave_config.cxl_bytes_to_use // kv_bytes_per_offloaded_block
                )

            kv_events_config = self.vllm_config.kv_events_config
            enable_events = (
                kv_events_config is not None and kv_events_config.enable_kv_cache_events
            )

            dram_backend = WeaveDRAMBackend(
                block_size=self.offloaded_block_size,
                num_blocks=num_dram_blocks,
            )
            cxl_backend = WeaveCXLBackend(
                block_size=self.offloaded_block_size,
                num_blocks=num_cxl_blocks,
                numa_node=self.weave_config.cxl_numa_node,
            )

            self._manager = TwoTierOffloadingManager(
                dram_backend=dram_backend,
                cxl_backend=cxl_backend,
                enable_events=enable_events,
            )
            
            if num_dram_blocks == 0:
                logger.warning(
                    "WeaveOffloadingSpec initialized with 0 DRAM offload blocks. "
                    "Offloading will likely be disabled. "
                    "Please increase dram_bytes_to_use (or cpu_bytes_to_use)."
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
                "WeaveOffloadingSpec (CPU-backed) is currently only supported on CUDA-alike GPUs"
            )

        if self._dram_handlers is None or self._cxl_handlers is None:
            kv_bytes_per_offloaded_block = self._get_kv_bytes_per_offloaded_block()
            if kv_bytes_per_offloaded_block <= 0:
                num_dram_blocks = 0
                num_cxl_blocks = 0
            else:
                num_dram_blocks = (
                    self.weave_config.dram_bytes_to_use // kv_bytes_per_offloaded_block
                )
                num_cxl_blocks = (
                    self.weave_config.cxl_bytes_to_use // kv_bytes_per_offloaded_block
                )

            self._dram_handlers = WeaveGPUDramOffloadingHandlers(
                attn_backends=attn_backends,
                gpu_block_size=self.gpu_block_size,
                cpu_block_size=self.offloaded_block_size,
                num_cpu_blocks=num_dram_blocks,
                gpu_caches=kv_caches,
            )

            with numa_membind(self.weave_config.cxl_numa_node):
                self._cxl_handlers = WeaveGPUDramOffloadingHandlers(
                    attn_backends=attn_backends,
                    gpu_block_size=self.gpu_block_size,
                    cpu_block_size=self.offloaded_block_size,
                    num_cpu_blocks=num_cxl_blocks,
                    gpu_caches=kv_caches,
                )

            self._dram_cxl_dram_to_cxl = WeaveDramCxlOffloadingHandler(
                src_tensors=self._dram_handlers.cpu_tensors,
                dst_tensors=self._cxl_handlers.cpu_tensors,
                kv_dim_before_num_blocks=self._dram_handlers.kv_dim_before_num_blocks,
                src_block_size_factor=self._dram_handlers.cpu_block_size_factor,
                dst_block_size_factor=self._cxl_handlers.cpu_block_size_factor,
            )
            self._dram_cxl_cxl_to_dram = WeaveDramCxlOffloadingHandler(
                src_tensors=self._cxl_handlers.cpu_tensors,
                dst_tensors=self._dram_handlers.cpu_tensors,
                kv_dim_before_num_blocks=self._dram_handlers.kv_dim_before_num_blocks,
                src_block_size_factor=self._cxl_handlers.cpu_block_size_factor,
                dst_block_size_factor=self._dram_handlers.cpu_block_size_factor,
            )

        assert self._dram_handlers is not None
        assert self._cxl_handlers is not None
        assert self._dram_cxl_dram_to_cxl is not None
        assert self._dram_cxl_cxl_to_dram is not None

        yield GPULoadStoreSpec, DRAMLoadStoreSpec, self._dram_handlers.gpu_to_cpu_handler
        yield DRAMLoadStoreSpec, GPULoadStoreSpec, self._dram_handlers.cpu_to_gpu_handler
        yield DRAMLoadStoreSpec, CXLLoadStoreSpec, self._dram_cxl_dram_to_cxl
        yield CXLLoadStoreSpec, DRAMLoadStoreSpec, self._dram_cxl_cxl_to_dram