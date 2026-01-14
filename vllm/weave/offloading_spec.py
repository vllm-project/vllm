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
from vllm.v1.kv_offload.backends.cpu import CPUBackend
from vllm.v1.kv_offload.lru_manager import LRUOffloadingManager
from vllm.v1.kv_offload.mediums import CPULoadStoreSpec, GPULoadStoreSpec
from vllm.v1.kv_offload.spec import OffloadingSpec
from vllm.v1.kv_offload.worker.cpu_gpu import CpuGpuOffloadingHandlers
from vllm.v1.kv_offload.worker.worker import OffloadingHandler

logger = init_logger(__name__)




def _coerce_int(name: str, value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a number, got {type(value).__name__}")
    int_value = int(value)
    if int_value < 0:
        raise ValueError(f"{name} must be >= 0, got {int_value}")
    return int_value


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
    # NOTE: for now we use vLLM's existing CPU offloading backend as a stand-in
    # for the DRAM tier. CXL tier will be wired in later.
    dram_bytes_to_use: int
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
        return cls(
            dram_bytes_to_use=dram_bytes_to_use,
            cxl_bytes_to_use=cxl_bytes_to_use,
            mode=mode,
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
        self._handlers: CpuGpuOffloadingHandlers | None = None

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
                num_blocks = 0
            else:
                num_blocks = self.weave_config.dram_bytes_to_use // kv_bytes_per_offloaded_block

            kv_events_config = self.vllm_config.kv_events_config
            enable_events = (
                kv_events_config is not None and kv_events_config.enable_kv_cache_events
            )

            backend = CPUBackend(block_size=self.offloaded_block_size, num_blocks=num_blocks)

            if self.eviction_policy == "lru":
                self._manager = LRUOffloadingManager(backend=backend, enable_events=enable_events)
            elif self.eviction_policy == "arc":
                self._manager = ARCOffloadingManager(backend=backend, enable_events=enable_events)
            else:
                raise ValueError(
                    f"Unknown eviction policy: {self.eviction_policy}. Supported policies: lru, arc"
                )
            
            if num_blocks == 0:
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

        if self._handlers is None:
            kv_bytes_per_offloaded_block = self._get_kv_bytes_per_offloaded_block()
            if kv_bytes_per_offloaded_block <= 0:
                num_cpu_blocks = 0
            else:
                num_cpu_blocks = self.weave_config.dram_bytes_to_use // kv_bytes_per_offloaded_block

            self._handlers = CpuGpuOffloadingHandlers(
                attn_backends=attn_backends,
                gpu_block_size=self.gpu_block_size,
                cpu_block_size=self.offloaded_block_size,
                num_cpu_blocks=num_cpu_blocks,
                gpu_caches=kv_caches,
            )

        assert self._handlers is not None
        yield GPULoadStoreSpec, CPULoadStoreSpec, self._handlers.gpu_to_cpu_handler
        yield CPULoadStoreSpec, GPULoadStoreSpec, self._handlers.cpu_to_gpu_handler