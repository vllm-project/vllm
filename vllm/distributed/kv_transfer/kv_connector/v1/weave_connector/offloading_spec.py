from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, Mapping, Literal

from vllm.config import VllmConfig
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.kv_offload.spec import OffloadingSpec




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


@dataclass
class WeaveOffloadingConfig:
    dram_pool_size_gb: int
    cxl_pool_size_gb: int
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
        dram_pool_size_gb = _coerce_int("dram_pool_size_gb", raw["dram_pool_size_gb"])
        cxl_pool_size_gb = _coerce_int("cxl_pool_size_gb", raw["cxl_pool_size_gb"])
        mode = _parse_mode(raw["mode"])
        return cls(
            dram_pool_size_gb=dram_pool_size_gb,
            cxl_pool_size_gb=cxl_pool_size_gb,
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