# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""QuestConfig: configuration for the Quest sparse offload attention backend.

Phase A only carries plumbing. Tiering / async / kernel fields are present so
later phases can flip them without re-introducing config-shape churn — but
they are validated and have safe defaults that keep Phase A behavior equal to
FlashAttention with the gate flipped.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

EvictionPolicy = Literal["lru", "arc"]
SelectionImpl = Literal["torch", "triton", "cuda"]
UnsupportedModelPolicy = Literal["error", "fallback"]


@dataclass
class QuestConfig:
    enabled: bool = False
    backend_name: str = "QUEST_SPARSE_OFFLOAD"

    # Quest algorithm (Phase B activates these).
    block_size: int = 32
    top_k: int = 64
    full_kv_layers: list[int] = field(default_factory=lambda: [0, 1])

    # GPU/CPU tiering (Phase B activates these).
    gpu_cache_blocks_per_seq: int = 256
    cpu_cache_blocks: int = 65536
    eviction_policy: EvictionPolicy = "lru"

    # Async (Phase C activates these).
    enable_async_prefetch: bool = False
    enable_double_buffering: bool = False
    num_h2d_streams: int = 1
    num_d2h_streams: int = 1
    prefetch_window_blocks: int = 0

    # Kernel dispatch (Phase D activates "cuda").
    selection_impl: SelectionImpl = "torch"

    # Debug.
    enable_debug_counters: bool = False

    # Compatibility behavior when the loaded model isn't whitelisted.
    unsupported_model_policy: UnsupportedModelPolicy = "error"

    def validate(self) -> None:
        if self.top_k <= 0:
            raise ValueError(f"top_k must be positive, got {self.top_k}")
        if self.top_k > self.gpu_cache_blocks_per_seq:
            raise ValueError(
                f"top_k ({self.top_k}) must be <= "
                f"gpu_cache_blocks_per_seq ({self.gpu_cache_blocks_per_seq})"
            )
        if self.block_size <= 0:
            raise ValueError(
                f"block_size must be positive, got {self.block_size}"
            )
        if self.cpu_cache_blocks < 0:
            raise ValueError(
                f"cpu_cache_blocks must be >= 0, got {self.cpu_cache_blocks}"
            )
        if self.eviction_policy not in ("lru", "arc"):
            raise ValueError(
                f"eviction_policy must be 'lru' or 'arc', "
                f"got {self.eviction_policy!r}"
            )
        if self.selection_impl not in ("torch", "triton", "cuda"):
            raise ValueError(
                f"selection_impl must be 'torch', 'triton', or 'cuda', "
                f"got {self.selection_impl!r}"
            )
        if self.unsupported_model_policy not in ("error", "fallback"):
            raise ValueError(
                f"unsupported_model_policy must be 'error' or 'fallback', "
                f"got {self.unsupported_model_policy!r}"
            )
        if not isinstance(self.full_kv_layers, list) or not all(
            isinstance(x, int) for x in self.full_kv_layers
        ):
            raise ValueError(
                f"full_kv_layers must be a list of int, "
                f"got {self.full_kv_layers!r}"
            )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "QuestConfig":
        return cls(**data)
