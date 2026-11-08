# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright contributors to the Foundry project
"""Foundry vLLM-integration configuration.

Provides `CUDAGraphExtensionMode`, `CUDAGraphExtensionConfig`,
`load_graph_extension_config`, and the rank-computation helper.
"""

from __future__ import annotations

import enum
import importlib.util
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from vllm.logger import init_logger

if TYPE_CHECKING:
    from vllm.config import ParallelConfig

log = init_logger("vllm.foundry.config")


class CUDAGraphExtensionMode(enum.StrEnum):
    NONE = "none"
    SAVE = "save"
    LOAD = "load"


@dataclass
class CUDAGraphExtensionConfig:
    mode: CUDAGraphExtensionMode = CUDAGraphExtensionMode.NONE
    hook_library_path: str | None = None
    # Path to libnvshmem_host.so for NVSHMEM-based kernels (e.g. DeepEP).
    nvshmem_host_path: str | None = None
    base_addr: int = 0x600000000000
    region_size: str = "64GB"
    workspace_root: str = "foundry_archive"
    workspace_dir: str | None = None
    # Re-capture graphs on LOAD when archived replay is incompatible with the
    # current runtime.  This is a bring-up fallback; the archive path remains
    # the default.
    recapture_on_load: bool = False
    # Scratch space reserved for non-deterministic NCCL allocations.
    # After skip_to_scratch_boundary(), allocations start from
    # base_addr + scratch_space_size.
    scratch_space_size: str = "64MB"
    # Contiguous physical pool for daemon-owned IPC weights. Importing one
    # large allocation preserves its base across independent engine processes.
    weight_cache_size: str | None = None

    @classmethod
    def from_toml(cls, path: str | Path) -> CUDAGraphExtensionConfig:
        with open(path, "rb") as f:
            data = tomllib.load(f)

        mode_str = data.get("mode", cls.mode.value)
        mode = CUDAGraphExtensionMode(mode_str)

        base_addr_value = data.get("base_addr", cls.base_addr)
        base_addr = (
            int(base_addr_value, 0)
            if isinstance(base_addr_value, str)
            else base_addr_value
        )

        hook_library_path = data.get("hook_library_path")
        if hook_library_path is None:
            hook_library_path = cls._detect_hook_so_path()

        return cls(
            mode=mode,
            hook_library_path=hook_library_path,
            nvshmem_host_path=data.get("nvshmem_host_path"),
            base_addr=base_addr,
            region_size=data.get("region_size", cls.region_size),
            workspace_root=data.get("workspace_root", cls.workspace_root),
            scratch_space_size=data.get("scratch_space_size", cls.scratch_space_size),
            weight_cache_size=data.get("weight_cache_size"),
            recapture_on_load=data.get("recapture_on_load", cls.recapture_on_load),
        )

    @staticmethod
    def _detect_hook_so_path() -> str | None:
        spec = importlib.util.find_spec("foundry.ops")
        if spec and spec.origin:
            ops_so_path = Path(spec.origin).resolve()
            hook_so_path = ops_so_path.parent / "libcuda_hook.so"
            if hook_so_path.exists():
                return str(hook_so_path)
        return None


# Module-global config singleton (process-local).
_config: CUDAGraphExtensionConfig | None = None


def load_graph_extension_config(path: str) -> None:
    """Load TOML and set the process-local `_config` singleton.

    Idempotent for the same path; loading a different path replaces.
    """
    global _config
    _config = CUDAGraphExtensionConfig.from_toml(path)


def get_config() -> CUDAGraphExtensionConfig | None:
    return _config


def get_graph_extension_mode() -> CUDAGraphExtensionMode:
    if _config is None:
        return CUDAGraphExtensionMode.NONE
    return _config.mode


def get_workspace_root() -> str | None:
    if _config is None:
        return None
    return _config.workspace_root


def get_workspace_dir() -> str | None:
    if _config is None:
        return None
    return _config.workspace_dir


def get_hook_library_path() -> str | None:
    if _config is None:
        return None
    return _config.hook_library_path


def get_nvshmem_host_path() -> str | None:
    if _config is None:
        return None
    return _config.nvshmem_host_path


def _compute_rank_from_parallel_config(
    parallel_config: ParallelConfig,
) -> int:
    """Unique global rank for workspace naming.

    Latest vLLM resets `data_parallel_rank=0` and `data_parallel_size=1`
    in each non-MoE DP replica (`vllm/v1/engine/core.py` "treat like DP=1"
    branch). `data_parallel_index` is the only DP shard identifier that
    survives that reset. `parallel_config.rank` is the TP/PP rank inside
    the shard; `parallel_config.world_size` is TP*PP*DCP per shard.
    """
    dp_index = parallel_config.data_parallel_index
    return dp_index * parallel_config.world_size + parallel_config.rank


def _uses_deepep_backend(parallel_config: ParallelConfig) -> bool:
    return parallel_config.all2all_backend in (
        "deepep_low_latency",
        "deepep_high_throughput",
    )
