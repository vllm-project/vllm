# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""FlashInfer autotune cache helpers."""

import os
import tempfile
from contextlib import suppress
from pathlib import Path
from typing import TYPE_CHECKING

import vllm.envs as envs
from vllm.config import CacheConfig, VllmConfig, replace
from vllm.config.utils import hash_factors

if TYPE_CHECKING:
    from vllm.distributed.parallel_state import GroupCoordinator
    from vllm.v1.worker.gpu.model_runner import GPUModelRunner


def _normalize_cache_config_for_hash(cache_config: CacheConfig) -> CacheConfig:
    """Normalize cache config for hashing"""
    if cache_config.mamba_block_size is None and cache_config.kv_cache_layout is None:
        return cache_config
    normalized = replace(cache_config, mamba_block_size=None)
    assert normalized.kv_cache_layout is None
    return normalized


def flashinfer_autotune_cache_hash(runner: "GPUModelRunner") -> str:
    from vllm.model_executor.layers.fused_moe.routed_experts import RoutedExperts

    vllm_config: VllmConfig = runner.vllm_config
    cache_config = _normalize_cache_config_for_hash(vllm_config.cache_config)
    if cache_config is not vllm_config.cache_config:
        vllm_config = replace(vllm_config, cache_config=cache_config)
    factors = vllm_config.compile_factors(include_version=False)
    # DP runs with padding skip tune TRT-LLM MoE's precomputed-routing ops
    # instead of the logits ops; a cache from before lacks them, and a hit
    # skips all tuning.
    if (
        vllm_config.parallel_config.data_parallel_size > 1
        and envs.VLLM_MOE_SKIP_PADDING
        and not envs.VLLM_MOE_ROUTING_SIMULATION_STRATEGY
        and any(
            isinstance(m, RoutedExperts) and m.supports_precomputed_routing
            for m in runner.get_model().modules()
        )
    ):
        factors = {**factors, "moe_routed": True}
    return hash_factors(factors)


def flashinfer_autotune_root() -> Path:
    """Root directory of the FlashInfer autotune cache (no runner needed)."""
    override_dir = envs.VLLM_FLASHINFER_AUTOTUNE_CACHE_DIR
    if override_dir:
        return Path(override_dir).expanduser()
    from flashinfer.jit import env as flashinfer_jit_env

    flashinfer_workspace = flashinfer_jit_env.FLASHINFER_WORKSPACE_DIR
    return (
        Path(envs.VLLM_CACHE_ROOT)
        / "flashinfer_autotune_cache"
        / flashinfer_workspace.parent.name
        / flashinfer_workspace.name
    )


def flashinfer_autotune_filename(dp_rank: int, rank: int) -> str:
    """Per-rank cache filename, shared by the engine reader and D2D exchange."""
    return f"autotune_configs_dp{dp_rank}_rank{rank}.json"


def resolve_flashinfer_autotune_file(runner: "GPUModelRunner") -> Path:
    output_dir = flashinfer_autotune_root() / flashinfer_autotune_cache_hash(runner)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir / "autotune_configs.json"


def resolve_rank_flashinfer_autotune_file(runner: "GPUModelRunner") -> Path:
    """The per-rank variant of `resolve_flashinfer_autotune_file`."""
    from vllm.distributed.parallel_state import get_world_group

    path = resolve_flashinfer_autotune_file(runner)
    dp_rank = runner.vllm_config.parallel_config.data_parallel_rank
    return path.with_name(
        flashinfer_autotune_filename(dp_rank, get_world_group().rank_in_group)
    )


def use_flashinfer_autotune_v2(runner: "GPUModelRunner") -> bool:
    from vllm.distributed.parallel_state import get_world_group
    from vllm.utils.flashinfer import has_flashinfer_autotune_v2

    # Elastic EP transfers tuning state to joining workers with v1
    # save_configs/load_configs, which do not export managed-store winners.
    if runner.vllm_config.parallel_config.enable_elastic_ep:
        return False
    # Multi-rank preload daemons publish legacy tables for their IPC engines.
    if (
        get_world_group().world_size > 1
        and runner.vllm_config.load_config.load_format == "ipc_cache"
    ):
        return False
    return has_flashinfer_autotune_v2()


def resolve_flashinfer_autotune_v2_root(runner: "GPUModelRunner") -> Path | None:
    """Isolate distributed writers while FlashInfer owns entry compatibility."""
    from vllm.distributed.parallel_state import get_world_group

    world = get_world_group()
    if world.world_size > 1:
        dp_rank = runner.vllm_config.parallel_config.data_parallel_rank
        return (
            flashinfer_autotune_root()
            / flashinfer_autotune_cache_hash(runner)
            / f"managed_dp{dp_rank}_rank{world.rank_in_group}"
        )
    override_dir = envs.VLLM_FLASHINFER_AUTOTUNE_CACHE_DIR
    return Path(override_dir).expanduser() if override_dir else None


def sync_flashinfer_autotune_cache(
    runner: "GPUModelRunner",
    group: "GroupCoordinator",
) -> None:
    cache: bytes | str | None = None
    if (
        group.rank_in_group == 0
        and runner.vllm_config.kernel_config.enable_flashinfer_autotune
    ):
        try:
            from vllm.platforms import current_platform
            from vllm.utils.flashinfer import has_flashinfer

            if has_flashinfer() and current_platform.has_device_capability(90):
                from flashinfer.autotuner import AutoTuner

                with tempfile.TemporaryDirectory() as temp_dir:
                    path = Path(temp_dir) / "autotune_configs.json"
                    AutoTuner.get().save_configs(str(path))
                    cache = path.read_bytes()
        except Exception as exc:
            cache = f"{type(exc).__name__}: {exc}"

    cache = group.broadcast_object(cache)
    if isinstance(cache, str):
        raise RuntimeError(f"Failed to serialize FlashInfer autotune state: {cache}")
    if cache is None or group.rank_in_group == 0:
        return

    from flashinfer.autotuner import AutoTuner

    with tempfile.NamedTemporaryFile() as f:
        f.write(cache)
        f.flush()
        if not AutoTuner.get().load_configs(f.name):
            raise RuntimeError("FlashInfer autotune cache is incompatible")


def write_flashinfer_autotune_cache(cache_path: Path, contents: bytes) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(
        dir=cache_path.parent, suffix=".tmp", prefix=f".{cache_path.name}."
    )
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(contents)
        os.replace(tmp_path, cache_path)
    except BaseException:
        with suppress(OSError):
            os.unlink(tmp_path)
        raise
