# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""FlashInfer autotune cache helpers."""

import json
import os
import tempfile
from collections.abc import Callable
from contextlib import suppress
from pathlib import Path
from typing import TYPE_CHECKING

import torch

import vllm.envs as envs
from vllm.config import CacheConfig, MultiModalConfig, VllmConfig, replace
from vllm.config.utils import hash_config, hash_factors, normalize_value
from vllm.logger import init_logger

if TYPE_CHECKING:
    from flashinfer.autotuner import AutoTuner
    from torch import nn

    from vllm.distributed.parallel_state import GroupCoordinator
    from vllm.v1.worker.gpu.model_runner import GPUModelRunner

logger = init_logger(__name__)


def _normalize_cache_config_for_hash(cache_config: CacheConfig) -> CacheConfig:
    """Normalize cache config for hashing"""
    if cache_config.mamba_block_size is None and cache_config.kv_cache_layout is None:
        return cache_config
    normalized = replace(cache_config, mamba_block_size=None)
    assert normalized.kv_cache_layout is None
    return normalized


def _flashinfer_mm_encoder_autotune_hash(
    mm_config: MultiModalConfig, *, enable_chunked_prefill: bool
) -> str:
    return hash_factors(
        {
            "encoder_config": hash_config(mm_config),
            "enable_chunked_prefill": enable_chunked_prefill,
            "video_pruning_rate": mm_config.video_pruning_rate,
            "video_pruning_method": mm_config.video_pruning_method,
            "mm_processor_kwargs": normalize_value(mm_config.mm_processor_kwargs),
            "limit_per_prompt": normalize_value(mm_config.limit_per_prompt),
        }
    )


def get_flashinfer_mm_encoder_autotune_config(
    vllm_config: VllmConfig,
) -> MultiModalConfig | None:
    """Return the multimodal config when encoder autotuning is enabled."""
    model_config = vllm_config.model_config
    if (
        model_config is None
        or not model_config.supports_multimodal_inputs
        or not vllm_config.kernel_config.enable_flashinfer_mm_encoder_autotune
    ):
        return None
    mm_config = model_config.multimodal_config
    if mm_config is None or mm_config.skip_mm_profiling:
        return None
    return mm_config


def flashinfer_autotune_cache_hash(runner: "GPUModelRunner") -> str:
    # The weight cache may request the key before the runner has a model.
    return _flashinfer_autotune_cache_hash(
        runner.vllm_config, lambda: runner.get_model()
    )


def _flashinfer_autotune_cache_hash(
    vllm_config: VllmConfig, get_model: Callable[[], "nn.Module"] | None = None
) -> str:
    from vllm.model_executor.layers.fused_moe.routed_experts import RoutedExperts

    cache_config = _normalize_cache_config_for_hash(vllm_config.cache_config)
    if cache_config is not vllm_config.cache_config:
        vllm_config = replace(vllm_config, cache_config=cache_config)
    factors = vllm_config.compile_factors(include_version=False)
    mm_config = get_flashinfer_mm_encoder_autotune_config(vllm_config)
    if mm_config is not None:
        factors["flashinfer_mm_encoder_autotune"] = (
            _flashinfer_mm_encoder_autotune_hash(
                mm_config,
                enable_chunked_prefill=vllm_config.scheduler_config.enable_chunked_prefill,
            )
        )
    # DP runs with padding skip tune TRT-LLM MoE's precomputed-routing ops
    # instead of the logits ops; a cache from before lacks them, and a hit
    # skips all tuning.
    if (
        vllm_config.parallel_config.data_parallel_size > 1
        and envs.VLLM_MOE_SKIP_PADDING
        and not envs.VLLM_MOE_ROUTING_SIMULATION_STRATEGY
    ):
        assert get_model is not None, "DP MoE autotune key requires a model"
        if any(
            isinstance(m, RoutedExperts) and m.supports_precomputed_routing
            for m in get_model().modules()
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


def resolve_flashinfer_autotune_file(
    runner: "GPUModelRunner", *, per_rank: bool = False
) -> Path:
    path = _flashinfer_autotune_path(
        runner.vllm_config, flashinfer_autotune_cache_hash(runner), per_rank=per_rank
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def resolve_rank_flashinfer_autotune_file(runner: "GPUModelRunner") -> Path:
    """The per-rank variant of `resolve_flashinfer_autotune_file`."""
    return resolve_flashinfer_autotune_file(runner, per_rank=True)


def _flashinfer_autotune_path(
    vllm_config: VllmConfig, config_hash: str, *, per_rank: bool
) -> Path:
    filename = "autotune_configs.json"
    if per_rank:
        from vllm.distributed.parallel_state import get_world_group

        filename = flashinfer_autotune_filename(
            vllm_config.parallel_config.data_parallel_rank,
            get_world_group().rank_in_group,
        )
    return flashinfer_autotune_root() / config_hash / filename


def _autotune_cache_fingerprint(path: Path) -> tuple[str, int] | None:
    """Identify a saved autotune file by its FlashInfer metadata and size."""
    try:
        configs = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    if not isinstance(configs, dict):
        return None
    metadata = configs.pop("_metadata", None)
    return json.dumps(metadata, sort_keys=True), len(configs)


def _all_ranks_accept_autotune_cache(accepted: bool, group: "GroupCoordinator") -> bool:
    if group.world_size == 1:
        return accepted
    gathered = [False] * group.world_size
    torch.distributed.all_gather_object(gathered, accepted, group=group.cpu_group)
    return all(gathered)


def _rank_tables_match(cache_path: Path | None, group: "GroupCoordinator") -> bool:
    fingerprint = _autotune_cache_fingerprint(cache_path) if cache_path else None
    if group.world_size > 1:
        gathered: list[tuple[str, int] | None] = [None] * group.world_size
        torch.distributed.all_gather_object(
            gathered, fingerprint, group=group.cpu_group
        )
        if any(other != fingerprint for other in gathered):
            return False
    return fingerprint is not None


def _load_rank_autotune_file(cache_path: Path, tuner: "AutoTuner") -> bool:
    try:
        return tuner.load_configs(str(cache_path))
    except (
        AttributeError,
        IndexError,
        KeyError,
        OSError,
        TypeError,
        ValueError,
    ) as exc:
        # FlashInfer can reject a malformed file or race a writer after the
        # fingerprint check. Every rank must still enter the load vote.
        logger.warning(
            "Could not load FlashInfer autotune cache %s: %s", cache_path, exc
        )
        return False


def load_shared_autotune_cache(
    cache_path: Path,
    tuner: "AutoTuner",
    group: "GroupCoordinator",
    *,
    preloaded_path: Path | None = None,
) -> bool:
    """Reuse a unanimous preload, or load the mutually compatible late files."""
    if preloaded_path is not None:
        if _all_ranks_accept_autotune_cache(preloaded_path == cache_path, group):
            return True
        # FlashInfer's per-op keys do not include the vLLM config hash.
        tuner.clear_cache()
        raise RuntimeError(
            "FlashInfer autotune cache key changed on at least one rank "
            "after specialized decode warmup. "
            f"This rank preloaded {preloaded_path} and resolved {cache_path}."
        )
    if not _rank_tables_match(cache_path, group):
        return False
    loaded = _all_ranks_accept_autotune_cache(
        _load_rank_autotune_file(cache_path, tuner), group
    )
    if not loaded:
        tuner.clear_cache()
    return loaded


def _autotune_file_stamp(path: Path) -> tuple[int, ...] | None:
    try:
        stat = path.stat()
    except OSError:
        return None
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns


def preload_shared_autotune_cache(
    vllm_config: VllmConfig | None, tuner: "AutoTuner", group: "GroupCoordinator"
) -> Path | None:
    """Try the legacy table before TP-only warmup. A None config votes to skip."""
    from flashinfer.autotuner import AutoTuner

    path = None
    stamp = None
    # Protect live legacy winners and namespaces that load_configs can replace.
    if vllm_config is not None and not (
        tuner._dirty
        or tuner.profiling_cache
        or tuner._file_configs
        or tuner._namespaced_records
    ):
        path = _flashinfer_autotune_path(
            vllm_config, _flashinfer_autotune_cache_hash(vllm_config), per_rank=True
        )
        stamp = _autotune_file_stamp(path)
    if not _rank_tables_match(path if stamp is not None else None, group):
        return None
    assert path is not None

    # FlashInfer's loader can mutate file state even on rejection. Validate
    # against a private tuner before any rank touches its live tuner.
    try:
        validated = _load_rank_autotune_file(path, AutoTuner())
    except Exception:
        logger.exception("Could not validate FlashInfer autotune cache %s", path)
        validated = False
    if not _all_ranks_accept_autotune_cache(
        validated and _autotune_file_stamp(path) == stamp, group
    ):
        return None

    # Use the canonical path so FlashInfer remembers the observed generation
    # for later saves. Once any rank installs, a group fallback would need
    # a rollback API. Stop before tuning if files change during installation.
    try:
        installed = _load_rank_autotune_file(path, tuner)
    except Exception:
        logger.exception("Could not install FlashInfer autotune cache %s", path)
        installed = False
    if not _all_ranks_accept_autotune_cache(
        installed and _autotune_file_stamp(path) == stamp, group
    ):
        raise RuntimeError(
            "FlashInfer autotune cache changed or rejected on at least one rank "
            f"during early installation. This rank resolved {path}."
        )
    return path


def use_flashinfer_autotune_v2(runner: "GPUModelRunner") -> bool:
    from vllm.distributed.parallel_state import get_node_count, get_world_group
    from vllm.utils.flashinfer import has_flashinfer_autotune_v2

    # Elastic EP transfers tuning state to joining workers with v1
    # save_configs/load_configs, which do not export managed-store winners.
    if runner.vllm_config.parallel_config.enable_elastic_ep:
        return False
    # PP stages profile independently. Keep their group-scoped v1 files until
    # managed-cache reads and publishes can be isolated across tuning groups.
    if runner.vllm_config.parallel_config.pipeline_parallel_size > 1:
        return False
    if not has_flashinfer_autotune_v2():
        return False
    # Use the initialized topology: Ray may span nodes with config.nnodes=1.
    # A cache hit skips profiling collectives, so node-local stores with
    # different contents cannot safely participate in synchronized tuning.
    if get_node_count() != 1:
        logger.info_once(
            "Using legacy FlashInfer autotune cache synchronization for "
            "multi-node deployments; managed-cache synchronization is "
            "currently supported only within one node."
        )
        return False
    # Lift this gate after pinning a release with the fix below and validating
    # distributed managed-cache tuning: https://github.com/flashinfer-ai/flashinfer/pull/6185
    if get_world_group().world_size > 1:
        logger.info_once(
            "Using legacy FlashInfer autotune cache synchronization for "
            "multi-rank deployments until FlashInfer coordinates cache hits."
        )
        return False
    return True


def resolve_flashinfer_autotune_v2_root() -> Path | None:
    """Placement root for FlashInfer's managed autotune store.

    Unlike `resolve_flashinfer_autotune_file`, no vLLM-side identity
    (config hash, flashinfer version, arch) is folded into the path: the
    managed store keys entries per op and hashes the software/hardware
    environment itself. `None` selects FlashInfer's default root.
    """
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
