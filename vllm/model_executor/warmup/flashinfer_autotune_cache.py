# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""FlashInfer autotune cache helpers."""

import hashlib
import os
import tempfile
from contextlib import suppress
from pathlib import Path
from typing import TYPE_CHECKING

import torch

import vllm.envs as envs

if TYPE_CHECKING:
    from vllm.distributed.parallel_state import GroupCoordinator
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner


def flashinfer_autotune_cache_hash(runner: "GPUModelRunner") -> str:
    config_hash = runner.vllm_config.compute_hash(include_version=False)
    return hashlib.sha256(config_hash.encode()).hexdigest()


def resolve_flashinfer_autotune_file(runner: "GPUModelRunner") -> Path:
    override_dir = envs.VLLM_FLASHINFER_AUTOTUNE_CACHE_DIR
    if override_dir:
        root = Path(override_dir).expanduser()
    else:
        from flashinfer.jit import env as flashinfer_jit_env

        flashinfer_workspace = flashinfer_jit_env.FLASHINFER_WORKSPACE_DIR
        root = (
            Path(envs.VLLM_CACHE_ROOT)
            / "flashinfer_autotune_cache"
            / flashinfer_workspace.parent.name
            / flashinfer_workspace.name
        )

    output_dir = root / flashinfer_autotune_cache_hash(runner)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir / "autotune_configs.json"


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


def load_flashinfer_autotune_cache_only(
    cache_path: Path,
    tune_group: "GroupCoordinator",
    world: "GroupCoordinator",
) -> None:
    """Load the tuning group leader's autotune cache on every rank, or fail.

    Every world rank makes the same decision: read, transfer and load errors
    are gathered over ``world`` and re-raised on all ranks, so no rank is left
    waiting in a later collective. A successful load only means FlashInfer
    accepted the file; ops without an entry use FlashInfer's default tactic.

    Raises:
        RuntimeError: If the cache is missing, empty, unreadable or rejected
            by FlashInfer on any rank.

    """
    is_leader = tune_group.rank_in_group == 0
    cache: bytes | str | None = None
    if is_leader:
        try:
            cache = cache_path.read_bytes() or f"{cache_path} is empty"
        except Exception as exc:
            cache = f"{type(exc).__name__}: {exc}"
    cache = tune_group.broadcast_object(cache, src=0)

    error: str | None = None
    if isinstance(cache, str):
        if is_leader:
            error = cache
    else:
        try:
            from flashinfer.autotuner import AutoTuner

            with tempfile.NamedTemporaryFile(suffix=".json") as f:
                f.write(cache)
                f.flush()
                if not AutoTuner.get().load_configs(f.name):
                    error = "FlashInfer rejected the cache (environment mismatch)"
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"

    errors: list[str | None] = [error]
    if world.world_size > 1:
        errors = [None] * world.world_size
        torch.distributed.all_gather_object(errors, error, group=world.cpu_group)
    failures = "; ".join(
        f"rank {rank}: {error}" for rank, error in enumerate(errors) if error
    )
    if failures:
        raise RuntimeError(
            "VLLM_FLASHINFER_AUTOTUNE_CACHE_ONLY is set but the FlashInfer "
            f"autotune cache could not be loaded: {failures}. Unset it to "
            "autotune, or prepare a cache for this configuration."
        )


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
