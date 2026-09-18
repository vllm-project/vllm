# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""FlashInfer autotune cache helpers."""

import hashlib
import os
import tempfile
from collections.abc import Callable
from contextlib import suppress
from pathlib import Path
from typing import TYPE_CHECKING, Any

import vllm.envs as envs

if TYPE_CHECKING:
    from vllm.distributed.parallel_state import GroupCoordinator
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner

# vLLM persists this name. FlashInfer 0.6.x dumps the singular sibling in
# the same directory; loading only one of them is the rank-0-only cache-hit
# deadlock in https://github.com/vllm-project/vllm/issues/57423.
_VLLM_AUTOTUNE_CACHE_FILENAME = "autotune_configs.json"
_FLASHINFER_AUTOTUNE_CACHE_FILENAME = "autotune_config.json"


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
    return output_dir / _VLLM_AUTOTUNE_CACHE_FILENAME


def iter_flashinfer_autotune_cache_files(cache_path: Path) -> tuple[Path, ...]:
    """Return vLLM and FlashInfer 0.6.x filenames for ``cache_path``.

    Args:
        cache_path: Canonical vLLM cache file (``autotune_configs.json``).

    Returns:
        Unique paths, vLLM name first, then FlashInfer's native name.

    """
    files: list[Path] = []
    for path in (
        cache_path,
        cache_path.with_name(_VLLM_AUTOTUNE_CACHE_FILENAME),
        cache_path.with_name(_FLASHINFER_AUTOTUNE_CACHE_FILENAME),
    ):
        if path not in files:
            files.append(path)
    return tuple(files)


def read_flashinfer_autotune_cache_bytes(cache_path: Path) -> bytes | None:
    """Read persisted configs, accepting either on-disk filename."""
    for path in iter_flashinfer_autotune_cache_files(cache_path):
        if path.is_file():
            return path.read_bytes()
    return None


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


def write_flashinfer_autotune_cache_aliases(cache_path: Path, contents: bytes) -> None:
    """Write the same payload to both vLLM and FlashInfer filenames."""
    for path in iter_flashinfer_autotune_cache_files(cache_path):
        write_flashinfer_autotune_cache(path, contents)


def reset_flashinfer_autotuner_state(tuner: Any) -> None:
    """Drop process-local hits so ranks cannot diverge before tuning."""
    clear_cache = getattr(tuner, "clear_cache", None)
    if callable(clear_cache):
        clear_cache()
        return
    for attr in ("profiling_cache", "_file_configs", "_ranked_tactics_cache"):
        cache = getattr(tuner, attr, None)
        if hasattr(cache, "clear"):
            cache.clear()


def share_flashinfer_autotune_cache(
    tuner: Any,
    cache_path: Path,
    *,
    is_leader: bool,
    broadcast_object: Callable[[Any], Any],
    barrier: Callable[[], None] | None = None,
) -> bool:
    """Load identical configs on every rank, or load none.

    FlashInfer's autotune process group deadlocks if one rank takes a
    config-file hit (skipping the per-tactic reduce) while others miss.
    Dual on-disk names and rank-local in-memory hits both cause that split.

    Args:
        tuner: FlashInfer ``AutoTuner`` instance.
        cache_path: Canonical vLLM cache file path.
        is_leader: Whether this rank should read from disk.
        broadcast_object: Rank-0 broadcast of the cache payload.
        barrier: Optional barrier after writing aliases, before load.

    Returns:
        True if configs were loaded on this rank.

    """
    payload = read_flashinfer_autotune_cache_bytes(cache_path) if is_leader else None
    payload = broadcast_object(payload)
    reset_flashinfer_autotuner_state(tuner)
    if payload is None:
        return False
    write_flashinfer_autotune_cache_aliases(cache_path, payload)
    if barrier is not None:
        barrier()
    loaded = tuner.load_configs(str(cache_path))
    return loaded is not False


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
                    path = Path(temp_dir) / _VLLM_AUTOTUNE_CACHE_FILENAME
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
