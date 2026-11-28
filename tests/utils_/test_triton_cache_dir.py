# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""vLLM's own Triton kernels must survive a restart that hits the AOT cache."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

from vllm.platforms import current_platform


def _boot_engine() -> None:
    """Start an engine, which is what compiles vLLM's @triton.jit kernels."""
    from vllm import LLM

    llm = LLM(
        "Qwen/Qwen3-0.6B",
        max_model_len=2048,
        max_num_seqs=8,
        gpu_memory_utilization=0.3,
        kv_cache_memory_bytes=256 * 1024 * 1024,
    )
    llm.llm_engine.engine_core.shutdown(timeout=60)


def _start(tmp_path: Path, name: str, cache_root: Path) -> list[str]:
    """Boot an engine with a private Triton home and report what landed there."""
    triton_home = tmp_path / name
    triton_home.mkdir()
    env = os.environ.copy()
    env.pop("TRITON_CACHE_DIR", None)
    # Pin the model cache before HOME moves, so neither run re-downloads.
    env.setdefault("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
    # Both, so the run is isolated whether Triton reads TRITON_HOME or falls
    # back to the home directory.
    env.update(
        HOME=str(triton_home),
        TRITON_HOME=str(triton_home),
        VLLM_CACHE_ROOT=str(cache_root),
        VLLM_USE_AOT_COMPILE="1",
    )
    proc = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "from tests.utils_.test_triton_cache_dir import"
                " _boot_engine; _boot_engine()"
            ),
        ],
        cwd=Path(__file__).resolve().parents[2],
        env=env,
        capture_output=True,
        text=True,
        timeout=900,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    cache = triton_home / ".triton" / "cache"
    return sorted(p.name for p in cache.iterdir()) if cache.is_dir() else []


@pytest.mark.skipif(not current_platform.is_cuda_alike(), reason="needs a GPU")
def test_restart_reuses_triton_kernels(tmp_path):
    """The second start shares the vLLM cache, so it must compile nothing.

    Each start gets its own TRITON_HOME. Anything written there is a kernel
    that the shared VLLM_CACHE_ROOT failed to carry over.
    """
    cache_root = tmp_path / "vllm"
    _start(tmp_path, "home-1", cache_root)
    # Layout-agnostic: only that the first start cached something, so that a
    # silent failure to compile cannot make the next assertion vacuous.
    assert any(cache_root.rglob("*")), f"the first start cached nothing in {cache_root}"

    rebuilt = _start(tmp_path, "home-2", cache_root)
    assert not rebuilt, (
        f"restart compiled {len(rebuilt)} Triton kernels into a private "
        f"Triton home instead of reusing {cache_root}"
    )
