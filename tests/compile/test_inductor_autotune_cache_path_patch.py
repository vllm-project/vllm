# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the Inductor best_config path patch in env_override.py."""

import copy
import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import pytest
import torch

from vllm.env_override import _apply_inductor_autotune_cache_path_patch
from vllm.utils.torch_version import is_torch_equal_or_newer

pytestmark = pytest.mark.skipif(
    not is_torch_equal_or_newer("2.12.0.dev"),
    reason="backport targets torch >= 2.12",
)


@dataclass
class _Config:
    # The cache only needs these Triton config fields, even on CPU-only builds.
    kwargs: dict[str, int]
    num_warps: int = 4
    num_stages: int = 1


def _bundle_autotuner(cache_dir: Path, monkeypatch):
    """Deep-copy an autotuner as Inductor bundles it, then save its winner."""
    from torch._inductor.runtime.autotune_cache import AutotuneCache
    from torch._inductor.runtime.hints import DeviceProperties, HeuristicType
    from torch._inductor.runtime.triton_heuristics import (
        CachingAutotuner,
        StaticTritonCompileResult,
        hash_configs,
    )

    monkeypatch.setenv("TORCHINDUCTOR_CACHE_DIR", str(cache_dir))
    monkeypatch.setenv("TRITON_CACHE_DIR", str(cache_dir / "triton"))
    filename = cache_dir / "ab" / "cab123.py"
    filename.parent.mkdir(parents=True)
    filename.touch()
    configs = [_Config({"XBLOCK": size}) for size in (16, 32)]
    meta = {"autotune_remote_cache": False}
    cache = AutotuneCache.create(meta, str(filename), hash_configs(configs))
    assert cache is not None
    tuner = CachingAutotuner(
        fn=abs,
        triton_meta={"device": DeviceProperties("cpu", 0, 1, 0)},
        configs=configs,
        save_cache_hook=cache.save,
        mutated_arg_names=[],
        optimize_mem=False,
        heuristic_type=HeuristicType.POINTWISE,
        filename=str(filename),
        inductor_meta=meta,
    )
    tuner.compile_results = [
        StaticTritonCompileResult(None, config, {}, meta) for config in configs
    ]
    bundled = copy.deepcopy(tuner)
    cache.save(configs[1], 1_000_000)
    return bundled, configs, filename


@pytest.fixture
def stock_recheck(monkeypatch):
    from torch._inductor.runtime.triton_heuristics import CachingAutotuner

    # The vLLM import hook already wrapped the stock method.
    original = CachingAutotuner.recheck_autotune_cache.__wrapped__
    monkeypatch.setattr(CachingAutotuner, "recheck_autotune_cache", original)


@pytest.fixture
def patched_recheck(stock_recheck):
    _apply_inductor_autotune_cache_path_patch()


def test_stock_autotuner_misses_moved_cache_and_saves_to_old_path(
    tmp_path, monkeypatch, stock_recheck
):
    old, moved = tmp_path / "old", tmp_path / "moved"
    bundled, configs, _ = _bundle_autotuner(old, monkeypatch)
    old.rename(moved)
    monkeypatch.setenv("TORCHINDUCTOR_CACHE_DIR", str(moved))

    loaded = copy.deepcopy(bundled)
    loaded.recheck_autotune_cache(reload_kernel_from_src=None)
    assert [result.config for result in loaded.compile_results] == configs

    loaded.save_cache_hook(configs[0], 2_000_000)
    (winner,) = (moved / "ab").glob("*.best_config")
    assert json.loads(winner.read_bytes())["XBLOCK"] == 32
    assert json.loads((old / "ab" / winner.name).read_bytes())["XBLOCK"] == 16


def test_bundled_autotuner_reads_and_saves_in_moved_cache(
    tmp_path, monkeypatch, patched_recheck
):
    old, moved = tmp_path / "old", tmp_path / "moved"
    bundled, configs, filename = _bundle_autotuner(old, monkeypatch)
    old.rename(moved)
    monkeypatch.setenv("TORCHINDUCTOR_CACHE_DIR", str(moved))

    loaded = copy.deepcopy(bundled)
    loaded.recheck_autotune_cache(reload_kernel_from_src=None)
    assert [result.config for result in loaded.compile_results] == [configs[1]]
    assert Path(loaded.filename) == moved / "ab" / filename.name

    loaded.save_cache_hook(configs[0], 2_000_000)
    (winner,) = (moved / "ab").glob("*.best_config")
    assert json.loads(winner.read_bytes())["XBLOCK"] == 16
    assert not old.exists()


def test_bundled_autotuner_keeps_its_path_in_another_cache(
    tmp_path, monkeypatch, patched_recheck
):
    # A cache directory that does not hold this kernel must not take it over.
    own, other = tmp_path / "own", tmp_path / "other"
    bundled, configs, filename = _bundle_autotuner(own, monkeypatch)
    monkeypatch.setenv("TORCHINDUCTOR_CACHE_DIR", str(other))

    loaded = copy.deepcopy(bundled)
    loaded.recheck_autotune_cache(reload_kernel_from_src=None)
    assert [result.config for result in loaded.compile_results] == [configs[1]]
    assert Path(loaded.filename) == filename

    loaded.save_cache_hook(configs[0], 2_000_000)
    assert not list(other.rglob("*.best_config"))


@pytest.mark.skipif(
    not torch.cuda.is_available() or not torch.version.cuda, reason="requires CUDA"
)
def test_torch_compile_reuses_moved_best_config(tmp_path):
    old, moved = tmp_path / "old", tmp_path / "moved"

    def run(cache_dir, phase):
        env = dict(
            os.environ,
            TORCHINDUCTOR_CACHE_DIR=str(cache_dir),
            TRITON_CACHE_DIR=str(cache_dir / "triton"),
        )
        subprocess.run([sys.executable, __file__, phase], env=env, check=True)

    def best_configs(root):
        return {
            p.relative_to(root): p.read_bytes() for p in root.rglob("*.best_config")
        }

    run(old, "compile")
    before = best_configs(old)
    if not before:
        pytest.skip(
            "Inductor emitted no .best_config (a single config needs no tuning)"
        )
    shutil.copytree(old, moved)
    shutil.rmtree(old)
    run(moved, "load")
    assert not old.exists()
    assert best_configs(moved) == before


if __name__ == "__main__":
    # The module-level vLLM import installs the hook in each subprocess.
    from torch._dynamo.utils import counters

    def rms_norm(x):
        return x * torch.rsqrt(x.square().mean(-1, keepdim=True) + 1e-6)

    # Exercise the multi-config reduction that saves a .best_config on its first call.
    compiled = torch.compile(
        rms_norm,
        fullgraph=True,
        options={"triton.persistent_reductions": False, "max_autotune_pointwise": True},
    )
    x = torch.randn(256, 5120, device="cuda")
    torch.testing.assert_close(compiled(x), rms_norm(x))
    if sys.argv[1] == "load":
        assert counters["inductor"]["triton_bundler_load_static_autotuner"] > 0
