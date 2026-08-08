# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Unit tests for the JSON-based config loader added to selective_state_update.

Tests cover:
  - Flat MoE-style filename generation
  - VLLM_TUNED_CONFIG_FOLDER env-var override
  - Fallback to heuristic when no config file exists
  - Nearest effective_batch interpolation
  - Edge cases: non-dict JSON, empty config
"""

import json

import pytest
import torch

from vllm.model_executor.layers.mamba.ops.mamba_ssm import (
    _get_default_ssm_launch_config,
    _try_get_optimal_ssm_config_cached,
    get_ssm_autotune_config_file_path,
    get_ssm_config_file_name,
    get_ssm_configs,
    get_ssm_device_name,
    save_ssm_configs,
    try_get_optimal_ssm_config,
)
from vllm.model_executor.layers.mamba.ops.ssu_tuning import (
    NUM_WARPS_CHOICES,
    SSUTuningCase,
    block_size_m_choices,
    tune_ssu_case,
    valid_request_batches,
)
from vllm.platforms import current_platform

# Common kwargs for try_get_optimal_ssm_config. Tests pick (batch, nheads) so
# their product (effective_batch) matches the value being probed.
_HEADDIM = 64
_CACHE_DTYPE = "float32"


def _clear_caches() -> None:
    get_ssm_configs.cache_clear()
    _try_get_optimal_ssm_config_cached.cache_clear()


def _write_config(tmp_path, dstate: int, payload: dict) -> None:
    """Write payload as the bundled config for (headdim, dstate, cache_dtype)."""
    device_name = get_ssm_device_name()
    config_path = tmp_path / get_ssm_config_file_name(
        _HEADDIM, dstate, _CACHE_DTYPE, device_name
    )
    with open(config_path, "w") as f:
        json.dump(payload, f)


# ---------------------------------------------------------------------------
# Config filename generation
# ---------------------------------------------------------------------------


def test_config_file_name_format():
    name = get_ssm_config_file_name(
        headdim=64, dstate=128, cache_dtype="float32", device_name="NVIDIA_B200"
    )
    assert name == (
        "headdim=64,dstate=128,device_name=NVIDIA_B200,cache_dtype=float32.json"
    )


# ---------------------------------------------------------------------------
# VLLM_TUNED_CONFIG_FOLDER override
# ---------------------------------------------------------------------------


def test_env_override_loads_custom_config(monkeypatch, tmp_path):
    """VLLM_TUNED_CONFIG_FOLDER should take precedence over the bundled dir."""
    _write_config(
        tmp_path,
        dstate=16,
        payload={
            "1": {"BLOCK_SIZE_M": 4, "num_warps": 1},
        },
    )

    monkeypatch.setenv("VLLM_TUNED_CONFIG_FOLDER", str(tmp_path))
    _clear_caches()

    cfg = get_ssm_configs(_HEADDIM, 16, _CACHE_DTYPE)
    assert cfg is not None
    assert cfg[1] == {"BLOCK_SIZE_M": 4, "num_warps": 1}

    _clear_caches()


def test_autotune_cache_path_uses_existing_filename_schema(monkeypatch, tmp_path):
    monkeypatch.setenv("VLLM_MAMBA_SSU_AUTOTUNE_CACHE_DIR", str(tmp_path))
    path = get_ssm_autotune_config_file_path(
        headdim=_HEADDIM,
        dstate=128,
        cache_dtype="bfloat16",
        device_name="GPU",
    )

    assert str(tmp_path) in path
    assert path.endswith(get_ssm_config_file_name(_HEADDIM, 128, "float16", "GPU"))


def test_cache_precedence_user_then_local_then_bundled(monkeypatch, tmp_path):
    user_dir = tmp_path / "user"
    cache_dir = tmp_path / "cache"
    bundled_dir = tmp_path / "bundled"
    for directory in (user_dir, cache_dir, bundled_dir):
        directory.mkdir()

    _write_config(user_dir, 16, {"1": {"BLOCK_SIZE_M": 4, "num_warps": 1}})
    _write_config(cache_dir, 16, {"1": {"BLOCK_SIZE_M": 8, "num_warps": 2}})
    _write_config(bundled_dir, 16, {"1": {"BLOCK_SIZE_M": 16, "num_warps": 4}})

    monkeypatch.setenv("VLLM_TUNED_CONFIG_FOLDER", str(user_dir))
    monkeypatch.setenv("VLLM_MAMBA_SSU_AUTOTUNE_CACHE_DIR", str(cache_dir))
    monkeypatch.setattr(
        "vllm.model_executor.layers.mamba.ops.mamba_ssm._CONFIGS_DIR",
        str(bundled_dir),
    )
    _clear_caches()
    assert get_ssm_configs(_HEADDIM, 16, _CACHE_DTYPE)[1] == {
        "BLOCK_SIZE_M": 4,
        "num_warps": 1,
    }

    monkeypatch.delenv("VLLM_TUNED_CONFIG_FOLDER")
    _clear_caches()
    assert get_ssm_configs(_HEADDIM, 16, _CACHE_DTYPE)[1] == {
        "BLOCK_SIZE_M": 8,
        "num_warps": 2,
    }

    monkeypatch.setenv("VLLM_MAMBA_SSU_AUTOTUNE_CACHE_DIR", str(tmp_path / "empty"))
    _clear_caches()
    assert get_ssm_configs(_HEADDIM, 16, _CACHE_DTYPE)[1] == {
        "BLOCK_SIZE_M": 16,
        "num_warps": 4,
    }

    _clear_caches()


def test_save_ssm_configs_loads_from_local_cache(monkeypatch, tmp_path):
    monkeypatch.setenv("VLLM_MAMBA_SSU_AUTOTUNE_CACHE_DIR", str(tmp_path))
    monkeypatch.setattr(
        "vllm.model_executor.layers.mamba.ops.mamba_ssm._CONFIGS_DIR",
        str(tmp_path / "empty"),
    )
    save_ssm_configs(
        _HEADDIM,
        32,
        _CACHE_DTYPE,
        {128: {"BLOCK_SIZE_M": 32, "num_warps": 8}},
    )
    _clear_caches()

    block_m, warps = try_get_optimal_ssm_config(
        headdim=_HEADDIM,
        dstate=32,
        batch=1,
        nheads=128,
        cache_dtype=_CACHE_DTYPE,
        is_blackwell=False,
    )
    assert (block_m, warps) == (32, 8)

    _clear_caches()


def test_warmup_request_batches_match_benchmark_grid():
    assert valid_request_batches(2) == [1, 2]
    assert valid_request_batches(20) == [1, 8, 16, 20]
    assert valid_request_batches(2048) == [
        1,
        8,
        16,
        32,
        64,
        128,
        256,
        512,
        1024,
        1536,
        2048,
    ]
    assert valid_request_batches(4096) == [
        1,
        8,
        16,
        32,
        64,
        128,
        256,
        512,
        1024,
        1536,
        2048,
        4096,
    ]


# ---------------------------------------------------------------------------
# Fallback to heuristic when no config file exists
# ---------------------------------------------------------------------------


def test_fallback_when_no_config(monkeypatch, tmp_path):
    """try_get_optimal_ssm_config must fall back to _get_default_ssm_launch_config
    when no JSON file is found for the current
    (device, headdim, dstate, cache_dtype) combination.
    """
    monkeypatch.setenv("VLLM_TUNED_CONFIG_FOLDER", str(tmp_path))
    monkeypatch.setenv("VLLM_MAMBA_SSU_AUTOTUNE_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setattr(
        "vllm.model_executor.layers.mamba.ops.mamba_ssm._CONFIGS_DIR",
        str(tmp_path),
    )

    for dstate in (8, 16, 32, 64, 128, 256):
        for is_blackwell in (False, True):
            _clear_caches()
            block_m, warps = try_get_optimal_ssm_config(
                headdim=_HEADDIM,
                dstate=dstate,
                batch=1,
                nheads=1,
                cache_dtype=_CACHE_DTYPE,
                is_blackwell=is_blackwell,
            )
            assert (block_m, warps) == _get_default_ssm_launch_config(
                dstate, is_blackwell=is_blackwell
            )

    _clear_caches()


# ---------------------------------------------------------------------------
# Nearest effective_batch interpolation
# ---------------------------------------------------------------------------


def test_nearest_effective_batch_interpolation(monkeypatch, tmp_path):
    """When effective_batch = batch*nheads is not an exact key, the closest
    key should be selected."""
    _write_config(
        tmp_path,
        dstate=32,
        payload={
            "64": {"BLOCK_SIZE_M": 8, "num_warps": 1},
            "4096": {"BLOCK_SIZE_M": 32, "num_warps": 4},
        },
    )

    monkeypatch.setenv("VLLM_TUNED_CONFIG_FOLDER", str(tmp_path))
    _clear_caches()

    # effective_batch = 1*128 = 128 -> closer to 64 than to 4096
    block_m, warps = try_get_optimal_ssm_config(
        headdim=_HEADDIM,
        dstate=32,
        batch=1,
        nheads=128,
        cache_dtype=_CACHE_DTYPE,
        is_blackwell=False,
    )
    assert block_m == 8 and warps == 1

    # effective_batch = 4*1024 = 4096 -> exact match on 4096
    block_m, warps = try_get_optimal_ssm_config(
        headdim=_HEADDIM,
        dstate=32,
        batch=4,
        nheads=1024,
        cache_dtype=_CACHE_DTYPE,
        is_blackwell=False,
    )
    assert block_m == 32 and warps == 4

    _clear_caches()


# ---------------------------------------------------------------------------
# Edge cases: malformed / empty config files
# ---------------------------------------------------------------------------


def test_non_dict_json_returns_none(monkeypatch, tmp_path):
    """A valid JSON file that is not a dict (e.g. a list) must be ignored
    and return None rather than raising AttributeError."""
    device_name = get_ssm_device_name()
    config_path = tmp_path / get_ssm_config_file_name(
        _HEADDIM, 16, _CACHE_DTYPE, device_name
    )
    with open(config_path, "w") as f:
        json.dump([1, 2, 3], f)

    monkeypatch.setenv("VLLM_TUNED_CONFIG_FOLDER", str(tmp_path))
    monkeypatch.setenv("VLLM_MAMBA_SSU_AUTOTUNE_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setattr(
        "vllm.model_executor.layers.mamba.ops.mamba_ssm._CONFIGS_DIR",
        str(tmp_path),
    )
    _clear_caches()

    assert get_ssm_configs(_HEADDIM, 16, _CACHE_DTYPE) is None

    _clear_caches()


def test_empty_config_falls_back_to_heuristic(monkeypatch, tmp_path):
    """An empty JSON object {} must not crash min() — should fall back
    to the hard-coded heuristic."""
    _write_config(tmp_path, dstate=64, payload={})

    monkeypatch.setenv("VLLM_TUNED_CONFIG_FOLDER", str(tmp_path))
    monkeypatch.setenv("VLLM_MAMBA_SSU_AUTOTUNE_CACHE_DIR", str(tmp_path / "cache"))
    _clear_caches()

    dstate = 64
    block_m, warps = try_get_optimal_ssm_config(
        headdim=_HEADDIM,
        dstate=dstate,
        batch=1,
        nheads=64,
        cache_dtype=_CACHE_DTYPE,
        is_blackwell=False,
    )
    assert (block_m, warps) == _get_default_ssm_launch_config(
        dstate=dstate, is_blackwell=False
    )

    _clear_caches()


@pytest.mark.skipif(
    not current_platform.is_cuda_alike(),
    reason="SSU autotune smoke test requires CUDA-like Triton platform.",
)
def test_tune_ssu_case_tiny_gpu():
    case = SSUTuningCase(
        batch=1,
        nheads=1,
        headdim=16,
        dstate=16,
        ngroups=1,
        dtype=torch.float16,
        state_dtype=torch.float32,
        device=torch.device("cuda"),
    )

    cfg = tune_ssu_case(case, num_iters=1, num_warmup=1)

    assert cfg is not None
    assert cfg["BLOCK_SIZE_M"] in block_size_m_choices(case.headdim)
    assert cfg["num_warps"] in NUM_WARPS_CHOICES
