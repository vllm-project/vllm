# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Unit tests for the JSON-based config loader added to selective_state_update.

Tests cover:
  - Config filename generation
  - VLLM_TUNED_CONFIG_FOLDER env-var override (per-GPU subfolder structure)
  - Fallback to heuristic when no config file exists
  - Nearest-batch interpolation
"""

import json

from vllm.model_executor.layers.mamba.ops.mamba_ssm import (
    _get_ssm_launch_config,
    get_ssm_config_file_name,
    get_ssm_configs,
    get_ssm_device_name,
)

# ---------------------------------------------------------------------------
# Config filename generation
# ---------------------------------------------------------------------------


def test_config_file_name_format():
    name = get_ssm_config_file_name(128)
    assert name == "dstate=128.json"


# ---------------------------------------------------------------------------
# VLLM_TUNED_CONFIG_FOLDER override (configs live in <folder>/<device>/dstate=N.json)
# ---------------------------------------------------------------------------


def test_env_override_loads_custom_config(monkeypatch, tmp_path):
    """VLLM_TUNED_CONFIG_FOLDER should take precedence over the bundled dir."""
    device_name = get_ssm_device_name()
    gpu_dir = tmp_path / device_name
    gpu_dir.mkdir()

    config_path = gpu_dir / get_ssm_config_file_name(16)
    payload = {"1": {"BLOCK_SIZE_M": 4, "num_warps": 1}}
    with open(config_path, "w") as f:
        json.dump(payload, f)

    monkeypatch.setenv("VLLM_TUNED_CONFIG_FOLDER", str(tmp_path))
    get_ssm_configs.cache_clear()

    cfg = get_ssm_configs(16)
    assert cfg is not None
    assert cfg[1] == {"BLOCK_SIZE_M": 4, "num_warps": 1}

    get_ssm_configs.cache_clear()


# ---------------------------------------------------------------------------
# Fallback to heuristic when no config file exists
# ---------------------------------------------------------------------------


def test_fallback_when_no_config(monkeypatch, tmp_path):
    """_get_ssm_launch_config must fall back to the hard-coded heuristic
    when no JSON file is found for the current device."""
    monkeypatch.setenv("VLLM_TUNED_CONFIG_FOLDER", str(tmp_path))
    monkeypatch.setattr(
        "vllm.model_executor.layers.mamba.ops.mamba_ssm._CONFIGS_DIR",
        str(tmp_path),
    )
    get_ssm_configs.cache_clear()

    # dstate=64 heuristic: BLOCK_SIZE_M=8, num_warps=4
    block_m, warps = _get_ssm_launch_config(dstate=64, batch=1, is_blackwell=False)
    assert block_m == 8
    assert warps == 4

    # dstate=16 heuristic: BLOCK_SIZE_M=32, num_warps=4
    block_m, warps = _get_ssm_launch_config(dstate=16, batch=1, is_blackwell=False)
    assert block_m == 32
    assert warps == 4

    get_ssm_configs.cache_clear()


# ---------------------------------------------------------------------------
# Nearest-batch interpolation
# ---------------------------------------------------------------------------


def test_nearest_batch_interpolation(monkeypatch, tmp_path):
    """When the exact batch size is not in the config, the closest key
    should be selected."""
    device_name = get_ssm_device_name()
    gpu_dir = tmp_path / device_name
    gpu_dir.mkdir()

    config_path = gpu_dir / get_ssm_config_file_name(32)
    payload = {
        "1": {"BLOCK_SIZE_M": 8, "num_warps": 1},
        "64": {"BLOCK_SIZE_M": 32, "num_warps": 4},
    }
    with open(config_path, "w") as f:
        json.dump(payload, f)

    monkeypatch.setenv("VLLM_TUNED_CONFIG_FOLDER", str(tmp_path))
    get_ssm_configs.cache_clear()

    # batch=5 is closer to 1 than to 64 — expects M=8, w=1
    block_m, warps = _get_ssm_launch_config(dstate=32, batch=5, is_blackwell=False)
    assert block_m == 8 and warps == 1

    # batch=40 is closer to 64 — expects M=32, w=4
    block_m, warps = _get_ssm_launch_config(dstate=32, batch=40, is_blackwell=False)
    assert block_m == 32 and warps == 4

    get_ssm_configs.cache_clear()


# ---------------------------------------------------------------------------
# Edge cases: malformed / empty config files
# ---------------------------------------------------------------------------


def test_non_dict_json_returns_none(monkeypatch, tmp_path):
    """A valid JSON file that is not a dict (e.g. a list) must be ignored
    and return None rather than raising AttributeError."""
    device_name = get_ssm_device_name()
    gpu_dir = tmp_path / device_name
    gpu_dir.mkdir()

    config_path = gpu_dir / get_ssm_config_file_name(16)
    with open(config_path, "w") as f:
        json.dump([1, 2, 3], f)

    monkeypatch.setenv("VLLM_TUNED_CONFIG_FOLDER", str(tmp_path))
    monkeypatch.setattr(
        "vllm.model_executor.layers.mamba.ops.mamba_ssm._CONFIGS_DIR",
        str(tmp_path),
    )
    get_ssm_configs.cache_clear()

    assert get_ssm_configs(16) is None

    get_ssm_configs.cache_clear()


def test_empty_config_falls_back_to_heuristic(monkeypatch, tmp_path):
    """An empty JSON object {} must not crash min() — should fall back
    to the hard-coded heuristic."""
    device_name = get_ssm_device_name()
    gpu_dir = tmp_path / device_name
    gpu_dir.mkdir()

    config_path = gpu_dir / get_ssm_config_file_name(64)
    with open(config_path, "w") as f:
        json.dump({}, f)

    monkeypatch.setenv("VLLM_TUNED_CONFIG_FOLDER", str(tmp_path))
    get_ssm_configs.cache_clear()

    # dstate=64 heuristic: BLOCK_SIZE_M=8, num_warps=4
    block_m, warps = _get_ssm_launch_config(dstate=64, batch=1, is_blackwell=False)
    assert block_m == 8
    assert warps == 4

    get_ssm_configs.cache_clear()
