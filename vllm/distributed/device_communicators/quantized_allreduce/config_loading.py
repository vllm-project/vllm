# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Config loader for quantized allreduce kernels."""

import json
import os

import torch

import vllm.envs as envs

_CONFIGS_DIR = os.path.join(os.path.dirname(__file__), "configs")
_cache: dict[tuple[str, int, int], dict[int, tuple[int, int, bool]]] = {}

_FALLBACK = (4096, 16, False)


def _get_device_name():
    """Get GPU device name, formatted like config filenames."""
    name = torch.cuda.get_device_name()
    return name.replace(" ", "_")


def _load_config(kernel, ws, group_size):
    """Load config for a specific kernel/ws/group_size combo."""
    key = (kernel, ws, group_size)
    if key in _cache:
        return _cache[key]

    device_name = _get_device_name()
    filename = (
        f"dtype={kernel},device_name={device_name},world_size={ws},gs{group_size}.json"
    )

    # Prioritize user-defined config folder (VLLM_TUNED_CONFIG_FOLDER)
    config_path = None
    user_folder = envs.VLLM_TUNED_CONFIG_FOLDER
    if user_folder is not None:
        candidate = os.path.join(user_folder, filename)
        if os.path.exists(candidate):
            config_path = candidate
    if config_path is None:
        candidate = os.path.join(_CONFIGS_DIR, filename)
        if os.path.exists(candidate):
            config_path = candidate
    if config_path is None:
        _cache[key] = {}
        return {}

    with open(config_path) as f:
        data = json.load(f)

    params = {}
    for numel_str, cfg in data.get("params", {}).items():
        params[int(numel_str)] = (
            cfg["BLOCK_SIZE"],
            cfg["num_warps"],
            cfg.get("use_p2p", False),
        )

    _cache[key] = params
    return params


def load_config(numel, ws, kernel="int8", group_size=256):
    """
    Look up optimal (BLOCK_SIZE, num_warps, use_p2p) from config files.

    Falls back to closest smaller size, or default (4096, 16).
    """
    params = _load_config(kernel, ws, group_size)
    if not params:
        return _FALLBACK

    candidates = [s for s in params if s <= numel]
    if candidates:
        return params[max(candidates)]

    return params[min(params)]
