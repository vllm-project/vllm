# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Config loader for quantized allreduce kernels."""

import glob
import json
import os

import torch

_CONFIGS_DIR = os.path.join(os.path.dirname(__file__), "configs")
_cache: dict[tuple[str, int, int], dict[int, tuple[int, int]]] = {}

_FALLBACK = (4096, 16)


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
    pattern = os.path.join(
        _CONFIGS_DIR,
        f"dtype={kernel},device_name={device_name},world_size={ws},gs{group_size}.json",
    )
    files = glob.glob(pattern)
    if not files:
        _cache[key] = {}
        return {}

    with open(files[0]) as f:
        data = json.load(f)

    params = {}
    for numel_str, cfg in data.get("params", {}).items():
        params[int(numel_str)] = (cfg["BLOCK_SIZE"], cfg["num_warps"])

    _cache[key] = params
    return params


def load_config(numel, ws, kernel="int8", group_size=256):
    """
    Look up optimal (BLOCK_SIZE, num_warps) from config files.

    Falls back to closest smaller size, or default (4096, 16).
    """
    params = _load_config(kernel, ws, group_size)
    if not params:
        return _FALLBACK

    candidates = [s for s in params if s <= numel]
    if candidates:
        return params[max(candidates)]

    return params[min(params)]
