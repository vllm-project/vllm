# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import multiprocessing
import os
from collections.abc import Sequence
from concurrent.futures.process import ProcessPoolExecutor
from functools import cache
from typing import Any

import regex as re
import torch


def cuda_is_initialized() -> bool:
    """Check if CUDA is initialized."""
    if not torch.cuda._is_compiled():
        return False
    return torch.cuda.is_initialized()


def xpu_is_initialized() -> bool:
    """Check if XPU is initialized."""
    if not torch.xpu._is_compiled():
        return False
    return torch.xpu.is_initialized()


def cuda_get_device_properties(
    device, names: Sequence[str], init_cuda=False
) -> tuple[Any, ...]:
    """Get specified CUDA device property values without initializing CUDA in
    the current process."""
    if init_cuda or cuda_is_initialized():
        props = torch.cuda.get_device_properties(device)
        return tuple(getattr(props, name) for name in names)

    # Run in subprocess to avoid initializing CUDA as a side effect.
    mp_ctx = multiprocessing.get_context("fork")
    with ProcessPoolExecutor(max_workers=1, mp_context=mp_ctx) as executor:
        return executor.submit(cuda_get_device_properties, device, names, True).result()


@cache
def is_pin_memory_available() -> bool:
    from vllm.platforms import current_platform

    return current_platform.is_pin_memory_available()


@cache
def is_uva_available() -> bool:
    """Check if Unified Virtual Addressing (UVA) is available."""
    # UVA requires pinned memory.
    from vllm.platforms import current_platform

    # TODO: Add more requirements for UVA if needed.
    return is_pin_memory_available() or current_platform.is_cpu()


@cache
def num_compute_units(device_id: int = 0) -> int:
    """Get the number of compute units of the current device."""
    from vllm.platforms import current_platform

    return current_platform.num_compute_units(device_id)


@cache
def get_device_name_as_file_name(device_id: int = 0) -> str:
    from vllm.platforms import current_platform

    name = current_platform.get_device_name(device_id)
    name = re.sub(r"[\s/]+", "_", name)
    return name


def _normalize_config_device_name(name: str) -> str:
    name = re.sub(r"[^a-z0-9]", "", name.lower())
    return name.removeprefix("amd").removesuffix("graphics")


def resolve_rocm_device_config_file_path(config_file_path: str) -> str:
    """Resolve a tuned config written under an equivalent device name."""
    if os.path.exists(config_file_path):
        return config_file_path

    from vllm.platforms import current_platform

    if not current_platform.is_rocm():
        return config_file_path

    config_dir, config_file_name = os.path.split(config_file_path)
    match = re.search(r"device_name=([^,]+?)(?=,|\.json$)", config_file_name)
    if match is None:
        return config_file_path

    prefix = config_file_name[: match.start(1)]
    suffix = config_file_name[match.end(1) :]
    normalized_name = _normalize_config_device_name(match.group(1))
    if not normalized_name:
        return config_file_path
    matching_paths: list[str] = []

    try:
        entries = os.scandir(config_dir or ".")
    except OSError:
        return config_file_path

    with entries:
        for entry in entries:
            if not entry.is_file():
                continue
            if not entry.name.startswith(prefix) or not entry.name.endswith(suffix):
                continue
            end = len(entry.name) - len(suffix) if suffix else len(entry.name)
            candidate_name = entry.name[len(prefix) : end]
            if _normalize_config_device_name(candidate_name) == normalized_name:
                matching_paths.append(entry.path)

    return matching_paths[0] if len(matching_paths) == 1 else config_file_path
