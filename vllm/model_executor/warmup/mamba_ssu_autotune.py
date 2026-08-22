# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Warmup-time autotuning for vLLM's Triton Mamba SSU backend."""

from __future__ import annotations

import json
import os
import tempfile
from contextlib import suppress
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

import vllm.envs as envs
from vllm.config.mamba import MambaBackendEnum
from vllm.logger import init_logger
from vllm.model_executor.layers.mamba.ops.mamba_ssm import (
    _try_get_optimal_ssm_config_cached,
    get_ssm_autotune_config_file_path,
    get_ssm_configs,
    save_ssm_configs,
)
from vllm.model_executor.layers.mamba.ops.ssu_tuning import (
    SSUTuningCase,
    make_active_cases,
    tune_ssu_case,
    valid_request_batches,
)
from vllm.platforms import current_platform

if TYPE_CHECKING:
    from vllm.v1.worker.gpu_worker import Worker

logger = init_logger(__name__)

_TUNE_NUM_ITERS = 10
_TUNE_NUM_WARMUP = 3


@dataclass(frozen=True)
class SSUWarmupShape:
    headdim: int
    dstate: int
    ngroups: int
    nheads: int
    dtype: torch.dtype
    state_dtype: torch.dtype
    device: torch.device
    is_blackwell: bool

    @property
    def cache_dtype_name(self) -> str:
        return str(self.state_dtype).removeprefix("torch.")


def _atomic_write_bytes(path: str, contents: bytes) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(
        dir=os.path.dirname(path),
        suffix=".tmp",
        prefix=f".{os.path.basename(path)}.",
    )
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(contents)
        os.replace(tmp_path, path)
    except BaseException:
        with suppress(OSError):
            os.unlink(tmp_path)
        raise


def _load_local_configs(path: str) -> dict[int, dict[str, int]]:
    if not os.path.exists(path):
        return {}
    try:
        with open(path) as f:
            raw = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        logger.warning("Ignoring invalid Mamba SSU autotune cache %s: %s", path, e)
        return {}
    if not isinstance(raw, dict):
        return {}
    raw.pop("triton_version", None)
    return {int(k): v for k, v in raw.items() if k.isdigit() and isinstance(v, dict)}


def _clear_ssu_config_caches() -> None:
    get_ssm_configs.cache_clear()
    _try_get_optimal_ssm_config_cached.cache_clear()


def _first_param_device(module: torch.nn.Module, default: torch.device) -> torch.device:
    for param in module.parameters(recurse=False):
        return param.device
    return default


def _state_dtype(module: torch.nn.Module) -> torch.dtype | None:
    get_state_dtype = getattr(module, "get_state_dtype", None)
    if get_state_dtype is None:
        return None
    try:
        state_dtypes = get_state_dtype()
    except Exception:
        return None
    if not state_dtypes:
        return None
    return state_dtypes[-1]


def _activation_dtype(module: torch.nn.Module) -> torch.dtype | None:
    model_config = getattr(module, "model_config", None)
    dtype = getattr(model_config, "dtype", None)
    if isinstance(dtype, torch.dtype):
        return dtype
    for param in module.parameters(recurse=False):
        if param.dtype != torch.float32:
            return param.dtype
    return None


def _shape_from_module(
    module: torch.nn.Module,
    default_device: torch.device,
) -> SSUWarmupShape | None:
    dstate = getattr(module, "ssm_state_size", None)
    if not isinstance(dstate, int) or dstate <= 0:
        return None

    state_dtype = _state_dtype(module)
    dtype = _activation_dtype(module)
    if state_dtype is None or dtype is None:
        return None

    tp_size = int(getattr(module, "tp_size", 1) or 1)
    nheads_attr = getattr(module, "num_heads", None)
    head_dim_attr = getattr(module, "head_dim", None)

    if isinstance(nheads_attr, int) and isinstance(head_dim_attr, int):
        nheads = max(1, nheads_attr // tp_size)
        headdim = head_dim_attr
        ngroups_attr = getattr(module, "n_groups", None)
        ngroups = (
            max(1, ngroups_attr // tp_size)
            if isinstance(ngroups_attr, int) and ngroups_attr > 0
            else 1
        )
    else:
        # Mamba-1 stores state as (batch, dim, dstate), which the SSU wrapper
        # treats as one synthetic head with dim == per-rank intermediate size.
        intermediate_size = getattr(module, "intermediate_size", None)
        if not isinstance(intermediate_size, int) or intermediate_size <= 0:
            return None
        nheads = 1
        headdim = max(1, intermediate_size // tp_size)
        ngroups = 1

    device = _first_param_device(module, default_device)
    if device.type != "cuda":
        device = default_device

    return SSUWarmupShape(
        headdim=headdim,
        dstate=dstate,
        ngroups=ngroups,
        nheads=nheads,
        dtype=dtype,
        state_dtype=state_dtype,
        device=device,
        is_blackwell=current_platform.is_device_capability_family(100),
    )


def _discover_ssu_shapes(worker: Worker) -> list[SSUWarmupShape]:
    runner = worker.model_runner
    default_device = getattr(runner, "device", torch.device("cuda"))
    shapes: set[SSUWarmupShape] = set()
    for module in worker.get_model().modules():
        shape = _shape_from_module(module, default_device)
        if shape is not None:
            shapes.add(shape)
    return sorted(
        shapes,
        key=lambda s: (
            s.headdim,
            s.dstate,
            s.ngroups,
            s.nheads,
            str(s.dtype),
            str(s.state_dtype),
        ),
    )


def _max_num_reqs(worker: Worker) -> int:
    runner = worker.model_runner
    for attr_owner in (runner, getattr(runner, "scheduler_config", None)):
        if attr_owner is None:
            continue
        for name in ("max_num_reqs", "max_num_seqs", "max_num_batched_tokens"):
            value = getattr(attr_owner, name, None)
            if isinstance(value, int) and value > 0:
                return value
    return 1


def _tune_shape(
    shape: SSUWarmupShape,
    request_batches: list[int],
    existing_configs: dict[int, dict[str, int]],
) -> dict[int, dict[str, int]]:
    configs = dict(existing_configs)
    active = make_active_cases(request_batches, shape.nheads, shape.ngroups)
    for effective_batch, batch, nheads in active:
        if effective_batch in configs and not envs.VLLM_MAMBA_SSU_AUTOTUNE_FORCE:
            continue
        case = SSUTuningCase(
            batch=batch,
            nheads=nheads,
            headdim=shape.headdim,
            dstate=shape.dstate,
            ngroups=shape.ngroups,
            dtype=shape.dtype,
            state_dtype=shape.state_dtype,
            device=shape.device,
            is_blackwell=shape.is_blackwell,
        )
        best = tune_ssu_case(
            case,
            num_iters=_TUNE_NUM_ITERS,
            num_warmup=_TUNE_NUM_WARMUP,
        )
        if best is not None:
            configs[effective_batch] = best
    return configs


def mamba_ssu_autotune_warmup(worker: Worker) -> None:
    """Tune Triton SSU configs on scratch tensors and persist local cache files."""
    if not worker.vllm_config.kernel_config.enable_mamba_ssu_autotune:
        return
    if worker.vllm_config.mamba_config.backend != MambaBackendEnum.TRITON:
        logger.info("Skipping Mamba SSU autotune because backend is not Triton.")
        return
    if not current_platform.is_cuda_alike():
        logger.info("Skipping Mamba SSU autotune on non-CUDA-like platform.")
        return

    shapes = _discover_ssu_shapes(worker)
    if not shapes:
        return

    from vllm.distributed.parallel_state import get_world_group

    world = get_world_group()
    is_leader = world.rank_in_group == 0
    request_batches = valid_request_batches(_max_num_reqs(worker))
    if not request_batches:
        return

    for shape in shapes:
        cache_path = get_ssm_autotune_config_file_path(
            shape.headdim,
            shape.dstate,
            shape.cache_dtype_name,
        )

        payload: bytes | None = None
        if is_leader:
            existing = _load_local_configs(cache_path)
            if envs.VLLM_MAMBA_SSU_AUTOTUNE_FORCE or not existing:
                logger.info(
                    "Autotuning Mamba SSU headdim=%d dstate=%d nheads=%d "
                    "ngroups=%d cache_dtype=%s request_batches=%s into %s",
                    shape.headdim,
                    shape.dstate,
                    shape.nheads,
                    shape.ngroups,
                    shape.cache_dtype_name,
                    request_batches,
                    cache_path,
                )
            tuned = _tune_shape(shape, request_batches, existing)
            if tuned:
                save_ssm_configs(
                    shape.headdim,
                    shape.dstate,
                    shape.cache_dtype_name,
                    tuned,
                )
                with open(cache_path, "rb") as f:
                    payload = f.read()

        payload = world.broadcast_object(payload, src=0)
        if payload is not None:
            _atomic_write_bytes(cache_path, payload)
            _clear_ssu_config_caches()

    world.barrier()
