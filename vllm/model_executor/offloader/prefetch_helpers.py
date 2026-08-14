# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Plumbing helpers for prefetch weight offloading.

These small utilities are kept out of :mod:`prefetch` so the core control
flow stays close to the upstream prefetch implementation.
"""

from contextlib import nullcontext
from typing import Any

import torch
import torch.nn as nn

import vllm.envs as envs
from vllm.config.offload import PrefetchOffloadSelector
from vllm.logger import init_logger
from vllm.model_executor.offloader.numa import bind_process_to_gpu_numa

logger = init_logger(__name__)


def nvtx_range(name: str):
    if not envs.VLLM_NVTX_SCOPES_FOR_PROFILING:
        return nullcontext()
    return torch.cuda.nvtx.range(name)


def maybe_bind_process_to_current_gpu_numa() -> bool:
    """Bind the current worker process to NUMA-local CPUs of its GPU."""
    try:
        gpu_index = torch.accelerator.current_device_index()
        bound = bind_process_to_gpu_numa(gpu_index)
    except Exception as exc:
        logger.warning(
            "[PrefetchOffloader] Failed to bind process to GPU NUMA node: %s",
            exc,
        )
        return False
    if bound:
        logger.info(
            "[PrefetchOffloader] Bound process to NUMA-local CPUs for GPU %d",
            gpu_index,
        )
    else:
        logger.warning(
            "[PrefetchOffloader] Could not find NUMA-local CPUs for GPU %d",
            gpu_index,
        )
    return bound


def pick_dependency_tensor(
    positional_tensors: list[torch.Tensor],
    *,
    preferred_tensor: torch.Tensor | None = None,
) -> torch.Tensor:
    """Choose the tensor that best represents the layer's main activation.

    torch.compile ordering depends on the tensor we declare as mutated. Many
    decoder-layer forwards take metadata tensors such as ``positions`` before
    ``hidden_states`` in their positional argument list, so blindly picking
    ``args[0]`` can anchor the dependency to an integer tensor that is not on
    the critical path of the actual compute.
    """
    if preferred_tensor is not None:
        return preferred_tensor
    floating = [t for t in positional_tensors if t.is_floating_point()]
    if floating:
        return floating[0]
    if positional_tensors:
        return positional_tensors[0]
    raise ValueError("Could not find a tensor argument for prefetch dependency.")


def pick_output_dependency_tensor(output: Any) -> torch.Tensor:
    """Choose the tensor output that carries the next layer's activation."""
    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, tuple):
        tensors = [item for item in output if isinstance(item, torch.Tensor)]
        return pick_dependency_tensor(tensors)
    raise ValueError("Could not find a tensor output for prefetch dependency.")


def _retarget_single_submodule_unit(
    module: nn.Module,
    param_names: list[str],
) -> tuple[nn.Module, tuple[str, ...]]:
    """Return the common selected submodule and relative param names."""
    if not param_names:
        return module, ()

    split_names = [name.split(".")[:-1] for name in param_names]
    common_segments: list[str] = []
    for segments_at_depth in zip(*split_names):
        if len(set(segments_at_depth)) != 1:
            break
        common_segments.append(segments_at_depth[0])

    for depth in range(len(common_segments), 0, -1):
        prefix = ".".join(common_segments[:depth])
        try:
            target_module = module.get_submodule(prefix)
        except AttributeError:
            continue
        target_param_names = {name for name, _ in target_module.named_parameters()}
        relative_names: list[str] = []
        for name in param_names:
            if not name.startswith(f"{prefix}."):
                break
            relative = name[len(prefix) + 1 :]
            if relative not in target_param_names:
                break
            relative_names.append(relative)
        else:
            return target_module, tuple(relative_names)

    return module, tuple(param_names)


def maybe_retarget_offload_unit(
    module: nn.Module,
    param_names: list[str],
    *,
    selectors: set[PrefetchOffloadSelector],
    include_names: set[str],
) -> tuple[nn.Module, tuple[str, ...]]:
    """Pick the smallest submodule that owns the selected parameters.

    Single-selector layers (e.g. ``routed_experts``) are retargeted to the
    sub-module that exclusively owns the matched parameters so the prefetch
    hook can be installed at the tightest scope.  Mixed selectors retain the
    enclosing layer.
    """
    if len(selectors) == 1 and not include_names:
        return _retarget_single_submodule_unit(module, param_names)
    return module, tuple(param_names)
