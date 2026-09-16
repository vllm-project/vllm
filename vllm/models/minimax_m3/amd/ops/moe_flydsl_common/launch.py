# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MiniMax-M3 FlyDSL MoE launch helpers."""

import functools

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch

from vllm.models.minimax_m3.amd.ops.moe_flydsl_common.reduce_bf16 import (
    compile_moe_reduce_bf16,
)
from vllm.models.minimax_m3.amd.ops.moe_flydsl_common.sort import compile_moe_sort
from vllm.models.minimax_m3.amd.ops.moe_flydsl_common.tile_map import compile_tile_map


def _u8_flat(t: torch.Tensor) -> torch.Tensor:
    """The flat byte view the kernels take their tensors as."""
    return t.view(torch.uint8).view(-1)


class _DeviceStream(fx.Stream):
    def __cache_signature__(self):
        # FlyDSL module handles belong to the device that loaded them.
        return (fx.Stream, self.value.device.index)


def _run_compiled(exe, *args):
    """Compile and run once per device, then reuse its CompiledFunction."""
    device = args[-1].device.index
    cache = getattr(exe, "_cf", None)
    if cache is None:
        exe._cf = cache = {}
    cf = cache.get(device)
    if cf is None:
        cache[device] = flyc.compile(exe, *args[:-1], _DeviceStream(args[-1]))
    else:
        cf(*args)


_launches: dict[str, object] = {}


def _get(compile_fn, **kw):
    """One launch object per distinct kernel. The kernels depend on ``n_tokens``
    only through a few buckets, so launches built for different ``n_tokens`` are
    shared by kernel name (the compiled code hangs off the launch object)."""
    launch = compile_fn(**kw)
    return _launches.setdefault(launch.kernel_name, launch)


@functools.cache
def _get_sort(num_experts: int, topk: int, block_m: int):
    return compile_moe_sort(E=num_experts, topk=topk, block_m=block_m)


@functools.cache
def _get_tile_map(intermediate_size: int, block_m: int):
    return compile_tile_map(I=intermediate_size, BM=block_m)


@functools.cache
def _get_reduce_bf16(hidden_size: int, topk: int):
    return compile_moe_reduce_bf16(H=hidden_size, topk=topk)
