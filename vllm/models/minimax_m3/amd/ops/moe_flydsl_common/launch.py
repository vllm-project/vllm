# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MiniMax-M3 FlyDSL MoE launch helpers."""

import functools

import torch

from vllm.models.minimax_m3.amd.ops.moe_flydsl_common.reduce_bf16 import (
    compile_moe_reduce_bf16,
)
from vllm.models.minimax_m3.amd.ops.moe_flydsl_common.sort import compile_moe_sort
from vllm.models.minimax_m3.amd.ops.moe_flydsl_common.tile_map import compile_tile_map


def _u8_flat(t: torch.Tensor) -> torch.Tensor:
    return t.view(torch.uint8).view(-1)


@functools.cache
def _get_sort(num_experts: int, topk: int, block_m: int):
    return compile_moe_sort(E=num_experts, topk=topk, block_m=block_m)


@functools.cache
def _get_tile_map(intermediate_size: int, block_m: int):
    return compile_tile_map(I=intermediate_size, BM=block_m)


@functools.cache
def _get_reduce_bf16(hidden_size: int, topk: int):
    return compile_moe_reduce_bf16(H=hidden_size, topk=topk)
