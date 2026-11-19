# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Scratch shared by sequential DEP layers, never by concurrent executors."""

import math

import torch

from vllm.models.deepseek_v41.nvidia.specialized_decode.moe.routed_expert_gemm import (
    HIDDEN,
)


class ExpertWorkspace:
    def __init__(self, device, max_rows=128, top_k=6):
        self.device = torch.device(device)
        shapes = {
            "front_x": (max_rows * HIDDEN, torch.float8_e4m3fn),
            "front_scale": (max_rows * HIDDEN // 32, torch.uint8),
            "front_ids": (max_rows * top_k, torch.int32),
            "front_weights": (max_rows * top_k, torch.float32),
        }
        self.buffers = {
            key: torch.empty(size, dtype=dtype, device=device)
            for key, (size, dtype) in shapes.items()
        }

    def view(self, name, shape, dtype):
        buffer = self.buffers[name]
        size = math.prod(shape)
        if dtype != buffer.dtype or size > buffer.numel():
            raise ValueError("expert scratch capacity exceeded")
        return buffer[:size].view(shape)
