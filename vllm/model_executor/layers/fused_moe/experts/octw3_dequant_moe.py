# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Octave-W3 experts on any GPU: dequantize the layer to fp16, then Triton.

Reference path for devices without the RDNA3 kernel. Each call expands the
planar checkpoint weights of one layer into a dense buffer shared by all
layers, ``W = blocks(s * LV[c]) @ Hb`` with the same normalized Sylvester
Hadamard the RDNA3 kernel applies to its input, and runs ``TritonExperts``.
"""

import math
import os

import torch

from vllm.model_executor.layers.fused_moe.config import (
    FUSED_MOE_UNQUANTIZED_CONFIG,
    FusedMoEConfig,
)
from vllm.model_executor.layers.fused_moe.experts.rdna3_octw3_moe import (
    OCTW3_LEVELS,
)
from vllm.model_executor.layers.fused_moe.experts.triton_moe import TritonExperts

_DENSE: dict[tuple, torch.Tensor] = {}
# Layers expanded once at load time and kept dense, up to this many GiB.
_DENSE_BUDGET = [float(os.environ.get("VLLM_OCTW3_DENSE_GB", "0")) * 2**30]


def take_dense_budget(nbytes: int) -> bool:
    if _DENSE_BUDGET[0] < nbytes:
        return False
    _DENSE_BUDGET[0] -= nbytes
    return True


def _hadamard(n: int, device: torch.device) -> torch.Tensor:
    h = torch.ones(1, 1, device=device, dtype=torch.float32)
    while h.shape[0] < n:
        h = torch.cat([torch.cat([h, h], 1), torch.cat([h, -h], 1)], 0)
    return h / math.sqrt(n)


def octw3_dequant(
    packed: torch.Tensor,
    scales: torch.Tensor,
    had: int,
    out: torch.Tensor | None = None,
    chunk: int = 16,
) -> torch.Tensor:
    """Planar ``[E, N, K/32*3]`` int32 codes and ``[E, N, K/32]`` scales to
    fp16 ``[E, N, K]``."""
    E, N, G3 = packed.shape
    K = G3 // 3 * 32
    dev = packed.device
    if out is None:
        out = torch.empty(E, N, K, dtype=torch.float16, device=dev)
    if packed.is_cuda:
        from vllm.model_executor.layers.fused_moe.experts.octw3_dequant_hip import (
            octw3_dequant_hip,
        )

        return octw3_dequant_hip(packed, scales, had, out)
    levels = torch.tensor(OCTW3_LEVELS, device=dev, dtype=torch.float32)
    bits = torch.arange(32, device=dev, dtype=torch.int32)
    hb = _hadamard(had, dev) if had else None
    for e0 in range(0, E, chunk):
        planes = packed[e0 : e0 + chunk].unflatten(-1, (-1, 3))
        codes = torch.zeros(*planes.shape[:-1], 32, dtype=torch.int32, device=dev)
        for b in range(3):
            codes |= ((planes[..., b, None] >> bits) & 1) << b
        w = (levels[codes] * scales[e0 : e0 + chunk, ..., None].float()).flatten(-2)
        if hb is not None:
            w = (w.unflatten(-1, (-1, had)) @ hb).flatten(-2)
        out[e0 : e0 + chunk] = w
    return out


def _dense(slot: int, packed: torch.Tensor) -> torch.Tensor:
    E, N, G3 = packed.shape
    key = (slot, E, N, G3 // 3 * 32, packed.device)
    if key not in _DENSE:
        _DENSE[key] = torch.empty(key[1:4], dtype=torch.float16, device=packed.device)
    return _DENSE[key]


class OctaveW3DequantExperts(TritonExperts):
    """``w1``/``w2`` arrive in the planar checkpoint layout."""

    def __init__(
        self,
        moe_config: FusedMoEConfig,
        w1_scale: torch.Tensor,
        w2_scale: torch.Tensor,
        had_w13: int,
        had_w2: int,
    ):
        super().__init__(moe_config, FUSED_MOE_UNQUANTIZED_CONFIG)
        self.octw3_scales = (w1_scale, w2_scale)
        self.had_w13 = had_w13
        self.had_w2 = had_w2

    def apply(self, output, hidden_states, w1, w2, *args, **kwargs):
        s1, s2 = self.octw3_scales
        w1 = octw3_dequant(w1, s1, self.had_w13, _dense(0, w1))
        w2 = octw3_dequant(w2, s2, self.had_w2, _dense(1, w2))
        return super().apply(output, hidden_states, w1, w2, *args, **kwargs)
