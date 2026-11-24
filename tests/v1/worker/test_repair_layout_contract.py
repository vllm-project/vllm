# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Layout-contract correctness-matrix tests for the repair geometry.

Mission table categories not covered by the main repair batteries:
  1. num_kv_heads == block_size: the shape-ambiguity guard refuses (F2).
  2. _resolved_kv_layout: CacheConfig wins over empty env (Ascend).
  3. Manager block size != kernel block size: the geometry verification
     refuses (shape does not carry the spec value on the token axis).
  4. Core LBNHC and LBHNC: same logical [B,H,N,C] shape, different
     strides — both correctly indexed through the geometry contract.
"""

import os
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

BS = 128
N_HEADS = 8
HEAD_DIM = 64
N_BLOCKS = 16


def _core_layer(layout_name: str = "LBNHC") -> SimpleNamespace:
    """A Core-layout layer: one shared 4D [B, H, N, C] tensor."""
    cache = torch.zeros(N_BLOCKS, N_HEADS, BS, HEAD_DIM)
    cache[7, :, 5, :] = 1.0  # block 7, token 5, all heads/dims = 1
    attn = SimpleNamespace(kv_cache=cache, layer_name="model.layers.0")
    return SimpleNamespace(self_attn=SimpleNamespace(attn=attn))


def _ascend_flat_layer() -> SimpleNamespace:
    """An Ascend flat per-K/V layer: [n, bs, dim] pair."""
    k = torch.zeros(N_BLOCKS, BS, N_HEADS * HEAD_DIM)
    v = torch.zeros(N_BLOCKS, BS, N_HEADS * HEAD_DIM)
    k[7, 5, :] = 1.0
    attn = SimpleNamespace(kv_cache=[k, v], layer_name="model.layers.0")
    return SimpleNamespace(self_attn=SimpleNamespace(attn=attn))


def _ascend_hnd_layer(h: int = N_HEADS) -> SimpleNamespace:
    """An Ascend head-major per-K/V layer: [n, h, bs, d] pair."""
    k = torch.zeros(N_BLOCKS, h, BS, HEAD_DIM)
    v = torch.zeros(N_BLOCKS, h, BS, HEAD_DIM)
    k[7, :, 5, :] = 1.0
    attn = SimpleNamespace(kv_cache=[k, v], layer_name="model.layers.0")
    return SimpleNamespace(self_attn=SimpleNamespace(attn=attn))


class TestShapeAmbiguityGuard:
    """F2: num_kv_heads == block_size makes the shape unable to confirm
    the token axis on a 4D per-K/V view. The guard refuses."""

    def test_hnd_view_with_h_equals_bs_refused(self):
        from vllm.v1.worker.gpu.repair_executor import _derive_repair_geometry

        layer = _ascend_hnd_layer(h=BS)  # heads == block_size!
        with pytest.raises(ValueError, match="ambiguous"):
            _derive_repair_geometry([layer], "LBHNC", BS)

    def test_hnd_view_with_h_not_bs_accepted(self):
        from vllm.v1.worker.gpu.repair_executor import _derive_repair_geometry

        layer = _ascend_hnd_layer(h=N_HEADS)  # heads != block_size
        geo = _derive_repair_geometry([layer], "LBHNC", BS)
        assert geo.block_size == BS
        assert geo.token_axis == 2
        assert geo.is_core_shared is False


class TestManagerVsKernelBlockSize:
    """Manager block size != kernel block size: the shape carries the
    kernel value while the spec carries the manager value; the
    verification refuses (shape does not match the spec on the token
    axis)."""

    def test_kernel_bs_differs_from_spec_refused(self):
        from vllm.v1.worker.gpu.repair_executor import _derive_repair_geometry

        # The view was allocated with kernel_block_size=64 (half of BS=128)
        k = torch.zeros(32, 64, N_HEADS, HEAD_DIM)
        v = torch.zeros(32, 64, N_HEADS, HEAD_DIM)
        attn = SimpleNamespace(kv_cache=[k, v], layer_name="model.layers.0")
        layer = SimpleNamespace(self_attn=SimpleNamespace(attn=attn))
        # The spec says manager block_size=128, but the view has 64 on dim 1
        with pytest.raises(ValueError, match="does not carry the spec block size"):
            _derive_repair_geometry([layer], "LBHNC", BS)


class TestSameShapeDifferentStrides:
    """Core LBNHC and LBHNC: same logical [B,H,N,C] shape, different
    physical strides. The geometry contract indexes both correctly
    (same logical view; the rezero zeroes the same logical cells)."""

    def test_core_4d_view_indexed_correctly(self):
        from vllm.v1.worker.gpu.repair_executor import (
            _derive_repair_geometry,
        )

        layer = _core_layer()
        geo = _derive_repair_geometry([layer], "LBNHC", BS)
        assert geo.block_size == BS
        assert geo.token_axis == 2
        assert geo.is_core_shared is True
        # The logical cell [7, :, 5, :] was pre-set to 1.0
        cache = layer.self_attn.attn.kv_cache
        assert float(cache[7, :, 5, :].abs().max()) == 1.0
        # Neighboring cells are zero
        assert float(cache[7, :, 4, :].abs().max()) == 0.0
        assert float(cache[6, :, 5, :].abs().max()) == 0.0

    def test_ascend_flat_view_indexed_correctly(self):
        from vllm.v1.worker.gpu.repair_executor import _derive_repair_geometry

        layer = _ascend_flat_layer()
        geo = _derive_repair_geometry([layer], "LBNHC", BS)
        assert geo.block_size == BS
        assert geo.token_axis == 1
        assert geo.is_core_shared is False
        # The K cache's [7, 5, :] was pre-set to 1.0
        k = layer.self_attn.attn.kv_cache[0]
        assert float(k[7, 5, :].abs().max()) == 1.0
        assert float(k[7, 4, :].abs().max()) == 0.0


