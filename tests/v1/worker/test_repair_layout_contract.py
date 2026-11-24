# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Binding-contract layout matrix tests for the repair geometry.

The geometry contract: the model runner supplies a RepairViewBinding
(how K and V are represented, which axis indexes tokens, that the
leading dim is blocks); the executor validates the actual bound views
against it by shape AND strides, and refuses on any disagreement --
never guessing an axis from a shape coincidence.

  1. num_kv_heads == block_size: the shape cannot confirm the token
     axis on a per-K/V view, but the binding is authoritative -- the
     view is accepted and indexed through the binding's axis.
  2. Manager block size != kernel block size: the view carries the
     kernel value where the binding carries the manager value; the
     geometry validation refuses before any write.
  3. Core logical [B, H, N, C]: the same logical shape under LBNHC and
     LBHNC physical strides; both bindings index the same logical rows,
     and the stride order verifies the declared physical layout.
"""

from types import SimpleNamespace

import pytest
import torch

from vllm.v1.worker.gpu.repair_executor import (
    RepairViewBinding,
    _repair_cache_views,
    _repair_geometry_from_binding,
    _verify_repair_layer_geometry,
    core_logical_repair_binding,
    per_kv_repair_binding,
)

BS = 128
N_HEADS = 8
HEAD_DIM = 64
N_BLOCKS = 16


def _core_layer(layout_name: str = "LBNHC") -> SimpleNamespace:
    """A Core logical [B, H, N, C] view over a physical allocation of
    the declared layout (LBNHC: a permuted view of [B, N, H, C]; LBHNC:
    the row-major allocation itself)."""
    if layout_name in ("LBNHC", "NHD"):
        storage = torch.zeros(N_BLOCKS, BS, N_HEADS, HEAD_DIM)
        view = storage.permute(0, 2, 1, 3)
    else:
        view = torch.zeros(N_BLOCKS, N_HEADS, BS, HEAD_DIM)
    view[7, :, 5, :] = 1.0  # block 7, token 5, all heads/dims = 1
    attn = SimpleNamespace(kv_cache=view, layer_name="model.layers.0")
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


class TestHeadEqualsBlockSize:
    """num_kv_heads == block_size makes the shape unable to confirm the
    token axis on a 4D per-K/V view; the binding is authoritative, so
    the view is ACCEPTED and indexed through the bound axis."""

    def test_hnd_binding_with_h_equals_bs_accepted(self):
        layer = _ascend_hnd_layer(h=BS)  # heads == block_size!
        binding = per_kv_repair_binding(BS, 2, "LBHNC")
        geo = _repair_geometry_from_binding(binding, _repair_cache_views(layer))
        assert geo.block_size == BS
        assert geo.token_axis == 2
        assert geo.is_core_shared is False
        _verify_repair_layer_geometry(layer, geo, binding)

    def test_lbnhc_binding_with_h_equals_bs_accepted(self):
        k = torch.zeros(N_BLOCKS, BS, BS, HEAD_DIM)  # [n, bs, h==bs, d]
        v = torch.zeros(N_BLOCKS, BS, BS, HEAD_DIM)
        attn = SimpleNamespace(kv_cache=[k, v], layer_name="model.layers.0")
        layer = SimpleNamespace(self_attn=SimpleNamespace(attn=attn))
        binding = per_kv_repair_binding(BS, 1, "LBNHC")
        geo = _repair_geometry_from_binding(binding, _repair_cache_views(layer))
        assert geo.token_axis == 1
        assert geo.is_core_shared is False

    def test_single_flat_view_accepted_with_per_kv_binding(self):
        # The flat convention may bind a lone view; its rows are zeroed
        # per token regardless of how K and V are packed into it.
        k = torch.zeros(N_BLOCKS, BS, 8)
        attn = SimpleNamespace(kv_cache=k, layer_name="model.layers.0")
        layer = SimpleNamespace(self_attn=SimpleNamespace(attn=attn))
        binding = per_kv_repair_binding(BS, 1, "LBNHC")
        geo = _repair_geometry_from_binding(binding, _repair_cache_views(layer))
        assert geo.token_axis == 1
        assert geo.is_core_shared is False

    def test_hnd_view_with_h_not_bs_accepted(self):
        layer = _ascend_hnd_layer(h=N_HEADS)  # heads != block_size
        binding = per_kv_repair_binding(BS, 2, "LBHNC")
        geo = _repair_geometry_from_binding(binding, _repair_cache_views(layer))
        assert geo.block_size == BS
        assert geo.token_axis == 2
        assert geo.is_core_shared is False


class TestManagerVsKernelBlockSize:
    """Manager block size != kernel block size: the shape carries the
    kernel value while the binding carries the manager value; the
    validation refuses (the view does not match the binding on the
    token axis)."""

    def test_kernel_bs_differs_from_binding_refused(self):
        # The view was allocated with kernel_block_size=64 (half of BS).
        k = torch.zeros(32, 64, N_HEADS, HEAD_DIM)
        v = torch.zeros(32, 64, N_HEADS, HEAD_DIM)
        attn = SimpleNamespace(kv_cache=[k, v], layer_name="model.layers.0")
        layer = SimpleNamespace(self_attn=SimpleNamespace(attn=attn))
        with pytest.raises(ValueError, match="block size"):
            _repair_geometry_from_binding(
                per_kv_repair_binding(BS, 1, "LBNHC"), _repair_cache_views(layer)
            )

    def test_kernel_bs_core_view_refused(self):
        layer = _core_layer("LBHNC")
        with pytest.raises(ValueError, match="block size"):
            _repair_geometry_from_binding(
                core_logical_repair_binding(BS * 2, "LBHNC"),
                _repair_cache_views(layer),
            )


class TestSameShapeDifferentStrides:
    """Core LBNHC and LBHNC: the same logical [B, H, N, C] shape with
    different physical strides. Both bindings accept their views, both
    re-zero the SAME logical rows, and a binding whose declared layout
    contradicts the view's stride order is refused."""

    def test_core_logical_view_indexed_correctly(self):
        layer = _core_layer("LBNHC")
        binding = core_logical_repair_binding(BS, "LBNHC")
        geo = _repair_geometry_from_binding(binding, _repair_cache_views(layer))
        assert geo.block_size == BS
        assert geo.token_axis == 2
        assert geo.is_core_shared is True
        cache = layer.self_attn.attn.kv_cache
        assert float(cache[7, :, 5, :].abs().max()) == 1.0
        # Neighboring cells are zero
        assert float(cache[7, :, 4, :].abs().max()) == 0.0
        assert float(cache[6, :, 5, :].abs().max()) == 0.0

    def test_stride_order_must_match_declared_layout(self):
        # The view IS the row-major LBHNC physical form; declaring the
        # LBNHC layout (which binds a permuted logical view) contradicts
        # the strides: refused, even when every shape matches.
        layer = _core_layer("LBHNC")
        with pytest.raises(ValueError, match="not consistent with the core"):
            _repair_geometry_from_binding(
                core_logical_repair_binding(BS, "LBNHC"),
                _repair_cache_views(layer),
            )
        # The truthful binding for the same view is accepted.
        _repair_geometry_from_binding(
            core_logical_repair_binding(BS, "LBHNC"), _repair_cache_views(layer)
        )

    def test_ascend_flat_view_indexed_correctly(self):
        layer = _ascend_flat_layer()
        binding = per_kv_repair_binding(BS, 1, "LBNHC")
        geo = _repair_geometry_from_binding(binding, _repair_cache_views(layer))
        assert geo.block_size == BS
        assert geo.token_axis == 1
        assert geo.is_core_shared is False
        k = layer.self_attn.attn.kv_cache[0]
        assert float(k[7, 5, :].abs().max()) == 1.0
        assert float(k[7, 4, :].abs().max()) == 0.0


class TestBindingContractRefusals:
    """Missing or contradictory binding facts fail closed."""

    def test_missing_binding_refused(self):
        layer = _core_layer("LBHNC")
        with pytest.raises(ValueError, match="no repair view binding"):
            _repair_geometry_from_binding(None, _repair_cache_views(layer))

    def test_unknown_binding_kind_refused(self):
        layer = _core_layer("LBHNC")
        binding = RepairViewBinding(
            kind="mystery",
            block_size=BS,
            token_axis=2,
            blocks_axis=0,
            layout_name="LBHNC",
        )
        with pytest.raises(ValueError, match="unknown repair view binding kind"):
            _repair_geometry_from_binding(binding, _repair_cache_views(layer))

    def test_non_leading_blocks_axis_refused(self):
        layer = _core_layer("LBHNC")
        binding = RepairViewBinding(
            kind="core_logical",
            block_size=BS,
            token_axis=2,
            blocks_axis=1,
            layout_name="LBHNC",
        )
        with pytest.raises(ValueError, match="blocks axis"):
            _repair_geometry_from_binding(binding, _repair_cache_views(layer))

    def test_per_kv_binding_refuses_shared_core_view(self):
        layer = _core_layer("LBHNC")
        with pytest.raises(ValueError, match="contradict the per-K/V binding"):
            _repair_geometry_from_binding(
                per_kv_repair_binding(BS, 2, "LBHNC"), _repair_cache_views(layer)
            )

    def test_core_binding_refuses_per_kv_views(self):
        layer = _ascend_hnd_layer()
        with pytest.raises(ValueError, match="contradict the core"):
            _repair_geometry_from_binding(
                core_logical_repair_binding(BS, "LBHNC"),
                _repair_cache_views(layer),
            )

    def test_factory_rejects_unknown_token_axis(self):
        with pytest.raises(ValueError, match="not a known per-K/V convention"):
            per_kv_repair_binding(BS, 3, "LBHNC")
