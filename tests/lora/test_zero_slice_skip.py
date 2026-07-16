# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for zero-slice early-exit optimization.

Validates the slice_active_mask, the _num_adapters_with_active_slices counter
that drives the layer-level skip, and the _kernel_slice_mask gate that decides
whether to hand the kernels the mask. CPU-only, no GPU or distributed init.
"""

import pytest
import torch


class FakeBaseLayer:
    """Minimal fake to avoid ReplicatedLinear's distributed init."""

    input_size = 64
    output_size = 32
    tp_size = 1
    tp_rank = 0
    weight = torch.nn.Parameter(torch.randn(32, 64))


@pytest.fixture
def lora_layer():
    """Create a BaseLinearLayerWithLoRA with fake base layer."""
    from vllm.lora.layers.base_linear import BaseLinearLayerWithLoRA

    layer = BaseLinearLayerWithLoRA.__new__(BaseLinearLayerWithLoRA)
    layer.base_layer = FakeBaseLayer()
    layer.input_size = 64
    layer.output_size = 32
    layer.tp_size = 1
    layer.tp_rank = 0
    layer.device = torch.device("cpu")
    layer.n_slices = 1

    layer.lora_a_stacked = (torch.zeros(4, 1, 8, 64, dtype=torch.float16),)
    layer.lora_b_stacked = (torch.zeros(4, 1, 32, 8, dtype=torch.float16),)
    layer.output_slices = (32,)
    layer.slice_active_mask = torch.zeros(4, 1, dtype=torch.int32)
    layer._num_adapters_with_active_slices = 0
    layer._kernel_slice_mask = None
    return layer


class TestSliceActiveMask:
    def test_initial_state(self, lora_layer):
        assert lora_layer.slice_active_mask.sum() == 0
        assert lora_layer._num_adapters_with_active_slices == 0

    def test_set_lora_nonzero(self, lora_layer):
        lora_a = torch.randn(8, 64, dtype=torch.float16)
        lora_b = torch.randn(32, 8, dtype=torch.float16) * 0.01
        lora_layer.set_lora(0, lora_a, lora_b)
        assert lora_layer.slice_active_mask[0, 0] == 1
        assert lora_layer._num_adapters_with_active_slices == 1

    def test_set_lora_zero_a(self, lora_layer):
        lora_a = torch.zeros(8, 64, dtype=torch.float16)
        lora_b = torch.randn(32, 8, dtype=torch.float16)
        lora_layer.set_lora(0, lora_a, lora_b)
        assert lora_layer.slice_active_mask[0, 0] == 0
        assert lora_layer._num_adapters_with_active_slices == 0

    def test_set_lora_zero_b(self, lora_layer):
        lora_a = torch.randn(8, 64, dtype=torch.float16)
        lora_b = torch.zeros(32, 8, dtype=torch.float16)
        lora_layer.set_lora(0, lora_a, lora_b)
        assert lora_layer.slice_active_mask[0, 0] == 0
        assert lora_layer._num_adapters_with_active_slices == 0

    def test_reset_lora(self, lora_layer):
        lora_layer.set_lora(
            0,
            torch.randn(8, 64, dtype=torch.float16),
            torch.randn(32, 8, dtype=torch.float16) * 0.01,
        )
        assert lora_layer._num_adapters_with_active_slices == 1
        lora_layer.reset_lora(0)
        assert lora_layer.slice_active_mask[0, 0] == 0
        assert lora_layer._num_adapters_with_active_slices == 0

    def test_multiple_adapters(self, lora_layer):
        lora_layer.set_lora(
            0,
            torch.randn(8, 64, dtype=torch.float16),
            torch.randn(32, 8, dtype=torch.float16) * 0.01,
        )
        assert lora_layer._num_adapters_with_active_slices == 1

        # Adapter 1: zero weights
        lora_layer.set_lora(
            1,
            torch.zeros(8, 64, dtype=torch.float16),
            torch.zeros(32, 8, dtype=torch.float16),
        )
        assert lora_layer._num_adapters_with_active_slices == 1

        # Adapter 2: non-zero
        lora_layer.set_lora(
            2,
            torch.randn(8, 64, dtype=torch.float16),
            torch.randn(32, 8, dtype=torch.float16) * 0.01,
        )
        assert lora_layer._num_adapters_with_active_slices == 2

        lora_layer.reset_lora(0)
        assert lora_layer._num_adapters_with_active_slices == 1

        lora_layer.reset_lora(2)
        assert lora_layer._num_adapters_with_active_slices == 0

    def test_reset_zero_adapter_no_underflow(self, lora_layer):
        lora_layer.set_lora(
            0,
            torch.zeros(8, 64, dtype=torch.float16),
            torch.zeros(32, 8, dtype=torch.float16),
        )
        assert lora_layer._num_adapters_with_active_slices == 0
        lora_layer.reset_lora(0)
        assert lora_layer._num_adapters_with_active_slices == 0

    def test_single_slice_never_can_skip(self, lora_layer):
        # A 1-slice layer can never have a partial (some-active, some-zero)
        # pattern, so the kernel mask can never skip a CTA -> pass None.
        lora_layer.set_lora(
            0,
            torch.randn(8, 64, dtype=torch.float16),
            torch.randn(32, 8, dtype=torch.float16) * 0.01,
        )
        assert lora_layer._kernel_slice_mask is None


class TestKernelSliceMaskGate:
    """_kernel_slice_mask is the mask tensor only when some loaded adapter has
    a partial slice pattern (the sole case a per-CTA skip helps), else None.
    """

    def _make_layer(self, n_slices):
        from vllm.lora.layers.base_linear import BaseLinearLayerWithLoRA

        layer = BaseLinearLayerWithLoRA.__new__(BaseLinearLayerWithLoRA)
        layer.n_slices = n_slices
        layer.slice_active_mask = torch.zeros(4, n_slices, dtype=torch.int32)
        layer._num_adapters_with_active_slices = 0
        layer._kernel_slice_mask = None
        return layer

    def test_dense_multi_slice_cannot_skip(self):
        # All slices of the adapter active -> nothing to skip -> None.
        layer = self._make_layer(3)
        layer.slice_active_mask[0] = torch.tensor([1, 1, 1], dtype=torch.int32)
        layer._recompute_mask_can_skip()
        assert layer._kernel_slice_mask is None

    def test_partial_multi_slice_can_skip(self):
        # q,v active but k zero (e.g. adapter targets q+v not k) -> skip helps.
        layer = self._make_layer(3)
        layer.slice_active_mask[0] = torch.tensor([1, 0, 1], dtype=torch.int32)
        layer._recompute_mask_can_skip()
        assert layer._kernel_slice_mask is layer.slice_active_mask

    def test_recompute_after_reset(self):
        layer = self._make_layer(2)
        layer.slice_active_mask[0] = torch.tensor([1, 0], dtype=torch.int32)
        layer._recompute_mask_can_skip()
        assert layer._kernel_slice_mask is layer.slice_active_mask
        # Clear the partial adapter -> back to no skip.
        layer.slice_active_mask[0] = 0
        layer._recompute_mask_can_skip()
        assert layer._kernel_slice_mask is None
