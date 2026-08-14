# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for W13Layout enum, interleave/deinterleave helpers, and the
merged oracle functions.

Run: pytest tests/kernels/moe/test_w13_layout.py -v
"""

from unittest.mock import patch

import pytest
import torch

from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.modular_kernel import W13Layout
from vllm.model_executor.layers.fused_moe.oracle.mxfp4 import (
    _deinterleave_w13,
    _interleave_w13,
    _swap_w13_halves,
    convert_w13_layout,
)

pytestmark = pytest.mark.cpu_test


class TestInterleaveDeinterleaveRoundTrip:
    """_interleave_w13 and _deinterleave_w13 must be inverses."""

    @staticmethod
    def _make_w13(e: int = 2, n: int = 8, k: int = 4) -> torch.Tensor:
        return torch.arange(e * n * k, dtype=torch.uint8).reshape(e, n, k)

    @staticmethod
    def _make_scale(e: int = 2, n: int = 8, k_scale: int = 2) -> torch.Tensor:
        return torch.arange(e * n * k_scale, dtype=torch.float32).reshape(e, n, k_scale)

    @staticmethod
    def _make_bias(e: int = 2, n: int = 8) -> torch.Tensor:
        return torch.arange(e * n, dtype=torch.float32).reshape(e, n)

    def test_interleave_then_deinterleave_is_identity(self):
        w = self._make_w13()
        s = self._make_scale()
        b = self._make_bias()

        wi, si, bi = _interleave_w13(w.clone(), s.clone(), b.clone())
        wd, sd, bd = _deinterleave_w13(wi, si, bi)

        torch.testing.assert_close(wd.view(torch.uint8), w)
        torch.testing.assert_close(sd, s)
        torch.testing.assert_close(bd, b)

    def test_deinterleave_then_interleave_is_identity(self):
        w = self._make_w13()
        s = self._make_scale()
        b = self._make_bias()

        wd, sd, bd = _deinterleave_w13(w.clone(), s.clone(), b.clone())
        wi, si, bi = _interleave_w13(wd, sd, bd)

        torch.testing.assert_close(wi.view(torch.uint8), w)
        torch.testing.assert_close(si, s)
        torch.testing.assert_close(bi, b)

    def test_interleave_produces_correct_pattern(self):
        w = torch.tensor([[[10, 11], [20, 21], [30, 31], [40, 41]]], dtype=torch.uint8)
        s = torch.tensor([[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]])
        # Contiguous: [gate0, gate1, up0, up1]
        # Interleaved: [gate0, up0, gate1, up1]
        wi, si, _ = _interleave_w13(w, s, None)
        expected_w = torch.tensor(
            [[[10, 11], [30, 31], [20, 21], [40, 41]]], dtype=torch.uint8
        )
        expected_s = torch.tensor([[[1.0, 2.0], [5.0, 6.0], [3.0, 4.0], [7.0, 8.0]]])
        torch.testing.assert_close(wi.view(torch.uint8), expected_w)
        torch.testing.assert_close(si, expected_s)

    def test_round_trip_without_bias(self):
        w = self._make_w13()
        s = self._make_scale()

        wi, si, bi = _interleave_w13(w.clone(), s.clone(), None)
        assert bi is None
        wd, sd, bd = _deinterleave_w13(wi, si, None)
        assert bd is None

        torch.testing.assert_close(wd.view(torch.uint8), w)
        torch.testing.assert_close(sd, s)


class TestSwapW13Halves:
    """_swap_w13_halves flips gate/up ordering."""

    @staticmethod
    def _make_w13(e: int = 1, n: int = 4, k: int = 2) -> torch.Tensor:
        return torch.arange(e * n * k, dtype=torch.uint8).reshape(e, n, k)

    @staticmethod
    def _make_scale(e: int = 1, n: int = 4, k_scale: int = 1) -> torch.Tensor:
        return torch.arange(e * n * k_scale, dtype=torch.float32).reshape(e, n, k_scale)

    @staticmethod
    def _make_bias(e: int = 1, n: int = 4) -> torch.Tensor:
        return torch.arange(e * n, dtype=torch.float32).reshape(e, n)

    def test_swap_flips_halves(self):
        w = torch.tensor([[[10, 11], [20, 21], [30, 31], [40, 41]]], dtype=torch.uint8)
        s = torch.tensor([[[1.0], [2.0], [3.0], [4.0]]])
        b = torch.tensor([[100.0, 200.0, 300.0, 400.0]])

        ws, ss, bs = _swap_w13_halves(w, s, b)

        expected_w = torch.tensor(
            [[[30, 31], [40, 41], [10, 11], [20, 21]]], dtype=torch.uint8
        )
        expected_s = torch.tensor([[[3.0], [4.0], [1.0], [2.0]]])
        expected_b = torch.tensor([[300.0, 400.0, 100.0, 200.0]])
        torch.testing.assert_close(ws, expected_w)
        torch.testing.assert_close(ss, expected_s)
        torch.testing.assert_close(bs, expected_b)

    def test_double_swap_is_identity(self):
        w = self._make_w13()
        s = self._make_scale()
        b = self._make_bias()

        w1, s1, b1 = _swap_w13_halves(w.clone(), s.clone(), b.clone())
        w2, s2, b2 = _swap_w13_halves(w1, s1, b1)

        torch.testing.assert_close(w2, w)
        torch.testing.assert_close(s2, s)
        torch.testing.assert_close(b2, b)

    def test_swap_without_bias(self):
        w = self._make_w13()
        s = self._make_scale()
        _, _, b_out = _swap_w13_halves(w, s, None)
        assert b_out is None


class TestConvertW13Layout:
    """convert_w13_layout handles all from/to combinations."""

    @staticmethod
    def _make_contiguous_w1w3():
        """[gate0, gate1, up0, up1] — 1 expert, 4 rows, 2 cols."""
        w = torch.tensor([[[10, 11], [20, 21], [30, 31], [40, 41]]], dtype=torch.uint8)
        s = torch.tensor([[[1.0], [2.0], [3.0], [4.0]]])
        b = torch.tensor([[100.0, 200.0, 300.0, 400.0]])
        return w, s, b

    def test_same_layout_is_noop(self):
        w, s, b = self._make_contiguous_w1w3()
        wo, so, bo = convert_w13_layout(
            w, s, b, W13Layout.CONTIGUOUS_W1W3, W13Layout.CONTIGUOUS_W1W3
        )
        assert wo is w
        assert so is s
        assert bo is b

    @pytest.mark.parametrize(
        "from_layout, to_layout",
        [
            (W13Layout.CONTIGUOUS_W1W3, W13Layout.CONTIGUOUS_W3W1),
            (W13Layout.CONTIGUOUS_W1W3, W13Layout.INTERLEAVED_W1W3),
            (W13Layout.CONTIGUOUS_W1W3, W13Layout.INTERLEAVED_W3W1),
            (W13Layout.CONTIGUOUS_W3W1, W13Layout.CONTIGUOUS_W1W3),
            (W13Layout.CONTIGUOUS_W3W1, W13Layout.INTERLEAVED_W1W3),
            (W13Layout.CONTIGUOUS_W3W1, W13Layout.INTERLEAVED_W3W1),
            (W13Layout.INTERLEAVED_W1W3, W13Layout.CONTIGUOUS_W1W3),
            (W13Layout.INTERLEAVED_W1W3, W13Layout.CONTIGUOUS_W3W1),
            (W13Layout.INTERLEAVED_W1W3, W13Layout.INTERLEAVED_W3W1),
            (W13Layout.INTERLEAVED_W3W1, W13Layout.CONTIGUOUS_W1W3),
            (W13Layout.INTERLEAVED_W3W1, W13Layout.CONTIGUOUS_W3W1),
            (W13Layout.INTERLEAVED_W3W1, W13Layout.INTERLEAVED_W1W3),
        ],
    )
    def test_round_trip(self, from_layout, to_layout):
        w, s, b = self._make_contiguous_w1w3()
        w_orig, s_orig, b_orig = w.clone(), s.clone(), b.clone()

        w1, s1, b1 = convert_w13_layout(
            w_orig, s_orig, b_orig, W13Layout.CONTIGUOUS_W1W3, from_layout
        )
        w2, s2, b2 = convert_w13_layout(w1, s1, b1, from_layout, to_layout)
        w3, s3, b3 = convert_w13_layout(
            w2, s2, b2, to_layout, W13Layout.CONTIGUOUS_W1W3
        )

        torch.testing.assert_close(w3.view(torch.uint8), w)
        torch.testing.assert_close(s3, s)
        torch.testing.assert_close(b3, b)

    def test_contiguous_to_interleaved_pattern(self):
        w, s, b = self._make_contiguous_w1w3()
        wi, si, _ = convert_w13_layout(
            w, s, None, W13Layout.CONTIGUOUS_W1W3, W13Layout.INTERLEAVED_W1W3
        )
        expected_w = torch.tensor(
            [[[10, 11], [30, 31], [20, 21], [40, 41]]], dtype=torch.uint8
        )
        expected_s = torch.tensor([[[1.0], [3.0], [2.0], [4.0]]])
        torch.testing.assert_close(wi.view(torch.uint8), expected_w)
        torch.testing.assert_close(si, expected_s)

    def test_contiguous_w1w3_to_contiguous_w3w1(self):
        w, s, b = self._make_contiguous_w1w3()
        ws, ss, bs = convert_w13_layout(
            w, s, b, W13Layout.CONTIGUOUS_W1W3, W13Layout.CONTIGUOUS_W3W1
        )
        expected_w = torch.tensor(
            [[[30, 31], [40, 41], [10, 11], [20, 21]]], dtype=torch.uint8
        )
        torch.testing.assert_close(ws, expected_w)

    def test_contiguous_w1w3_to_interleaved_w3w1(self):
        w, s, b = self._make_contiguous_w1w3()
        wo, so, bo = convert_w13_layout(
            w, s, b, W13Layout.CONTIGUOUS_W1W3, W13Layout.INTERLEAVED_W3W1
        )
        expected_w = torch.tensor(
            [[[30, 31], [10, 11], [40, 41], [20, 21]]], dtype=torch.uint8
        )
        torch.testing.assert_close(wo.view(torch.uint8), expected_w)


class TestExpectedW13Layout:
    """Each expert class must return the correct W13Layout."""

    @pytest.mark.parametrize(
        "activation, expected",
        [
            (MoEActivation.SILU, W13Layout.CONTIGUOUS_W1W3),
            (MoEActivation.GELU, W13Layout.CONTIGUOUS_W1W3),
            (MoEActivation.SWIGLUOAI, W13Layout.INTERLEAVED_W1W3),
            (MoEActivation.SWIGLUOAI_UNINTERLEAVE, W13Layout.CONTIGUOUS_W1W3),
            (MoEActivation.SITU, W13Layout.CONTIGUOUS_W1W3),
            (MoEActivation.SWIGLUSTEP, W13Layout.CONTIGUOUS_W1W3),
        ],
    )
    def test_base_class_default(self, activation, expected):
        from vllm.model_executor.layers.fused_moe.modular_kernel import (
            FusedMoEExperts,
        )

        assert FusedMoEExperts._expected_w13_layout(activation) == expected


class TestDeprecatedWrappers:
    """Deprecated functions must delegate to the merged versions."""

    def test_select_gpt_oss_delegates(self):
        from vllm.model_executor.layers.fused_moe.oracle import mxfp4 as mod

        sentinel = (object(), object())
        with patch.object(
            mod,
            "select_mxfp4_moe_backend",
            return_value=sentinel,
        ) as mocked:
            config = object()
            result = mod.select_gpt_oss_mxfp4_moe_backend(config)

        mocked.assert_called_once_with(config, None, use_gpt_oss_priority=True)
        assert result is sentinel

    def test_convert_gpt_oss_delegates(self):
        from vllm.model_executor.layers.fused_moe.oracle import mxfp4 as mod

        sentinel = (None,) * 6
        with patch.object(
            mod,
            "convert_weight_to_mxfp4_moe_kernel_format",
            return_value=sentinel,
        ) as mocked:
            w = torch.zeros(1)
            result = mod.convert_gpt_oss_weight_to_mxfp4_moe_kernel_format(
                mxfp4_backend=mod.Mxfp4MoeBackend.MARLIN,
                layer=None,
                w13_weight=w,
                w2_weight=w,
                w13_weight_scale=w,
                w2_weight_scale=w,
            )

        assert result is sentinel
        _, kwargs = mocked.call_args
        assert kwargs["input_w13_layout"] == W13Layout.INTERLEAVED_W1W3
