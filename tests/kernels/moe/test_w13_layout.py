# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the MXFP4 w13 interleave/deinterleave helpers used to share one
conversion path between interleaved (GPT-OSS) and contiguous checkpoints.

Run: pytest tests/kernels/moe/test_w13_layout.py -v
"""

import pytest
import torch

from vllm.model_executor.layers.fused_moe.oracle.mxfp4 import (
    _deinterleave_w13,
    _interleave_w13,
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
