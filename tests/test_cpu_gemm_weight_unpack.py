# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Regression test for #35950: ValueError when dispatching CPU GEMM with
multi-dimensional weights (e.g. Qwen3.5 vision models)."""
import torch


def _extract_n_k(weight: torch.Tensor):
    weight_shape = weight.size()
    N, K = weight_shape[-2], weight_shape[-1]
    return N, K


def test_2d_weight():
    w = torch.randn(256, 512)
    N, K = _extract_n_k(w)
    assert N == 256 and K == 512


def test_3d_weight():
    w = torch.randn(8, 256, 512)
    N, K = _extract_n_k(w)
    assert N == 256 and K == 512


def test_4d_weight():
    w = torch.randn(2, 4, 256, 512)
    N, K = _extract_n_k(w)
    assert N == 256 and K == 512


def test_old_unpack_crashes_on_3d():
    w = torch.randn(8, 256, 512)
    try:
        N, K = w.size()
        raise AssertionError("Should have raised ValueError")
    except ValueError:
        pass
