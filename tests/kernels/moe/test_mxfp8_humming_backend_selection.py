# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MXFP8 MoE backend selection tests for Humming backend."""

import dataclasses
from unittest.mock import patch

import pytest

from tests.kernels.moe.utils import make_dummy_moe_config
from vllm.model_executor.layers.fused_moe.oracle.fp8 import Fp8MoeBackend
from vllm.model_executor.layers.fused_moe.oracle.mxfp8 import (
    select_mxfp8_moe_backend,
)


def test_humming_rejected_for_online_mxfp8():
    cfg = make_dummy_moe_config(num_experts=8, hidden_dim=256)
    cfg = dataclasses.replace(cfg, moe_backend="humming")

    with pytest.raises(
        ValueError,
        match="moe_backend='humming' cannot be used with quantization='mxfp8'",
    ):
        select_mxfp8_moe_backend(cfg, is_online=True)


def test_humming_skipped_in_auto_for_online_mxfp8():
    cfg = make_dummy_moe_config(num_experts=8, hidden_dim=256)
    cfg = dataclasses.replace(cfg, moe_backend="auto")

    def mock_select(backend, config):
        if backend == Fp8MoeBackend.HUMMING:
            return object
        if backend == Fp8MoeBackend.TRITON_MXFP8:
            return object
        raise ValueError("unsupported")

    with patch(
        "vllm.model_executor.layers.fused_moe.oracle.mxfp8._select_kernel_cls",
        side_effect=mock_select,
    ):
        backend, _ = select_mxfp8_moe_backend(cfg, is_online=True)
        assert backend == Fp8MoeBackend.TRITON_MXFP8


def test_humming_allowed_for_offline_mxfp8():
    cfg = make_dummy_moe_config(num_experts=8, hidden_dim=256)
    cfg = dataclasses.replace(cfg, moe_backend="humming")

    sentinel_cls = object()
    with patch(
        "vllm.model_executor.layers.fused_moe.oracle.mxfp8._select_kernel_cls",
        return_value=sentinel_cls,
    ):
        backend, cls = select_mxfp8_moe_backend(cfg, is_online=False)
        assert backend == Fp8MoeBackend.HUMMING
        assert cls is sentinel_cls
