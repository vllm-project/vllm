# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Test deepgemm_post_process_fp8_weight_block edge cases.

Regression tests for GitHub Issue #43174:
ZeroDivisionError in deepgemm_post_process_fp8_weight_block when loading
FP8 model with TP=16 on dual-node H20.
"""

import pytest
import torch

from vllm.model_executor.layers.quantization.utils.fp8_utils import (
    deepgemm_post_process_fp8_weight_block,
)


class TestDeepgemmPostProcess:
    """Test deepgemm_post_process_fp8_weight_block edge cases."""

    def test_bmm_batch_size_zero_returns_empty_3d(self):
        """Test that bmm_batch_size=0 returns empty 3D tensors.

        Regression test for Issue #43174: ZeroDivisionError when
        bmm_batch_size is 0 due to TP+EP partitioning.
        """
        wq = torch.randn(128, 256, dtype=torch.float8_e4m3fn)
        ws = torch.randn(1, 2, dtype=torch.float32)
        block_shape = (128, 128)

        # Should not raise ZeroDivisionError
        result_wq, result_ws = deepgemm_post_process_fp8_weight_block(
            wq=wq,
            ws=ws,
            quant_block_shape=block_shape,
            use_e8m0=False,
            is_bmm=True,
            bmm_batch_size=0,  # Invalid value
        )

        # Should return empty 3D tensors with correct layout
        assert result_wq.ndim == 3
        assert result_wq.shape[0] == 0  # Empty batch dimension
        assert result_ws.ndim == 3
        assert result_ws.shape[0] == 0  # Empty batch dimension

    def test_bmm_batch_size_negative_returns_empty_3d(self):
        """Test that negative bmm_batch_size returns empty 3D tensors."""
        wq = torch.randn(128, 256, dtype=torch.float8_e4m3fn)
        ws = torch.randn(1, 2, dtype=torch.float32)
        block_shape = (128, 128)

        result_wq, result_ws = deepgemm_post_process_fp8_weight_block(
            wq=wq,
            ws=ws,
            quant_block_shape=block_shape,
            use_e8m0=False,
            is_bmm=True,
            bmm_batch_size=-1,
        )

        # Should return empty 3D tensors with correct layout
        assert result_wq.ndim == 3
        assert result_wq.shape[0] == 0  # Empty batch dimension
        assert result_ws.ndim == 3
        assert result_ws.shape[0] == 0  # Empty batch dimension

    def test_bmm_batch_size_normal_case(self):
        """Test normal BMM reshape with valid batch size.
        
        Note: r must be a multiple of the block size (128) for the
        view operation to work correctly.
        """
        g, r, d = 4, 128, 256  # r=128 is a multiple of block_size=128
        wq = torch.randn(g * r, d, dtype=torch.float8_e4m3fn)
        ws = torch.randn(
            g * r // 128, d // 128, dtype=torch.float32
        )
        block_shape = (128, 128)

        result_wq, result_ws = deepgemm_post_process_fp8_weight_block(
            wq=wq,
            ws=ws,
            quant_block_shape=block_shape,
            use_e8m0=False,
            is_bmm=True,
            bmm_batch_size=g,
        )

        assert result_wq.shape == (g, r, d)
        assert result_ws.ndim == 3
        assert result_ws.shape[0] == g