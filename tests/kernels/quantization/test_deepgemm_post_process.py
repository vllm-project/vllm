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

    def test_bmm_batch_size_zero_raises_assertion(self):
        """Test that bmm_batch_size=0 raises AssertionError.

        Regression test for Issue #43174: ZeroDivisionError when
        bmm_batch_size is 0 due to TP+EP partitioning.
        Now we explicitly assert that this case is invalid.
        """
        wq = torch.randn(128, 256, dtype=torch.float32).to(torch.float8_e4m3fn)
        ws = torch.randn(1, 2, dtype=torch.float32)
        block_shape = (128, 128)

        with pytest.raises(AssertionError, match="bmm_batch_size must be > 0"):
            deepgemm_post_process_fp8_weight_block(
                wq=wq,
                ws=ws,
                quant_block_shape=block_shape,
                use_e8m0=False,
                is_bmm=True,
                bmm_batch_size=0,  # Invalid value
            )

    def test_bmm_batch_size_negative_raises_assertion(self):
        """Test that negative bmm_batch_size raises AssertionError."""
        wq = torch.randn(128, 256, dtype=torch.float32).to(torch.float8_e4m3fn)
        ws = torch.randn(1, 2, dtype=torch.float32)
        block_shape = (128, 128)

        with pytest.raises(AssertionError, match="bmm_batch_size must be > 0"):
            deepgemm_post_process_fp8_weight_block(
                wq=wq,
                ws=ws,
                quant_block_shape=block_shape,
                use_e8m0=False,
                is_bmm=True,
                bmm_batch_size=-1,
            )

    def test_bmm_batch_size_normal_case(self):
        """Test normal BMM reshape with valid batch size.

        Note: r must be a multiple of the block size (128) for the
        view operation to work correctly.
        """
        g, r, d = 4, 128, 256  # r=128 is a multiple of block_size=128
        wq = torch.randn(g * r, d, dtype=torch.float32).to(torch.float8_e4m3fn)
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


