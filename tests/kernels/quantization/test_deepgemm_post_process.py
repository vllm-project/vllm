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

    def test_bmm_batch_size_not_divisible_raises_assertion(self):
        """Test that non-divisible bmm_batch_size raises AssertionError.

        When n_groups is not evenly divisible by TP size, the computed
        bmm_batch_size would cause groups to be lost.
        """
        # wq size=128, bmm_batch_size=3 → 128 % 3 != 0
        wq = torch.randn(128, 256, dtype=torch.float32).to(torch.float8_e4m3fn)
        ws = torch.randn(1, 2, dtype=torch.float32)
        block_shape = (128, 128)

        with pytest.raises(AssertionError, match="not evenly divisible"):
            deepgemm_post_process_fp8_weight_block(
                wq=wq,
                ws=ws,
                quant_block_shape=block_shape,
                use_e8m0=False,
                is_bmm=True,
                bmm_batch_size=3,  # 128 % 3 != 0
            )
