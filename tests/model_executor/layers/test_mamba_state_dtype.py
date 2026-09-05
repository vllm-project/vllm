# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for MambaStateDtypeCalculator's Mamba1 SSM cache dtype
validation.

The Mamba1 selective_scan CUDA kernel (csrc/libtorch_stable/mamba/
selective_scan_fwd.cu) only accepts an SSM state dtype that is either
float32 or equal to the model's activation dtype; any other explicit
mamba_ssm_cache_dtype crashes EngineCore on the first request instead of
failing at config time. See https://github.com/vllm-project/vllm/issues/53239
"""

import pytest
import torch

from vllm.model_executor.layers.mamba.mamba_utils import MambaStateDtypeCalculator


class TestMamba1StateDtype:
    def test_auto_follows_model_dtype(self):
        conv_dtype, ssm_dtype = MambaStateDtypeCalculator.mamba1_state_dtype(
            torch.float16, "auto", "auto"
        )
        assert conv_dtype == torch.float16
        assert ssm_dtype == torch.float16

    def test_explicit_float32_allowed_for_any_model_dtype(self):
        for model_dtype in (torch.float16, torch.bfloat16, torch.float32):
            _, ssm_dtype = MambaStateDtypeCalculator.mamba1_state_dtype(
                model_dtype, "auto", "float32"
            )
            assert ssm_dtype == torch.float32

    def test_explicit_dtype_matching_model_dtype_allowed(self):
        _, ssm_dtype = MambaStateDtypeCalculator.mamba1_state_dtype(
            torch.bfloat16, "auto", "bfloat16"
        )
        assert ssm_dtype == torch.bfloat16

    def test_explicit_bfloat16_cache_with_float16_model_rejected(self):
        """Reproduces the crash from issue #53239: fp16 model activations
        with an explicit bf16 SSM cache trips the kernel's
        state_type == input_type || state_type == float32 assertion."""
        with pytest.raises(ValueError, match="mamba_ssm_cache_dtype"):
            MambaStateDtypeCalculator.mamba1_state_dtype(
                torch.float16, "auto", "bfloat16"
            )

    def test_explicit_float16_cache_with_bfloat16_model_rejected(self):
        """Same kernel constraint, opposite direction."""
        with pytest.raises(ValueError, match="mamba_ssm_cache_dtype"):
            MambaStateDtypeCalculator.mamba1_state_dtype(
                torch.bfloat16, "auto", "float16"
            )


def test_mamba2_state_dtype_does_not_apply_mamba1_kernel_restriction():
    """Mamba2 never calls selective_scan_fn (it uses
    mamba_chunk_scan_combined_varlen / selective_state_update instead), so
    the same mismatched dtype combination that is invalid for Mamba1 must
    stay valid here."""
    _, ssm_dtype = MambaStateDtypeCalculator.mamba2_state_dtype(
        torch.float16, "auto", "bfloat16"
    )
    assert ssm_dtype == torch.bfloat16


def test_gated_delta_net_state_dtype_does_not_apply_mamba1_kernel_restriction():
    _, ssm_dtype = MambaStateDtypeCalculator.gated_delta_net_state_dtype(
        torch.float16, "auto", "bfloat16"
    )
    assert ssm_dtype == torch.bfloat16
