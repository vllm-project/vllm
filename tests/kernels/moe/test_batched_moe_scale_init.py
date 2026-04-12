# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Regression test for uninitialized expert activation scales in batched MoE.

In vllm/model_executor/layers/fused_moe/fused_batched_moe.py, the
BatchedPrepareAndFinalize.prepare() method allocates expert activation
scale tensors.  When an expert receives 0 tokens (common in DP+EP),
its scale rows must be safely initialized (zeros), not garbage/NaN.

The bug: torch.empty() was used, leaving NaN in 0-token expert scales.
The fix: torch.zeros() ensures safe initialization.

These tests call the actual BatchedPrepareAndFinalize.prepare() method
with inputs crafted so some experts receive 0 tokens, then check the
returned scale tensor for NaN.
"""

import torch

from vllm.platforms import current_platform

DEVICE = torch.device(current_platform.device_type)


class TestBatchedMoEScaleInit:
    """Regression test for uninitialized expert activation scales.

    When an expert receives 0 tokens, its scale tensor must be zero-filled,
    not NaN from torch.empty().  NaN scales propagate through downstream
    GEMM operations and corrupt model outputs.
    """

    @staticmethod
    def _nan_empty(*args, **kwargs):
        """Drop-in replacement for torch.empty that fills with NaN.
        This simulates the worst case of recycled GPU memory containing
        NaN values, making the test deterministic."""
        t = torch._nan_empty_original(*args, **kwargs)
        if t.is_floating_point():
            t.fill_(float('nan'))
        return t

    def test_prepare_zero_token_experts_no_nan(self):
        """Call BatchedPrepareAndFinalize.prepare() with topk_ids that
        route NO tokens to some experts.  The returned b_a1_scale must
        not contain NaN for those experts.

        Fails without fix: torch.empty() leaves NaN/garbage in scale
        rows for experts that receive 0 tokens."""
        from unittest.mock import patch

        from vllm.model_executor.layers.fused_moe.config import (
            FusedMoEQuantConfig,
        )
        from vllm.model_executor.layers.fused_moe.fused_batched_moe import (
            BatchedPrepareAndFinalize,
        )

        num_local_experts = 8
        num_tokens = 16
        hidden_dim = 128
        topk = 2
        max_num_tokens = 64

        prep = BatchedPrepareAndFinalize(
            max_num_tokens=max_num_tokens,
            num_local_experts=num_local_experts,
            num_dispatchers=1,
            rank=0,
        )

        # Create inputs that route tokens to only experts 0 and 1
        # Experts 2-7 get 0 tokens
        a1 = torch.randn(
            num_tokens, hidden_dim, dtype=torch.float16, device=DEVICE,
        )
        topk_weights = torch.ones(
            num_tokens, topk, dtype=torch.float16, device=DEVICE,
        )
        # All tokens go to experts 0 and 1 only
        topk_ids = torch.zeros(
            num_tokens, topk, dtype=torch.int64, device=DEVICE,
        )
        topk_ids[:, 0] = 0  # first choice: expert 0
        topk_ids[:, 1] = 1  # second choice: expert 1

        quant_config = FusedMoEQuantConfig.make(
            quant_dtype=torch.float8_e4m3fn,
            per_act_token_quant=True,
        )

        # Patch torch.empty to deterministically return NaN-filled tensors.
        # This simulates the worst case for recycled GPU memory.
        torch._nan_empty_original = torch.empty
        try:
            with patch.object(torch, 'empty', self._nan_empty):
                b_a1, b_a1_scale, expert_tokens_meta, _, _ = prep.prepare(
                    a1=a1,
                    topk_weights=topk_weights,
                    topk_ids=topk_ids,
                    num_experts=num_local_experts,
                    expert_map=None,
                    apply_router_weight_on_input=False,
                    quant_config=quant_config,
                )
        finally:
            del torch._nan_empty_original

        torch.cuda.synchronize()

        assert b_a1_scale is not None, (
            "Scale tensor should exist for quantized config"
        )

        # Check experts that received 0 tokens (experts 2-7)
        for expert_id in range(2, num_local_experts):
            expert_scale = b_a1_scale[expert_id].cpu()
            has_nan = torch.isnan(expert_scale).any().item()
            assert not has_nan, (
                f"Expert {expert_id} received 0 tokens but has NaN in "
                f"scale tensor. This will corrupt MoE output when the "
                f"scale is used in downstream GEMM operations."
            )

    def test_prepare_block_quant_zero_token_experts(self):
        """Same test but with block quantization (block_shape=[128, 128])
        instead of per-token quantization."""
        from unittest.mock import patch

        from vllm.model_executor.layers.fused_moe.config import (
            FusedMoEQuantConfig,
        )
        from vllm.model_executor.layers.fused_moe.fused_batched_moe import (
            BatchedPrepareAndFinalize,
        )

        num_local_experts = 4
        num_tokens = 8
        hidden_dim = 256
        topk = 1
        max_num_tokens = 32

        prep = BatchedPrepareAndFinalize(
            max_num_tokens=max_num_tokens,
            num_local_experts=num_local_experts,
            num_dispatchers=1,
            rank=0,
        )

        a1 = torch.randn(
            num_tokens, hidden_dim, dtype=torch.float16, device=DEVICE,
        )
        topk_weights = torch.ones(
            num_tokens, topk, dtype=torch.float16, device=DEVICE,
        )
        # All tokens go to expert 0 only; experts 1-3 get nothing
        topk_ids = torch.zeros(
            num_tokens, topk, dtype=torch.int64, device=DEVICE,
        )

        quant_config = FusedMoEQuantConfig.make(
            quant_dtype=torch.float8_e4m3fn,
            block_shape=[128, 128],
        )

        # Patch torch.empty to deterministically return NaN
        torch._nan_empty_original = torch.empty
        try:
            with patch.object(torch, 'empty', self._nan_empty):
                b_a1, b_a1_scale, expert_tokens_meta, _, _ = prep.prepare(
                    a1=a1,
                    topk_weights=topk_weights,
                    topk_ids=topk_ids,
                    num_experts=num_local_experts,
                    expert_map=None,
                    apply_router_weight_on_input=False,
                    quant_config=quant_config,
                )
        finally:
            del torch._nan_empty_original

        torch.cuda.synchronize()

        assert b_a1_scale is not None
        for expert_id in range(1, num_local_experts):
            expert_scale = b_a1_scale[expert_id].cpu()
            has_nan = torch.isnan(expert_scale).any().item()
            assert not has_nan, (
                f"Expert {expert_id} (0 tokens, block quant) has NaN "
                f"in scale tensor."
            )

    def test_expert_scale_nan_propagation(self):
        """Demonstrate that NaN in scale tensors propagates through
        matrix multiplication — the mechanism by which uninitialized
        scales corrupt model outputs and eventually the KV cache."""
        num_experts = 4
        max_tokens = 32
        hidden_dim = 64

        activations = torch.randn(
            num_experts, max_tokens, hidden_dim,
            dtype=torch.float32, device=DEVICE,
        )
        weights = torch.randn(
            num_experts, hidden_dim, hidden_dim,
            dtype=torch.float32, device=DEVICE,
        )

        # Buggy: scales contain NaN for expert 2 (0 tokens)
        scales_buggy = torch.ones(
            num_experts, max_tokens, 1,
            dtype=torch.float32, device=DEVICE,
        )
        scales_buggy[2] = float('nan')

        # Fixed: scales are zero for expert 2
        scales_fixed = torch.ones(
            num_experts, max_tokens, 1,
            dtype=torch.float32, device=DEVICE,
        )
        scales_fixed[2] = 0.0

        output_buggy = torch.bmm(activations * scales_buggy, weights)
        output_fixed = torch.bmm(activations * scales_fixed, weights)

        torch.cuda.synchronize()

        assert torch.isnan(output_buggy[2]).all(), (
            "NaN scale propagates to ALL output values for that expert"
        )
        assert not torch.isnan(output_fixed).any(), (
            "Zero scale prevents NaN propagation"
        )
