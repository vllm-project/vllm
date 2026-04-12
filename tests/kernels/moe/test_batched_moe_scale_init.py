# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Regression test for uninitialized expert activation scales in batched MoE.

In vllm/model_executor/layers/fused_moe/fused_batched_moe.py, the
BatchedPrepareAndFinalize.prepare() method allocates expert activation
scale tensors.  When an expert receives 0 tokens, its scale rows must
be safely initialized (zeros), not garbage/NaN.

The bug: torch.empty() was used, leaving NaN in 0-token expert scales.
The fix: torch.zeros() ensures safe initialization.

These tests verify:
1. The prepare() method doesn't leave NaN in scale tensors (unit test)
2. End-to-end MoE kernels (Triton, DeepGEMM, NVFP4) produce NaN-free
   output when some experts receive 0 tokens
"""

from unittest.mock import patch

import pytest
import torch

from tests.kernels.moe.utils import make_dummy_moe_config, make_test_weights
from vllm.config import ParallelConfig, VllmConfig, set_current_vllm_config  # noqa: E501
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEQuantConfig,
    fp8_w8a8_moe_quant_config,
    nvfp4_moe_quant_config,
)
from vllm.model_executor.layers.fused_moe.fused_batched_moe import (
    BatchedPrepareAndFinalize,
    BatchedTritonExperts,
)
from vllm.model_executor.layers.fused_moe.modular_kernel import (
    FusedMoEKernel,
    MoEActivation,
)
from vllm.platforms import current_platform

DEVICE = torch.device(current_platform.device_type)

BLOCK_SHAPE = [128, 128]


def _nan_empty(*args, **kwargs):
    """Drop-in replacement for torch.empty that fills with NaN.
    Simulates worst-case recycled GPU memory."""
    t = torch._nan_empty_original(*args, **kwargs)
    if t.is_floating_point():
        t.fill_(float('nan'))
    return t


def _patch_empty():
    """Context manager that patches torch.empty to return NaN-filled tensors."""
    torch._nan_empty_original = torch.empty
    return patch.object(torch, 'empty', _nan_empty)


def _make_routing_with_empty_experts(
    num_tokens: int,
    num_experts: int,
    topk: int,
    active_experts: list[int],
    dtype: torch.dtype = torch.bfloat16,
):
    """Create topk_weights/topk_ids that route all tokens to active_experts
    only, leaving the rest with 0 tokens."""
    topk_ids = torch.zeros(num_tokens, topk, dtype=torch.int32, device=DEVICE)
    for k_idx in range(topk):
        expert = active_experts[k_idx % len(active_experts)]
        topk_ids[:, k_idx] = expert
    topk_weights = torch.ones(num_tokens, topk, dtype=dtype, device=DEVICE)
    topk_weights /= topk  # normalize
    return topk_weights, topk_ids


# ============================================================================
# 1. Unit tests: prepare() scale initialization
# ============================================================================


class TestBatchedMoEScaleInit:
    """Verify that prepare() zero-fills scale tensors for 0-token experts."""

    def test_prepare_zero_token_experts_no_nan(self):
        """Per-token quant: 0-token experts must have zero scales, not NaN."""
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

        a1 = torch.randn(num_tokens, hidden_dim,
                          dtype=torch.float16, device=DEVICE)
        topk_weights, topk_ids = _make_routing_with_empty_experts(
            num_tokens, num_local_experts, topk, active_experts=[0, 1],
            dtype=torch.float16,
        )

        quant_config = FusedMoEQuantConfig.make(
            quant_dtype=torch.float8_e4m3fn,
            per_act_token_quant=True,
        )

        torch._nan_empty_original = torch.empty
        try:
            with _patch_empty():
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

        for expert_id in range(2, num_local_experts):
            expert_scale = b_a1_scale[expert_id].cpu()
            assert not torch.isnan(expert_scale).any(), (
                f"Expert {expert_id} received 0 tokens but has NaN in "
                f"scale tensor."
            )

    def test_prepare_block_quant_zero_token_experts(self):
        """Block quant: 0-token experts must have zero scales, not NaN."""
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

        a1 = torch.randn(num_tokens, hidden_dim,
                          dtype=torch.float16, device=DEVICE)
        topk_weights, topk_ids = _make_routing_with_empty_experts(
            num_tokens, num_local_experts, topk, active_experts=[0],
            dtype=torch.float16,
        )

        quant_config = FusedMoEQuantConfig.make(
            quant_dtype=torch.float8_e4m3fn,
            block_shape=BLOCK_SHAPE,
        )

        torch._nan_empty_original = torch.empty
        try:
            with _patch_empty():
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
            assert not torch.isnan(expert_scale).any(), (
                f"Expert {expert_id} (0 tokens, block quant) has NaN "
                f"in scale tensor."
            )


# ============================================================================
# 2. End-to-end: full kernel with 0-token experts produces NaN-free output
# ============================================================================


def _make_block_quant_fp8_weights(e, n, k):
    """Create FP8 block-quantized weights for e experts."""
    from tests.kernels.moe.test_deepgemm import make_block_quant_fp8_weights
    return make_block_quant_fp8_weights(e, n, k, BLOCK_SHAPE)


class TestBatchedMoEEndToEnd:
    """End-to-end tests: run full MoE kernel with 0-token experts and
    verify the final output is NaN-free."""

    def test_batched_triton_zero_token_experts(self, workspace_init):
        """BatchedTritonExperts with FP8 block quant: 0-token experts
        must not inject NaN into output."""
        E, N, K = 8, 512, 256
        num_tokens = 32
        topk = 2

        w1, w2, w1_s, w2_s = _make_block_quant_fp8_weights(E, N, K)

        a = torch.randn(num_tokens, K, device=DEVICE,
                         dtype=torch.bfloat16) / 10
        fp8_info = torch.finfo(torch.float8_e4m3fn)
        a.clamp_(fp8_info.min, fp8_info.max)

        # Route all tokens to experts 0 and 1 only
        topk_weights, topk_ids = _make_routing_with_empty_experts(
            num_tokens, E, topk, active_experts=[0, 1])

        max_num_tokens = 64
        quant_config = fp8_w8a8_moe_quant_config(
            w1_scale=w1_s, w2_scale=w2_s,
            per_act_token_quant=False, block_shape=BLOCK_SHAPE,
        )

        kernel = FusedMoEKernel(
            BatchedPrepareAndFinalize(
                max_num_tokens=max_num_tokens,
                num_local_experts=E,
                num_dispatchers=1, rank=0,
            ),
            BatchedTritonExperts(
                max_num_tokens=max_num_tokens,
                num_dispatchers=1,
                quant_config=quant_config,
                moe_config=make_dummy_moe_config(),
            ),
            inplace=False,
        )

        with set_current_vllm_config(VllmConfig()):
            output = kernel.apply(
                hidden_states=a, w1=w1, w2=w2,
                topk_weights=topk_weights, topk_ids=topk_ids,
                global_num_experts=E,
                activation=MoEActivation.SILU,
                apply_router_weight_on_input=False,
                expert_map=None,
            )

        torch.cuda.synchronize()
        assert not torch.isnan(output).any(), (
            "BatchedTritonExperts output contains NaN with 0-token experts"
        )
        assert not torch.isinf(output).any(), (
            "BatchedTritonExperts output contains Inf with 0-token experts"
        )

    @pytest.mark.skipif(
        not current_platform.has_device_capability(89),
        reason="Requires sm89+ for FP8",
    )
    def test_batched_triton_per_token_quant_zero_token_experts(
        self, workspace_init,
    ):
        """BatchedTritonExperts with per-token FP8 quant: 0-token experts
        must not inject NaN into output."""
        E, N, K = 8, 512, 256
        num_tokens = 32
        topk = 2

        # Per-token quant needs per-channel weight scales [E, 1, 1]
        fp8_info = torch.finfo(torch.float8_e4m3fn)
        w1_bf16 = torch.randn(E, 2 * N, K, device=DEVICE,
                               dtype=torch.bfloat16) / 10
        w2_bf16 = torch.randn(E, K, N, device=DEVICE,
                               dtype=torch.bfloat16) / 10
        w1_bf16.clamp_(fp8_info.min, fp8_info.max)
        w2_bf16.clamp_(fp8_info.min, fp8_info.max)

        w1 = w1_bf16.to(torch.float8_e4m3fn)
        w2 = w2_bf16.to(torch.float8_e4m3fn)
        w1_s = torch.ones(E, dtype=torch.float32, device=DEVICE)
        w2_s = torch.ones(E, dtype=torch.float32, device=DEVICE)

        a = torch.randn(num_tokens, K, device=DEVICE,
                         dtype=torch.bfloat16) / 10

        topk_weights, topk_ids = _make_routing_with_empty_experts(
            num_tokens, E, topk, active_experts=[0, 1])

        max_num_tokens = 64
        quant_config = fp8_w8a8_moe_quant_config(
            w1_scale=w1_s, w2_scale=w2_s,
            per_act_token_quant=True, block_shape=None,
        )

        kernel = FusedMoEKernel(
            BatchedPrepareAndFinalize(
                max_num_tokens=max_num_tokens,
                num_local_experts=E,
                num_dispatchers=1, rank=0,
            ),
            BatchedTritonExperts(
                max_num_tokens=max_num_tokens,
                num_dispatchers=1,
                quant_config=quant_config,
                moe_config=make_dummy_moe_config(),
            ),
            inplace=False,
        )

        with set_current_vllm_config(VllmConfig()):
            output = kernel.apply(
                hidden_states=a, w1=w1, w2=w2,
                topk_weights=topk_weights, topk_ids=topk_ids,
                global_num_experts=E,
                activation=MoEActivation.SILU,
                apply_router_weight_on_input=False,
                expert_map=None,
            )

        torch.cuda.synchronize()
        assert not torch.isnan(output).any(), (
            "BatchedTritonExperts (per-token quant) output contains NaN"
        )

    @pytest.mark.skipif(
        not current_platform.has_device_capability(89),
        reason="Requires sm89+ for DeepGEMM FP8",
    )
    def test_batched_deepgemm_zero_token_experts(
        self, monkeypatch, workspace_init,
    ):
        """BatchedDeepGemmExperts with FP8 block quant: 0-token experts
        must not inject NaN into output."""
        from vllm.utils.deep_gemm import is_deep_gemm_supported

        if not is_deep_gemm_supported():
            pytest.skip("Requires deep_gemm kernels")

        from vllm.model_executor.layers.fused_moe.experts.batched_deep_gemm_moe import (
            BatchedDeepGemmExperts,
        )

        monkeypatch.setenv("VLLM_USE_DEEP_GEMM", "1")

        E, N, K = 8, 512, 256
        num_tokens = 32
        topk = 2

        w1, w2, w1_s, w2_s = _make_block_quant_fp8_weights(E, N, K)

        a = torch.randn(num_tokens, K, device=DEVICE,
                         dtype=torch.bfloat16) / 10
        fp8_info = torch.finfo(torch.float8_e4m3fn)
        a.clamp_(fp8_info.min, fp8_info.max)

        topk_weights, topk_ids = _make_routing_with_empty_experts(
            num_tokens, E, topk, active_experts=[0, 1])

        max_num_tokens = 64
        quant_config = fp8_w8a8_moe_quant_config(
            w1_scale=w1_s, w2_scale=w2_s,
            per_act_token_quant=False, block_shape=BLOCK_SHAPE,
        )

        kernel = FusedMoEKernel(
            BatchedPrepareAndFinalize(
                max_num_tokens=max_num_tokens,
                num_local_experts=E,
                num_dispatchers=1, rank=0,
            ),
            BatchedDeepGemmExperts(
                max_num_tokens=max_num_tokens,
                num_dispatchers=1,
                quant_config=quant_config,
                moe_config=make_dummy_moe_config(),
            ),
            inplace=False,
        )

        with set_current_vllm_config(VllmConfig()):
            output = kernel.apply(
                hidden_states=a, w1=w1, w2=w2,
                topk_weights=topk_weights, topk_ids=topk_ids,
                global_num_experts=E,
                activation=MoEActivation.SILU,
                apply_router_weight_on_input=False,
                expert_map=None,
            )

        torch.cuda.synchronize()
        assert not torch.isnan(output).any(), (
            "BatchedDeepGemmExperts output contains NaN with 0-token experts"
        )
        assert not torch.isinf(output).any(), (
            "BatchedDeepGemmExperts output contains Inf with 0-token experts"
        )

    @pytest.mark.skipif(
        not current_platform.has_device_capability(100),
        reason="NVFP4 requires sm100+",
    )
    def test_cutlass_fp4_zero_token_experts(self, workspace_init):
        """CutlassExpertsFp4 (NVFP4): 0-token experts must not inject
        NaN into output."""
        from vllm.model_executor.layers.fused_moe.cutlass_moe import (
            CutlassExpertsFp4,
        )
        from vllm.model_executor.layers.fused_moe.all2all_utils import (
            maybe_make_prepare_finalize,
        )

        E, N, K = 8, 1024, 1024
        num_tokens = 32
        topk = 2

        (_, w1_q, w1_blockscale, w1_gs), (_, w2_q, w2_blockscale, w2_gs) = (
            make_test_weights(
                E, N, K,
                in_dtype=torch.bfloat16,
                quant_dtype="nvfp4",
            )
        )

        a = torch.randn(num_tokens, K, device=DEVICE,
                         dtype=torch.bfloat16) / 10

        topk_weights, topk_ids = _make_routing_with_empty_experts(
            num_tokens, E, topk, active_experts=[0, 1])

        a1_gs = torch.ones((E,), device=DEVICE, dtype=torch.float32)
        a2_gs = torch.ones((E,), device=DEVICE, dtype=torch.float32)

        quant_config = nvfp4_moe_quant_config(
            g1_alphas=(1 / w1_gs),
            g2_alphas=(1 / w2_gs),
            a1_gscale=a1_gs,
            a2_gscale=a2_gs,
            w1_scale=w1_blockscale,
            w2_scale=w2_blockscale,
        )
        moe_config = make_dummy_moe_config()

        kernel = FusedMoEKernel(
            maybe_make_prepare_finalize(
                moe=moe_config,
                quant_config=quant_config,
                allow_new_interface=True,
                use_monolithic=False,
            ),
            CutlassExpertsFp4(
                moe_config=moe_config,
                quant_config=quant_config,
            ),
            inplace=False,
        )

        with set_current_vllm_config(
            VllmConfig(
                parallel_config=ParallelConfig(pipeline_parallel_size=1)
            )
        ):
            output = kernel.apply(
                hidden_states=a, w1=w1_q, w2=w2_q,
                topk_weights=topk_weights, topk_ids=topk_ids,
                global_num_experts=E,
                activation=MoEActivation.SILU,
                apply_router_weight_on_input=False,
                expert_map=None,
            )

        torch.cuda.synchronize()
        assert not torch.isnan(output).any(), (
            "CutlassExpertsFp4 (NVFP4) output contains NaN with "
            "0-token experts"
        )
        assert not torch.isinf(output).any(), (
            "CutlassExpertsFp4 (NVFP4) output contains Inf with "
            "0-token experts"
        )
