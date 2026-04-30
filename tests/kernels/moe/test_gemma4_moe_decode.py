# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for Gemma4 MoE decode GEMV kernels.

Validates correctness of the optimized CUDA GEMV kernels against PyTorch
reference implementations:

- Expert GEMV: gate_up + GELU + down projection
- Routing: softmax -> top-K -> renormalize -> per_expert_scale

These kernels target Gemma4 MoE decode (small batch T<=8, E=128 experts,
top_k=8, GELU activation, bf16 weights).
"""

import pytest
import torch

# Gemma4-26B-A4B dimensions
HIDDEN_SIZE = 2816
INTERMEDIATE_SIZE = 352  # per TP shard (moe_intermediate_size=704 / TP=2)
NUM_EXPERTS = 128
TOP_K = 8

BATCH_SIZES = [1, 2, 4, 8]


def _skip_if_no_kernels():
    """Skip test if CUDA GEMV kernels cannot be compiled."""
    try:
        from vllm.model_executor.layers.fused_moe.gemma4_moe_decode import (
            is_available,
        )

        if not is_available():
            pytest.skip("Gemma4 decode GEMV kernels not available")
    except Exception as e:
        pytest.skip(f"Cannot import gemma4_moe_decode: {e}")


def _gelu_tanh(x: torch.Tensor) -> torch.Tensor:
    """GELU with tanh approximation matching CUDA kernel."""
    return torch.nn.functional.gelu(x, approximate="tanh")


def _reference_expert_forward(
    hidden_states: torch.Tensor,
    w13: torch.Tensor,
    w2: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    intermediate_size: int,
) -> torch.Tensor:
    """Pure PyTorch reference for MoE expert forward.

    Processes each (token, expert_slot) assignment independently:
      gate_out = x @ W_gate.T, up_out = x @ W_up.T
      hidden = GELU(gate_out) * up_out
      out += routing_weight * hidden @ W_down.T
    """
    T, H = hidden_states.shape
    K = topk_ids.shape[1]
    N = intermediate_size

    output = torch.zeros(T, H, dtype=torch.float32, device=hidden_states.device)

    for t in range(T):
        for k in range(K):
            expert_id = topk_ids[t, k].item()
            weight = topk_weights[t, k].item()

            x = hidden_states[t].float()

            # w13 is [E, 2*N, H] - gate is first N rows, up is second N rows
            w_gate = w13[expert_id, :N, :].float()  # [N, H]
            w_up = w13[expert_id, N:, :].float()  # [N, H]
            w_down = w2[expert_id].float()  # [H, N]

            gate_out = x @ w_gate.T  # [N]
            up_out = x @ w_up.T  # [N]
            hidden = _gelu_tanh(gate_out) * up_out  # [N]
            out = hidden @ w_down.T  # [H]

            output[t] += weight * out

    return output.to(torch.bfloat16)


def _reference_routing(
    router_logits: torch.Tensor,
    per_expert_scale: torch.Tensor,
    top_k: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pure PyTorch reference for Gemma4 routing.

    Matches gemma4_routing_function_torch from vllm/model_executor/models/gemma4.py.
    """
    _, topk_ids = torch.topk(router_logits, k=top_k, dim=-1)
    router_probs = torch.nn.functional.softmax(router_logits, dim=-1)
    indicator = torch.nn.functional.one_hot(
        topk_ids, num_classes=router_logits.size(-1)
    ).sum(dim=-2)
    gate_weights = indicator * router_probs
    renorm_factor = torch.sum(gate_weights, dim=-1, keepdim=True)
    renorm_factor = torch.where(renorm_factor > 0.0, renorm_factor, 1.0)
    dispatch_weights = gate_weights / renorm_factor
    topk_weights = dispatch_weights.gather(1, topk_ids)

    expert_scales = per_expert_scale[topk_ids].to(topk_weights.dtype)
    topk_weights = topk_weights * expert_scales

    return topk_weights.to(torch.float32), topk_ids.to(torch.int32)


def _sort_by_id(
    weights: torch.Tensor, ids: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sort topk results by expert id for comparison."""
    order = ids.argsort(dim=-1)
    return weights.gather(1, order), ids.gather(1, order)


# -----------------------------------------------------------------------
# Expert GEMV correctness tests
# -----------------------------------------------------------------------


class TestGemma4MoEDecodeForward:
    """Test optimized CUDA expert GEMV against PyTorch reference."""

    @pytest.mark.parametrize("batch_size", BATCH_SIZES)
    def test_correctness(self, batch_size: int):
        _skip_if_no_kernels()
        from vllm.model_executor.layers.fused_moe.gemma4_moe_decode import (
            gemma4_decode_expert_forward,
        )

        gen = torch.Generator(device="cuda").manual_seed(42)

        hidden_states = torch.randn(
            batch_size,
            HIDDEN_SIZE,
            dtype=torch.bfloat16,
            device="cuda",
            generator=gen,
        )
        # Use smaller expert count for test speed, but still "many experts"
        num_experts = 16
        w13 = (
            torch.randn(
                num_experts,
                2 * INTERMEDIATE_SIZE,
                HIDDEN_SIZE,
                dtype=torch.bfloat16,
                device="cuda",
                generator=gen,
            )
            * 0.01
        )
        w2 = (
            torch.randn(
                num_experts,
                HIDDEN_SIZE,
                INTERMEDIATE_SIZE,
                dtype=torch.bfloat16,
                device="cuda",
                generator=gen,
            )
            * 0.01
        )

        topk_ids = torch.randint(
            0,
            num_experts,
            (batch_size, TOP_K),
            dtype=torch.int32,
            device="cuda",
        )
        topk_weights = torch.rand(
            batch_size,
            TOP_K,
            dtype=torch.float32,
            device="cuda",
            generator=gen,
        )
        # Normalize weights per token
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)

        cuda_out = gemma4_decode_expert_forward(
            hidden_states, w13, w2, topk_ids, topk_weights, INTERMEDIATE_SIZE
        )
        ref_out = _reference_expert_forward(
            hidden_states, w13, w2, topk_ids, topk_weights, INTERMEDIATE_SIZE
        )

        assert cuda_out.shape == ref_out.shape
        assert cuda_out.dtype == torch.bfloat16
        # bf16 accumulation tolerance
        torch.testing.assert_close(cuda_out, ref_out, atol=5e-2, rtol=5e-2)

    def test_single_token(self):
        """BS=1 is the most common decode case."""
        _skip_if_no_kernels()
        from vllm.model_executor.layers.fused_moe.gemma4_moe_decode import (
            gemma4_decode_expert_forward,
        )

        gen = torch.Generator(device="cuda").manual_seed(123)
        hidden = torch.randn(
            1, HIDDEN_SIZE, dtype=torch.bfloat16, device="cuda", generator=gen
        )
        num_experts = 16
        w13 = (
            torch.randn(
                num_experts,
                2 * INTERMEDIATE_SIZE,
                HIDDEN_SIZE,
                dtype=torch.bfloat16,
                device="cuda",
                generator=gen,
            )
            * 0.01
        )
        w2 = (
            torch.randn(
                num_experts,
                HIDDEN_SIZE,
                INTERMEDIATE_SIZE,
                dtype=torch.bfloat16,
                device="cuda",
                generator=gen,
            )
            * 0.01
        )
        topk_ids = torch.randint(
            0, num_experts, (1, TOP_K), dtype=torch.int32, device="cuda"
        )
        topk_weights = torch.ones(1, TOP_K, dtype=torch.float32, device="cuda") / TOP_K

        out = gemma4_decode_expert_forward(
            hidden, w13, w2, topk_ids, topk_weights, INTERMEDIATE_SIZE
        )
        assert out.shape == (1, HIDDEN_SIZE)
        assert out.dtype == torch.bfloat16
        assert torch.isfinite(out).all()

    def test_all_tokens_same_expert(self):
        """Edge case: all tokens route to the same expert."""
        _skip_if_no_kernels()
        from vllm.model_executor.layers.fused_moe.gemma4_moe_decode import (
            gemma4_decode_expert_forward,
        )

        gen = torch.Generator(device="cuda").manual_seed(7)
        T = 4
        num_experts = 16
        hidden = torch.randn(
            T, HIDDEN_SIZE, dtype=torch.bfloat16, device="cuda", generator=gen
        )
        w13 = (
            torch.randn(
                num_experts,
                2 * INTERMEDIATE_SIZE,
                HIDDEN_SIZE,
                dtype=torch.bfloat16,
                device="cuda",
                generator=gen,
            )
            * 0.01
        )
        w2 = (
            torch.randn(
                num_experts,
                HIDDEN_SIZE,
                INTERMEDIATE_SIZE,
                dtype=torch.bfloat16,
                device="cuda",
                generator=gen,
            )
            * 0.01
        )

        # All tokens select expert 0 for all K slots
        topk_ids = torch.zeros(T, TOP_K, dtype=torch.int32, device="cuda")
        topk_weights = torch.ones(T, TOP_K, dtype=torch.float32, device="cuda") / TOP_K

        cuda_out = gemma4_decode_expert_forward(
            hidden, w13, w2, topk_ids, topk_weights, INTERMEDIATE_SIZE
        )
        ref_out = _reference_expert_forward(
            hidden, w13, w2, topk_ids, topk_weights, INTERMEDIATE_SIZE
        )

        assert cuda_out.shape == ref_out.shape
        torch.testing.assert_close(cuda_out, ref_out, atol=5e-2, rtol=5e-2)


# -----------------------------------------------------------------------
# Routing correctness tests
# -----------------------------------------------------------------------


class TestGemma4DecodeRouting:
    """Test CUDA routing kernel against PyTorch reference."""

    @pytest.mark.parametrize("batch_size", BATCH_SIZES)
    def test_routing_correctness(self, batch_size: int):
        _skip_if_no_kernels()
        from vllm.model_executor.layers.fused_moe.gemma4_moe_decode import (
            gemma4_decode_routing,
        )

        gen = torch.Generator(device="cuda").manual_seed(42)
        router_logits = torch.randn(
            batch_size,
            NUM_EXPERTS,
            dtype=torch.float32,
            device="cuda",
            generator=gen,
        )
        per_expert_scale = (
            torch.rand(
                NUM_EXPERTS,
                dtype=torch.float32,
                device="cuda",
                generator=gen,
            )
            + 0.5
        )

        cuda_weights, cuda_ids = gemma4_decode_routing(
            router_logits, per_expert_scale, TOP_K
        )
        ref_weights, ref_ids = _reference_routing(
            router_logits, per_expert_scale, TOP_K
        )

        # Sort by expert id to handle tie-breaking differences
        cuda_w_s, cuda_ids_s = _sort_by_id(cuda_weights, cuda_ids)
        ref_w_s, ref_ids_s = _sort_by_id(ref_weights, ref_ids)

        assert (cuda_ids_s == ref_ids_s).all(), (
            f"Expert ids mismatch at batch_size={batch_size}"
        )
        torch.testing.assert_close(cuda_w_s, ref_w_s, atol=1e-3, rtol=1e-3)

    def test_routing_output_dtypes(self):
        _skip_if_no_kernels()
        from vllm.model_executor.layers.fused_moe.gemma4_moe_decode import (
            gemma4_decode_routing,
        )

        logits = torch.randn(4, NUM_EXPERTS, dtype=torch.float32, device="cuda")
        scale = torch.ones(NUM_EXPERTS, dtype=torch.float32, device="cuda")
        weights, ids = gemma4_decode_routing(logits, scale, TOP_K)
        assert weights.dtype == torch.float32
        assert ids.dtype == torch.int32

    def test_routing_output_shapes(self):
        _skip_if_no_kernels()
        from vllm.model_executor.layers.fused_moe.gemma4_moe_decode import (
            gemma4_decode_routing,
        )

        logits = torch.randn(8, NUM_EXPERTS, dtype=torch.float32, device="cuda")
        scale = torch.ones(NUM_EXPERTS, dtype=torch.float32, device="cuda")
        weights, ids = gemma4_decode_routing(logits, scale, TOP_K)
        assert weights.shape == (8, TOP_K)
        assert ids.shape == (8, TOP_K)

    def test_expert_ids_valid_range(self):
        _skip_if_no_kernels()
        from vllm.model_executor.layers.fused_moe.gemma4_moe_decode import (
            gemma4_decode_routing,
        )

        logits = torch.randn(16, NUM_EXPERTS, dtype=torch.float32, device="cuda")
        scale = torch.ones(NUM_EXPERTS, dtype=torch.float32, device="cuda")
        _, ids = gemma4_decode_routing(logits, scale, TOP_K)
        assert (ids >= 0).all()
        assert (ids < NUM_EXPERTS).all()

    def test_no_duplicate_experts_per_token(self):
        _skip_if_no_kernels()
        from vllm.model_executor.layers.fused_moe.gemma4_moe_decode import (
            gemma4_decode_routing,
        )

        logits = torch.randn(8, NUM_EXPERTS, dtype=torch.float32, device="cuda")
        scale = torch.ones(NUM_EXPERTS, dtype=torch.float32, device="cuda")
        _, ids = gemma4_decode_routing(logits, scale, TOP_K)
        for i in range(ids.shape[0]):
            assert ids[i].unique().numel() == TOP_K, (
                f"Token {i} has duplicate expert ids: {ids[i].tolist()}"
            )


# -----------------------------------------------------------------------
# Edge case tests
# -----------------------------------------------------------------------


class TestEdgeCases:
    """Edge cases for decode GEMV correctness."""

    def test_identical_tokens_same_routing(self):
        """All tokens with identical hidden states should get same routing."""
        _skip_if_no_kernels()
        from vllm.model_executor.layers.fused_moe.gemma4_moe_decode import (
            gemma4_decode_routing,
        )

        single_logits = torch.randn(1, NUM_EXPERTS, dtype=torch.float32, device="cuda")
        logits = single_logits.expand(8, -1).contiguous()
        scale = torch.rand(NUM_EXPERTS, dtype=torch.float32, device="cuda") + 0.5

        weights, ids = gemma4_decode_routing(logits, scale, TOP_K)

        for i in range(1, 8):
            w0_s, id0_s = _sort_by_id(weights[0:1], ids[0:1])
            wi_s, idi_s = _sort_by_id(weights[i : i + 1], ids[i : i + 1])
            assert (id0_s == idi_s).all(), f"Token 0 and {i} selected different experts"
            torch.testing.assert_close(w0_s, wi_s, atol=1e-6, rtol=1e-6)

    def test_weights_positive(self):
        """Routing weights should be positive."""
        _skip_if_no_kernels()
        from vllm.model_executor.layers.fused_moe.gemma4_moe_decode import (
            gemma4_decode_routing,
        )

        logits = torch.randn(4, NUM_EXPERTS, dtype=torch.float32, device="cuda")
        scale = torch.rand(NUM_EXPERTS, dtype=torch.float32, device="cuda") + 0.5
        weights, _ = gemma4_decode_routing(logits, scale, TOP_K)
        assert (weights > 0).all()
