# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Pytest tests for flashinfer_trtllm_bf16_moe in vllm.utils.flashinfer."""

import pytest
import torch
from unittest.mock import Mock

import vllm.utils.flashinfer as flashinfer_mod
from vllm.utils.flashinfer import flashinfer_trtllm_bf16_moe


def _clear_wrapper_cache() -> None:
    """Clear the lazy _get_impl cache so the next call re-resolves the backend."""
    for cell in flashinfer_trtllm_bf16_moe.__closure__ or ():
        try:
            fn = cell.cell_contents
            if callable(fn) and hasattr(fn, "cache_clear"):
                fn.cache_clear()
                break
        except (ValueError, AttributeError):
            continue


class TestFlashinferTrtllmBf16MoeWrapper:
    """Tests for the flashinfer_trtllm_bf16_moe lazy wrapper."""

    def test_wrapper_is_callable(self) -> None:
        """The wrapper is a callable (function)."""
        assert callable(flashinfer_trtllm_bf16_moe)

    def test_fallback_raises_when_flashinfer_unavailable(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When FlashInfer is not available, the wrapper raises RuntimeError."""
        monkeypatch.setattr(flashinfer_mod, "has_flashinfer", lambda: False)
        _clear_wrapper_cache()

        with pytest.raises(RuntimeError, match="FlashInfer backend is not available"):
            flashinfer_trtllm_bf16_moe(
                routing_logits=torch.zeros(1, 1, dtype=torch.bfloat16),
                routing_bias=None,
                hidden_states=torch.zeros(1, 1, dtype=torch.bfloat16),
                gemm1_weights=torch.zeros(1, 1, 1, dtype=torch.bfloat16),
                gemm2_weights=torch.zeros(1, 1, 1, dtype=torch.bfloat16),
                num_experts=4,
                top_k=1,
                n_group=0,
                topk_group=0,
                intermediate_size=128,
                local_expert_offset=0,
                local_num_experts=4,
                routing_method_type=0,
                tune_max_num_tokens=8192,
            )

    def test_wrapper_forwards_to_impl_when_mocked(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When the underlying impl is available (mocked), the wrapper forwards call."""
        mock_return = torch.tensor([1.0, 2.0], dtype=torch.bfloat16)
        mock_fn = Mock(return_value=mock_return)

        monkeypatch.setattr(flashinfer_mod, "has_flashinfer", lambda: True)
        monkeypatch.setattr(
            flashinfer_mod,
            "_get_submodule",
            lambda name: type(
                "FakeModule",
                (),
                {"trtllm_bf16_moe": mock_fn},
            )()
            if name == "flashinfer.fused_moe"
            else None,
        )
        _clear_wrapper_cache()

        result = flashinfer_trtllm_bf16_moe(
            routing_logits=torch.zeros(2, 4, dtype=torch.bfloat16),
            routing_bias=None,
            hidden_states=torch.zeros(2, 128, dtype=torch.bfloat16),
            gemm1_weights=torch.zeros(4, 256, 128, dtype=torch.bfloat16),
            gemm2_weights=torch.zeros(4, 128, 256, dtype=torch.bfloat16),
            num_experts=4,
            top_k=1,
            n_group=0,
            topk_group=0,
            intermediate_size=256,
            local_expert_offset=0,
            local_num_experts=4,
            routing_method_type=0,
            tune_max_num_tokens=8192,
        )

        assert mock_fn.called
        assert result is mock_return


def _make_bf16_moe_inputs(
    m: int,
    k: int,
    n: int,
    num_experts: int,
    top_k: int,
    device: str = "cuda",
    seed: int = 42,
):
    """Create bf16 MoE inputs: routing_logits, hidden_states, gemm1/gemm2 weights."""
    g = torch.Generator(device=device).manual_seed(seed)
    routing_logits = torch.randn(
        m, num_experts, device=device, dtype=torch.bfloat16, generator=g
    ) * 0.1
    hidden_states = torch.randn(
        m, k, device=device, dtype=torch.bfloat16, generator=g
    ) * 0.1
    # gemm1: (E, 2*n, k), gemm2: (E, k, n)
    gemm1_weights = torch.randn(
        num_experts,
        2 * n,
        k,
        device=device,
        dtype=torch.bfloat16,
        generator=g,
    ) * 0.1
    gemm2_weights = torch.randn(
        num_experts, k, n, device=device, dtype=torch.bfloat16, generator=g
    ) * 0.1
    return routing_logits, hidden_states, gemm1_weights, gemm2_weights


@pytest.mark.skipif(
    not flashinfer_mod.has_flashinfer(),
    reason="FlashInfer is not available",
)
@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA is required for bf16 MoE",
)
class TestFlashinferTrtllmBf16MoeIntegration:
    """Integration tests when FlashInfer and CUDA are available."""

    @pytest.mark.parametrize("m,k,n", [(8, 256, 512), (32, 512, 1024)])
    @pytest.mark.parametrize("num_experts,top_k", [(8, 2), (4, 1)])
    def test_wrapper_accepts_minimal_kwargs(
        self, m: int, k: int, n: int, num_experts: int, top_k: int
    ) -> None:
        intermediate_size = n

        routing_logits, hidden_states, gemm1_weights, gemm2_weights = (
            _make_bf16_moe_inputs(m, k, n, num_experts, top_k)
        )

        with torch.inference_mode():
            out = flashinfer_trtllm_bf16_moe(
                routing_logits=routing_logits,
                routing_bias=None,
                hidden_states=hidden_states,
                gemm1_weights=gemm1_weights,
                gemm2_weights=gemm2_weights,
                num_experts=num_experts,
                top_k=top_k,
                n_group=0,
                topk_group=0,
                intermediate_size=intermediate_size,
                local_expert_offset=0,
                local_num_experts=num_experts,
                routing_method_type=0,
                tune_max_num_tokens=8192,
            )

        assert out.dtype == torch.bfloat16

    @pytest.mark.parametrize("m,k,n", [(8, 256, 512), (32, 512, 1024)])
    @pytest.mark.parametrize("num_experts,top_k", [(8, 2), (4, 1)])
    def test_flashinfer_trtllm_bf16_moe_ones_routing_bias(
        self, m: int, k: int, n: int, num_experts: int, top_k: int
    ) -> None:
        routing_logits, hidden_states, gemm1_weights, gemm2_weights = (
            _make_bf16_moe_inputs(m, k, n, num_experts, top_k)
        )

        routing_bias = torch.ones(num_experts, device="cuda", dtype=torch.bfloat16)
        with torch.inference_mode():
            out = flashinfer_trtllm_bf16_moe(
                routing_logits=routing_logits,
                routing_bias=routing_bias,
                hidden_states=hidden_states,
                gemm1_weights=gemm1_weights,
                gemm2_weights=gemm2_weights,
                num_experts=num_experts,
                top_k=top_k,
                n_group=0,
                topk_group=0,
                intermediate_size=n,
                local_expert_offset=0,
                local_num_experts=num_experts,
                routing_method_type=0,
                tune_max_num_tokens=8192,
            )

        assert out.dtype == torch.bfloat16

    @pytest.mark.parametrize("m,k,n", [(8, 256, 512), (32, 512, 1024)])
    @pytest.mark.parametrize("num_experts,top_k", [(8, 2), (4, 1)])
    def test_flashinfer_trtllm_bf16_moe_with_zero_routing_bias(
        self, m: int, k: int, n: int, num_experts: int, top_k: int
    ) -> None:
        """Wrapper accepts optional routing_bias tensor."""
        routing_logits, hidden_states, gemm1_weights, gemm2_weights = (
            _make_bf16_moe_inputs(m, k, n, num_experts, top_k)
        )
        routing_bias = torch.zeros(num_experts, device="cuda", dtype=torch.bfloat16)

        with torch.inference_mode():
            out = flashinfer_trtllm_bf16_moe(
                routing_logits=routing_logits,
                routing_bias=routing_bias,
                hidden_states=hidden_states,
                gemm1_weights=gemm1_weights,
                gemm2_weights=gemm2_weights,
                num_experts=num_experts,
                top_k=top_k,
                n_group=0,
                topk_group=0,
                intermediate_size=n,
                local_expert_offset=0,
                local_num_experts=num_experts,
                routing_method_type=0,
                tune_max_num_tokens=8192,
            )

        assert out.dtype == torch.bfloat16

    @pytest.mark.parametrize("m,k,n", [(8, 256, 512), (32, 512, 1024)])
    @pytest.mark.parametrize("num_experts,top_k", [(8, 2), (4, 1)])
    def test_flashinfer_trtllm_bf16_moe_deterministic_same_input(
        self, m: int, k: int, n: int, num_experts: int, top_k: int
    ) -> None:
        """Same input produces same output (determinism)."""
        routing_logits, hidden_states, gemm1_weights, gemm2_weights = (
            _make_bf16_moe_inputs(m, k, n, num_experts, top_k, seed=123)
        )

        with torch.inference_mode():
            out1 = flashinfer_trtllm_bf16_moe(
                routing_logits=routing_logits,
                routing_bias=None,
                hidden_states=hidden_states,
                gemm1_weights=gemm1_weights,
                gemm2_weights=gemm2_weights,
                num_experts=num_experts,
                top_k=top_k,
                n_group=0,
                topk_group=0,
                intermediate_size=n,
                local_expert_offset=0,
                local_num_experts=num_experts,
                routing_method_type=0,
                tune_max_num_tokens=8192,
            )
            out2 = flashinfer_trtllm_bf16_moe(
                routing_logits=routing_logits,
                routing_bias=None,
                hidden_states=hidden_states,
                gemm1_weights=gemm1_weights,
                gemm2_weights=gemm2_weights,
                num_experts=num_experts,
                top_k=top_k,
                n_group=0,
                topk_group=0,
                intermediate_size=n,
                local_expert_offset=0,
                local_num_experts=num_experts,
                routing_method_type=0,
                tune_max_num_tokens=8192,
            )

        torch.testing.assert_close(out1, out2)
