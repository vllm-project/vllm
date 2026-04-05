# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Pytest tests for flashinfer_trtllm_fp8_block_scale_moe in vllm.utils.flashinfer."""

import pytest
import torch
from unittest.mock import Mock

import vllm.utils.flashinfer as flashinfer_mod
from vllm.utils.flashinfer import flashinfer_trtllm_fp8_block_scale_moe

# Block layout required by FlashInfer FP8 block-scale MoE
BLOCK_M, BLOCK_N = 128, 128


def _clear_wrapper_cache() -> None:
    """Clear the lazy _get_impl cache so the next call re-resolves the backend."""
    for cell in flashinfer_trtllm_fp8_block_scale_moe.__closure__ or ():
        try:
            fn = cell.cell_contents
            if callable(fn) and hasattr(fn, "cache_clear"):
                fn.cache_clear()
                break
        except (ValueError, AttributeError):
            continue


class TestFlashinferTrtllmFp8BlockScaleMoeWrapper:
    """Tests for the flashinfer_trtllm_fp8_block_scale_moe lazy wrapper."""

    def test_wrapper_is_callable(self) -> None:
        """The wrapper is a callable (function)."""
        assert callable(flashinfer_trtllm_fp8_block_scale_moe)

    def test_fallback_raises_when_flashinfer_unavailable(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When FlashInfer is not available, the wrapper raises RuntimeError."""
        monkeypatch.setattr(flashinfer_mod, "has_flashinfer", lambda: False)
        _clear_wrapper_cache()

        with pytest.raises(RuntimeError, match="FlashInfer backend is not available"):
            flashinfer_trtllm_fp8_block_scale_moe(
                routing_logits=torch.zeros(1, 1),
                routing_bias=None,
                hidden_states=torch.zeros(1, 1, dtype=torch.float8_e4m3fn),
                hidden_states_scale=torch.ones(1, 1),
                gemm1_weights=torch.zeros(1, 1, dtype=torch.float8_e4m3fn),
                gemm1_weights_scale=torch.ones(1, 1),
                gemm2_weights=torch.zeros(1, 1, dtype=torch.float8_e4m3fn),
                gemm2_weights_scale=torch.ones(1, 1),
                num_experts=4,
                top_k=1,
                n_group=0,
                topk_group=0,
                intermediate_size=128,
                local_expert_offset=0,
                local_num_experts=4,
                routed_scaling_factor=1.0,
                routing_method_type=0,
                use_shuffled_weight=False,
            )

    def test_wrapper_forwards_to_impl_when_mocked(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When the underlying impl is available (mocked), the wrapper forwards call."""
        mock_return = torch.tensor([1.0, 2.0])
        mock_fn = Mock(return_value=mock_return)

        monkeypatch.setattr(flashinfer_mod, "has_flashinfer", lambda: True)
        monkeypatch.setattr(
            flashinfer_mod,
            "_get_submodule",
            lambda name: type(
                "FakeModule",
                (),
                {"trtllm_fp8_block_scale_moe": mock_fn},
            )()
            if name == "flashinfer.fused_moe"
            else None,
        )
        _clear_wrapper_cache()

        result = flashinfer_trtllm_fp8_block_scale_moe(
            routing_logits=torch.zeros(2, 4),
            routing_bias=None,
            hidden_states=torch.zeros(2, 128, dtype=torch.float8_e4m3fn),
            hidden_states_scale=torch.ones(1, 2),
            gemm1_weights=torch.zeros(4, 256, 128, dtype=torch.float8_e4m3fn),
            gemm1_weights_scale=torch.ones(4, 2),
            gemm2_weights=torch.zeros(4, 128, 256, dtype=torch.float8_e4m3fn),
            gemm2_weights_scale=torch.ones(4, 2),
            num_experts=4,
            top_k=1,
            n_group=0,
            topk_group=0,
            intermediate_size=256,
            local_expert_offset=0,
            local_num_experts=4,
            routed_scaling_factor=1.0,
            routing_method_type=0,
            use_shuffled_weight=False,
        )

        assert mock_fn.called
        assert result is mock_return


def _make_fp8_block_scale_moe_inputs(
    m: int,
    k: int,
    n: int,
    num_experts: int,
    top_k: int,
    device: str = "cuda",
    seed: int = 42,
):
    """Create FP8 block-scale MoE inputs with correct scale shapes (128x128 blocks)."""
    g = torch.Generator(device=device).manual_seed(seed)
    routing_logits = torch.randn(
        m, num_experts, device=device, dtype=torch.bfloat16, generator=g
    ) * 0.1
    hidden_states = (
        torch.randn(m, k, device=device, dtype=torch.float32, generator=g) * 0.1
    ).to(torch.float8_e4m3fn)
    # hidden_states_scale: (num_blocks_k, m) with block_n=128 along k
    num_blocks_k = (k + BLOCK_N - 1) // BLOCK_N
    hidden_states_scale = torch.ones(
        num_blocks_k, m, device=device, dtype=torch.float32
    )
    intermediate_size = n
    gemm1_weights = (
        torch.randn(
            num_experts,
            2 * intermediate_size,
            k,
            device=device,
            dtype=torch.float32,
            generator=g,
        )
        * 0.1
    ).to(torch.float8_e4m3fn)
    num_blocks_gemm1 = (
        2 * intermediate_size * k + BLOCK_M * BLOCK_N - 1
    ) // (BLOCK_M * BLOCK_N)
    gemm1_weights_scale = torch.ones(
        num_experts, num_blocks_gemm1, device=device, dtype=torch.float32
    )
    gemm2_weights = (
        torch.randn(
            num_experts, k, intermediate_size,
            device=device,
            dtype=torch.float32,
            generator=g,
        )
        * 0.1
    ).to(torch.float8_e4m3fn)
    num_blocks_gemm2 = (
        k * intermediate_size + BLOCK_M * BLOCK_N - 1
    ) // (BLOCK_M * BLOCK_N)
    gemm2_weights_scale = torch.ones(
        num_experts, num_blocks_gemm2, device=device, dtype=torch.float32
    )
    return (
        routing_logits,
        hidden_states,
        hidden_states_scale,
        gemm1_weights,
        gemm1_weights_scale,
        gemm2_weights,
        gemm2_weights_scale,
    )


@pytest.mark.skipif(
    not flashinfer_mod.has_flashinfer(),
    reason="FlashInfer is not available",
)
@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA is required for FP8 block-scale MoE",
)
class TestFlashinferTrtllmFp8BlockScaleMoeIntegration:
    """Integration tests when FlashInfer and CUDA are available."""

    @pytest.mark.parametrize("m,k,n", [(8, 256, 256), (16, 256, 512)])
    @pytest.mark.parametrize("num_experts,top_k", [(4, 1), (8, 2)])
    def test_wrapper_accepts_minimal_kwargs_sm100(
        self, m: int, k: int, n: int, num_experts: int, top_k: int
    ) -> None:
        from vllm.platforms import current_platform

        if not current_platform.has_device_capability(100):
            pytest.skip("FP8 block-scale MoE is only supported on sm >= 100")

        (
            routing_logits,
            hidden_states,
            hidden_states_scale,
            gemm1_weights,
            gemm1_weights_scale,
            gemm2_weights,
            gemm2_weights_scale,
        ) = _make_fp8_block_scale_moe_inputs(m, k, n, num_experts, top_k)

        with torch.inference_mode():
            out = flashinfer_trtllm_fp8_block_scale_moe(
                routing_logits=routing_logits,
                routing_bias=None,
                hidden_states=hidden_states,
                hidden_states_scale=hidden_states_scale,
                gemm1_weights=gemm1_weights,
                gemm1_weights_scale=gemm1_weights_scale,
                gemm2_weights=gemm2_weights,
                gemm2_weights_scale=gemm2_weights_scale,
                num_experts=num_experts,
                top_k=top_k,
                n_group=0,
                topk_group=0,
                intermediate_size=n,
                local_expert_offset=0,
                local_num_experts=num_experts,
                routed_scaling_factor=1.0,
                routing_method_type=0,
                use_shuffled_weight=False,
            )

        assert out.dtype == torch.float32

    @pytest.mark.parametrize("m,k,n", [(8, 256, 256), (16, 256, 512)])
    @pytest.mark.parametrize("num_experts,top_k", [(4, 1), (8, 2)])
    def test_flashinfer_trtllm_fp8_block_scale_moe_with_ones_routing_bias_sm100(
        self, m: int, k: int, n: int, num_experts: int, top_k: int
    ) -> None:
        from vllm.platforms import current_platform

        if not current_platform.has_device_capability(100):
            pytest.skip("FP8 block-scale MoE is only supported on sm >= 100")

        (
            routing_logits,
            hidden_states,
            hidden_states_scale,
            gemm1_weights,
            gemm1_weights_scale,
            gemm2_weights,
            gemm2_weights_scale,
        ) = _make_fp8_block_scale_moe_inputs(m, k, n, num_experts, top_k)

        routing_bias = torch.ones(
            num_experts, device="cuda", dtype=torch.bfloat16
        )
        with torch.inference_mode():
            out = flashinfer_trtllm_fp8_block_scale_moe(
                routing_logits=routing_logits,
                routing_bias=None,
                hidden_states=hidden_states,
                hidden_states_scale=hidden_states_scale,
                gemm1_weights=gemm1_weights,
                gemm1_weights_scale=gemm1_weights_scale,
                gemm2_weights=gemm2_weights,
                gemm2_weights_scale=gemm2_weights_scale,
                num_experts=num_experts,
                top_k=top_k,
                n_group=0,
                topk_group=0,
                intermediate_size=n,
                local_expert_offset=0,
                local_num_experts=num_experts,
                routed_scaling_factor=1.0,
                routing_method_type=0,
                use_shuffled_weight=False,
            )

        assert out.dtype == torch.float32

    @pytest.mark.parametrize("m,k,n", [(8, 256, 256), (16, 256, 512)])
    @pytest.mark.parametrize("num_experts,top_k", [(4, 1), (8, 2)])
    def test_flashinfer_trtllm_fp8_block_scale_moe_with_zeros_routing_bias_sm100(
        self, m: int, k: int, n: int, num_experts: int, top_k: int
    ) -> None:
        """Wrapper accepts optional routing_bias tensor (SM100)."""
        from vllm.platforms import current_platform

        if not current_platform.has_device_capability(100):
            pytest.skip("FP8 block-scale MoE is only supported on sm >= 100")

        (
            routing_logits,
            hidden_states,
            hidden_states_scale,
            gemm1_weights,
            gemm1_weights_scale,
            gemm2_weights,
            gemm2_weights_scale,
        ) = _make_fp8_block_scale_moe_inputs(m, k, n, num_experts, top_k)
        routing_bias = torch.zeros(
            num_experts, device="cuda", dtype=torch.bfloat16
        )

        with torch.inference_mode():
            out = flashinfer_trtllm_fp8_block_scale_moe(
                routing_logits=routing_logits,
                routing_bias=routing_bias,
                hidden_states=hidden_states,
                hidden_states_scale=hidden_states_scale,
                gemm1_weights=gemm1_weights,
                gemm1_weights_scale=gemm1_weights_scale,
                gemm2_weights=gemm2_weights,
                gemm2_weights_scale=gemm2_weights_scale,
                num_experts=num_experts,
                top_k=top_k,
                n_group=0,
                topk_group=0,
                intermediate_size=n,
                local_expert_offset=0,
                local_num_experts=num_experts,
                routed_scaling_factor=1.0,
                routing_method_type=0,
                use_shuffled_weight=False,
            )

        assert out.dtype == torch.float32

    @pytest.mark.parametrize("m,k,n", [(8, 256, 256), (16, 256, 512)])
    @pytest.mark.parametrize("num_experts,top_k", [(4, 1), (8, 2)])
    def test_flashinfer_trtllm_fp8_block_scale_moe_deterministic_same_input_sm100(
        self, m: int, k: int, n: int, num_experts: int, top_k: int
    ) -> None:
        """Same input produces same output (determinism) on SM100."""
        from vllm.platforms import current_platform

        if not current_platform.has_device_capability(100):
            pytest.skip("FP8 block-scale MoE is only supported on sm >= 100")

        (
            routing_logits,
            hidden_states,
            hidden_states_scale,
            gemm1_weights,
            gemm1_weights_scale,
            gemm2_weights,
            gemm2_weights_scale,
        ) = _make_fp8_block_scale_moe_inputs(
            m, k, n, num_experts, top_k, seed=123
        )

        with torch.inference_mode():
            out1 = flashinfer_trtllm_fp8_block_scale_moe(
                routing_logits=routing_logits,
                routing_bias=None,
                hidden_states=hidden_states,
                hidden_states_scale=hidden_states_scale,
                gemm1_weights=gemm1_weights,
                gemm1_weights_scale=gemm1_weights_scale,
                gemm2_weights=gemm2_weights,
                gemm2_weights_scale=gemm2_weights_scale,
                num_experts=num_experts,
                top_k=top_k,
                n_group=0,
                topk_group=0,
                intermediate_size=n,
                local_expert_offset=0,
                local_num_experts=num_experts,
                routed_scaling_factor=1.0,
                routing_method_type=0,
                use_shuffled_weight=False,
            )
            out2 = flashinfer_trtllm_fp8_block_scale_moe(
                routing_logits=routing_logits,
                routing_bias=None,
                hidden_states=hidden_states,
                hidden_states_scale=hidden_states_scale,
                gemm1_weights=gemm1_weights,
                gemm1_weights_scale=gemm1_weights_scale,
                gemm2_weights=gemm2_weights,
                gemm2_weights_scale=gemm2_weights_scale,
                num_experts=num_experts,
                top_k=top_k,
                n_group=0,
                topk_group=0,
                intermediate_size=n,
                local_expert_offset=0,
                local_num_experts=num_experts,
                routed_scaling_factor=1.0,
                routing_method_type=0,
                use_shuffled_weight=False,
            )

        torch.testing.assert_close(out1, out2)
