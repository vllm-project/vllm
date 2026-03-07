# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Pytest tests for trtllm_fp4_block_scale_moe in vllm.utils.flashinfer."""

import pytest
import torch
from unittest.mock import Mock

import vllm.utils.flashinfer as flashinfer_mod
from vllm.utils.flashinfer import trtllm_fp4_block_scale_moe

BLOCK_SIZE = 16


def _clear_wrapper_cache() -> None:
    """Clear the lazy _get_impl cache so the next call re-resolves the backend."""
    for cell in trtllm_fp4_block_scale_moe.__closure__ or ():
        try:
            fn = cell.cell_contents
            if callable(fn) and hasattr(fn, "cache_clear"):
                fn.cache_clear()
                break
        except (ValueError, AttributeError):
            continue


class TestTrtllmFp4BlockScaleMoeWrapper:
    """Tests for the trtllm_fp4_block_scale_moe lazy wrapper."""

    def test_wrapper_is_callable(self) -> None:
        """The wrapper is a callable (function)."""
        assert callable(trtllm_fp4_block_scale_moe)

    def test_fallback_raises_when_flashinfer_unavailable(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When FlashInfer is not available, the wrapper raises RuntimeError."""
        monkeypatch.setattr(flashinfer_mod, "has_flashinfer", lambda: False)
        _clear_wrapper_cache()

        with pytest.raises(RuntimeError, match="FlashInfer backend is not available"):
            trtllm_fp4_block_scale_moe(
                routing_logits=torch.zeros(1, 1),
                routing_bias=None,
                hidden_states=torch.zeros(1, 1, dtype=torch.uint8),
                hidden_states_scale=torch.ones(1, dtype=torch.float32),
                gemm1_weights=torch.zeros(1, 1, dtype=torch.uint8),
                gemm1_weights_scale=torch.ones(1, dtype=torch.float32),
                gemm1_bias=None,
                gemm1_alpha=None,
                gemm1_beta=None,
                gemm1_clamp_limit=None,
                gemm2_weights=torch.zeros(1, 1, dtype=torch.uint8),
                gemm2_weights_scale=torch.ones(1, dtype=torch.float32),
                gemm2_bias=None,
                output1_scale_scalar=None,
                output1_scale_gate_scalar=None,
                output2_scale_scalar=None,
                num_experts=4,
                top_k=1,
                n_group=0,
                topk_group=0,
                intermediate_size=128,
                local_expert_offset=0,
                local_num_experts=4,
                routed_scaling_factor=None,
                routing_method_type=0,
                do_finalize=True,
            )

    def test_wrapper_forwards_to_impl_when_mocked(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When the underlying impl is available (mocked), the wrapper forwards call."""
        mock_return = (torch.tensor([1.0, 2.0]),)
        mock_fn = Mock(return_value=mock_return)

        monkeypatch.setattr(flashinfer_mod, "has_flashinfer", lambda: True)
        monkeypatch.setattr(
            flashinfer_mod,
            "_get_submodule",
            lambda name: type(
                "FakeModule",
                (),
                {"trtllm_fp4_block_scale_moe": mock_fn},
            )()
            if name == "flashinfer"
            else None,
        )
        _clear_wrapper_cache()

        result = trtllm_fp4_block_scale_moe(
            routing_logits=torch.zeros(2, 4),
            routing_bias=None,
            hidden_states=torch.zeros(2, 128, dtype=torch.uint8),
            hidden_states_scale=torch.ones(2, dtype=torch.float32),
            gemm1_weights=torch.zeros(4, 256, 128, dtype=torch.uint8),
            gemm1_weights_scale=torch.ones(4, dtype=torch.float32),
            gemm1_bias=None,
            gemm1_alpha=None,
            gemm1_beta=None,
            gemm1_clamp_limit=None,
            gemm2_weights=torch.zeros(4, 128, 256, dtype=torch.uint8),
            gemm2_weights_scale=torch.ones(4, dtype=torch.float32),
            gemm2_bias=None,
            output1_scale_scalar=None,
            output1_scale_gate_scalar=None,
            output2_scale_scalar=None,
            num_experts=4,
            top_k=1,
            n_group=0,
            topk_group=0,
            intermediate_size=256,
            local_expert_offset=0,
            local_num_experts=4,
            routed_scaling_factor=None,
            routing_method_type=0,
            do_finalize=True,
        )

        assert mock_fn.called
        assert result == mock_return


def _make_fp4_block_scale_moe_inputs(
    m: int,
    k: int,
    n: int,
    num_experts: int,
    top_k: int,
    device: str = "cuda",
    block_size: int = BLOCK_SIZE,
):
    """Create FP4 block-scale MoE inputs (uint8 fp4-packed, block_size=16)."""
    routing_logits = torch.randn(
        m, num_experts, device=device, dtype=torch.bfloat16
    ) * 0.1
    hidden_states = torch.zeros(m, k, device=device, dtype=torch.uint8)
    hidden_states_scale = torch.ones(
        m, k // block_size, device=device, dtype=torch.float32
    )
    # gemm1: (E, 2*n, k), gemm2: (E, k, n)
    gemm1_weights = torch.zeros(
        num_experts, 2 * n, k, device=device, dtype=torch.uint8
    )
    gemm1_weights_scale = torch.ones(
        num_experts,
        2 * n // block_size,
        k // block_size,
        device=device,
        dtype=torch.float32,
    )
    gemm2_weights = torch.zeros(
        num_experts, k, n, device=device, dtype=torch.uint8
    )
    gemm2_weights_scale = torch.ones(
        num_experts,
        k // block_size,
        n // block_size,
        device=device,
        dtype=torch.float32,
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


def _call_trtllm_fp4_block_scale_moe(
    routing_logits,
    hidden_states,
    hidden_states_scale,
    gemm1_weights,
    gemm1_weights_scale,
    gemm2_weights,
    gemm2_weights_scale,
    num_experts: int,
    top_k: int,
    n: int,
    routing_bias=None,
    **kwargs,
):
    """Call trtllm_fp4_block_scale_moe with common defaults; return output tensor."""
    result = trtllm_fp4_block_scale_moe(
        routing_logits=routing_logits,
        routing_bias=routing_bias,
        hidden_states=hidden_states,
        hidden_states_scale=hidden_states_scale,
        gemm1_weights=gemm1_weights,
        gemm1_weights_scale=gemm1_weights_scale,
        gemm1_bias=None,
        gemm1_alpha=None,
        gemm1_beta=None,
        gemm1_clamp_limit=None,
        gemm2_weights=gemm2_weights,
        gemm2_weights_scale=gemm2_weights_scale,
        gemm2_bias=None,
        output1_scale_scalar=None,
        output1_scale_gate_scalar=None,
        output2_scale_scalar=None,
        num_experts=num_experts,
        top_k=top_k,
        n_group=0,
        topk_group=0,
        intermediate_size=n,
        local_expert_offset=0,
        local_num_experts=num_experts,
        routed_scaling_factor=None,
        routing_method_type=0,
        do_finalize=True,
        **kwargs,
    )
    return result[0] if isinstance(result, tuple) else result


@pytest.mark.skipif(
    not flashinfer_mod.has_flashinfer(),
    reason="FlashInfer is not available",
)
@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA is required for FP4 block-scale MoE",
)
class TestTrtllmFp4BlockScaleMoeIntegration:
    """Integration tests when FlashInfer and CUDA are available."""

    @pytest.mark.parametrize("m,k,n", [(8, 128, 256), (16, 256, 256)])
    @pytest.mark.parametrize("num_experts,top_k", [(4, 1), (8, 2)])
    def test_wrapper_accepts_minimal_kwargs_sm100(
        self, m: int, k: int, n: int, num_experts: int, top_k: int
    ) -> None:
        from vllm.platforms import current_platform

        if not current_platform.has_device_capability(100):
            pytest.skip("FP4 block-scale MoE is only supported on sm >= 100")

        (
            routing_logits,
            hidden_states,
            hidden_states_scale,
            gemm1_weights,
            gemm1_weights_scale,
            gemm2_weights,
            gemm2_weights_scale,
        ) = _make_fp4_block_scale_moe_inputs(m, k, n, num_experts, top_k)

        with torch.inference_mode():
            out = _call_trtllm_fp4_block_scale_moe(
                routing_logits=routing_logits,
                hidden_states=hidden_states,
                hidden_states_scale=hidden_states_scale,
                gemm1_weights=gemm1_weights,
                gemm1_weights_scale=gemm1_weights_scale,
                gemm2_weights=gemm2_weights,
                gemm2_weights_scale=gemm2_weights_scale,
                num_experts=num_experts,
                top_k=top_k,
                n=n,
            )

        assert out.dtype in (torch.bfloat16, torch.float32)

    @pytest.mark.parametrize("m,k,n", [(8, 128, 256), (16, 256, 256)])
    @pytest.mark.parametrize("num_experts,top_k", [(4, 1), (8, 2)])
    def test_trtllm_fp4_block_scale_moe_with_ones_routing_bias_sm100(
        self, m: int, k: int, n: int, num_experts: int, top_k: int
    ) -> None:
        from vllm.platforms import current_platform

        if not current_platform.has_device_capability(100):
            pytest.skip("FP4 block-scale MoE is only supported on sm >= 100")

        (
            routing_logits,
            hidden_states,
            hidden_states_scale,
            gemm1_weights,
            gemm1_weights_scale,
            gemm2_weights,
            gemm2_weights_scale,
        ) = _make_fp4_block_scale_moe_inputs(m, k, n, num_experts, top_k)

        routing_bias = torch.ones(
            num_experts, device="cuda", dtype=torch.bfloat16
        )
        with torch.inference_mode():
            out = _call_trtllm_fp4_block_scale_moe(
                routing_logits=routing_logits,
                hidden_states=hidden_states,
                hidden_states_scale=hidden_states_scale,
                gemm1_weights=gemm1_weights,
                gemm1_weights_scale=gemm1_weights_scale,
                gemm2_weights=gemm2_weights,
                gemm2_weights_scale=gemm2_weights_scale,
                num_experts=num_experts,
                top_k=top_k,
                n=n,
            )

        assert out.dtype in (torch.bfloat16, torch.float32)

    @pytest.mark.parametrize("m,k,n", [(8, 128, 256), (16, 256, 256)])
    @pytest.mark.parametrize("num_experts,top_k", [(4, 1), (8, 2)])
    def test_trtllm_fp4_block_scale_moe_with_zeros_routing_bias_sm100(
        self, m: int, k: int, n: int, num_experts: int, top_k: int
    ) -> None:
        """Wrapper accepts optional routing_bias tensor (SM100)."""
        from vllm.platforms import current_platform

        if not current_platform.has_device_capability(100):
            pytest.skip("FP4 block-scale MoE is only supported on sm >= 100")

        (
            routing_logits,
            hidden_states,
            hidden_states_scale,
            gemm1_weights,
            gemm1_weights_scale,
            gemm2_weights,
            gemm2_weights_scale,
        ) = _make_fp4_block_scale_moe_inputs(m, k, n, num_experts, top_k)
        routing_bias = torch.zeros(
            num_experts, device="cuda", dtype=torch.bfloat16
        )

        with torch.inference_mode():
            out = _call_trtllm_fp4_block_scale_moe(
                routing_logits=routing_logits,
                hidden_states=hidden_states,
                hidden_states_scale=hidden_states_scale,
                gemm1_weights=gemm1_weights,
                gemm1_weights_scale=gemm1_weights_scale,
                gemm2_weights=gemm2_weights,
                gemm2_weights_scale=gemm2_weights_scale,
                num_experts=num_experts,
                top_k=top_k,
                n=n,
                routing_bias=routing_bias,
            )

        assert out.dtype in (torch.bfloat16, torch.float32)

    @pytest.mark.parametrize("m,k,n", [(8, 128, 256), (16, 256, 256)])
    @pytest.mark.parametrize("num_experts,top_k", [(4, 1), (8, 2)])
    def test_trtllm_fp4_block_scale_moe_deterministic_same_input_sm100(
        self, m: int, k: int, n: int, num_experts: int, top_k: int
    ) -> None:
        """Same input produces same output (determinism) on SM100."""
        from vllm.platforms import current_platform

        if not current_platform.has_device_capability(100):
            pytest.skip("FP4 block-scale MoE is only supported on sm >= 100")

        (
            routing_logits,
            hidden_states,
            hidden_states_scale,
            gemm1_weights,
            gemm1_weights_scale,
            gemm2_weights,
            gemm2_weights_scale,
        ) = _make_fp4_block_scale_moe_inputs(m, k, n, num_experts, top_k)

        with torch.inference_mode():
            out1 = _call_trtllm_fp4_block_scale_moe(
                routing_logits=routing_logits,
                hidden_states=hidden_states,
                hidden_states_scale=hidden_states_scale,
                gemm1_weights=gemm1_weights,
                gemm1_weights_scale=gemm1_weights_scale,
                gemm2_weights=gemm2_weights,
                gemm2_weights_scale=gemm2_weights_scale,
                num_experts=num_experts,
                top_k=top_k,
                n=n,
            )
            out2 = _call_trtllm_fp4_block_scale_moe(
                routing_logits=routing_logits,
                hidden_states=hidden_states,
                hidden_states_scale=hidden_states_scale,
                gemm1_weights=gemm1_weights,
                gemm1_weights_scale=gemm1_weights_scale,
                gemm2_weights=gemm2_weights,
                gemm2_weights_scale=gemm2_weights_scale,
                num_experts=num_experts,
                top_k=top_k,
                n=n,
            )

        torch.testing.assert_close(out1, out2)
