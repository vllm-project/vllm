# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kernel correctness tests for the AITER NVFP4 MoE backend (gfx942).

Tests the Triton dot_scaled kernel (moe_gemm_a8w4_nvfp4) against a
dequant+matmul reference. Covers:
- GEMM1 (gather) and GEMM2 (no-gather / scatter) paths
- Multiple experts with uneven token counts
- K not a multiple of 128 (tail / mask path)
- Per-row activation scale correctness (catches the GEMM2 offset bug)
- FNUZ/OCP activation encoding (normal-range fp8 values)
"""

import pytest
import torch

from vllm.platforms import current_platform
from vllm.platforms.rocm import on_gfx942

pytestmark = [
    pytest.mark.skipif(not current_platform.is_rocm(), reason="ROCm-specific tests"),
    pytest.mark.skipif(
        not current_platform.is_rocm() or not on_gfx942(),
        reason="Requires gfx942 (CDNA3)",
    ),
]


# ---------------------------------------------------------------------------
# Reference helpers
# ---------------------------------------------------------------------------


def _pack_e2m1(w_float: torch.Tensor) -> torch.Tensor:
    """Pack a float weight tensor into NVFP4 e2m1 uint8 (2 values per byte).

    Uses a simple nearest-value LUT quantisation to the 16 e2m1 values
    (sign bit + 3-bit magnitude).  The low nibble holds the even-index
    element, the high nibble holds the odd-index element (K-major order).
    """
    # e2m1 representable magnitudes (positive)
    e2m1_pos = torch.tensor(
        [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0],
        dtype=torch.float32,
        device=w_float.device,
    )
    flat = w_float.float().reshape(-1)
    sign = (flat < 0).to(torch.uint8) << 3  # bit 3 = sign
    mag = flat.abs()
    # nearest e2m1 magnitude index
    dists = (mag[:, None] - e2m1_pos[None, :]).abs()
    idx = dists.argmin(dim=1).to(torch.uint8)  # 0..7
    nibbles = sign | idx  # 4-bit code per value

    # pack pairs: even index -> low nibble, odd index -> high nibble
    assert nibbles.numel() % 2 == 0
    nibbles = nibbles.view(-1, 2)
    packed = nibbles[:, 0] | (nibbles[:, 1] << 4)
    return packed.to(torch.uint8)


def _dequant_e2m1(packed: torch.Tensor, shape: tuple[int, ...]) -> torch.Tensor:
    """Unpack NVFP4 e2m1 uint8 -> float32 tensor of given shape."""
    e2m1_pos = torch.tensor(
        [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0],
        dtype=torch.float32,
        device=packed.device,
    )
    lo = packed & 0x0F
    hi = (packed >> 4) & 0x0F
    # interleave: lo is even index, hi is odd
    codes = torch.stack([lo, hi], dim=-1).reshape(-1)
    sign = ((codes >> 3) & 1).float() * -2.0 + 1.0
    mag_idx = (codes & 0x07).long()
    vals = sign * e2m1_pos[mag_idx]
    return vals.reshape(shape)


def _make_nvfp4_weights(
    E: int, N: int, K: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create random NVFP4 weights with per-16 fp8 scales and per-expert
    global scales.

    Returns (w_packed, w_scale, w_global_scale).
    w_packed: [E, K//2, N] uint8 (K-major packed e2m1)
    w_scale:  [E, N, K//16] fp8_e4m3fn
    w_global_scale: [E] float32
    """
    assert K % 16 == 0
    # Random float weights in a small range
    w_float = (torch.randn(E, N, K, device=device) * 2.0).clamp(-6.0, 6.0)
    # Give each 16-element block its own magnitude so per-16 scales differ
    # within every 32-element group (catches per-32 scale handling).
    block_mag = 2.0 ** torch.randint(-3, 3, (E, N, K // 16), device=device)
    w_float = w_float * block_mag.repeat_interleave(16, dim=-1)

    # Compute per-16 block scales
    w_blocks = w_float.reshape(E, N, K // 16, 16)
    block_max = w_blocks.abs().amax(dim=-1).clamp(min=1e-12)  # [E, N, K//16]
    # Scale so block values fit in e2m1 range [0, 6]
    block_scale = block_max / 6.0  # [E, N, K//16]

    # Global scale per expert
    global_scale = block_scale.amax(dim=(1, 2)).clamp(min=1e-12)  # [E]

    # Normalize: w_quant = w_float / (block_scale * global_scale)
    # but we need to be careful about the scale chain
    # The kernel computes: result = dot(X, W) * block_scale * global_scale
    # So we quantize: w_normalized = w / (block_scale_exp * gs_exp)
    block_scale_exp = block_scale.unsqueeze(-1).expand_as(w_blocks)
    global_scale_exp = global_scale[:, None, None, None].expand_as(w_blocks)
    w_normalized = w_blocks / (block_scale_exp * global_scale_exp + 1e-12)

    # Quantize to e2m1 and pack adjacent K pairs into one byte, then
    # transpose to the kernel's K-major [E, K//2, N] layout.
    w_normalized = w_normalized.reshape(E, N, K)
    w_packed_list = []
    for e in range(E):
        w_packed_list.append(_pack_e2m1(w_normalized[e]).reshape(N, K // 2).T)
    w_packed = torch.stack(w_packed_list).contiguous()  # [E, K//2, N]

    # Convert block scales to fp8
    w_scale_fp8 = block_scale.to(torch.float8_e4m3fn)  # [E, N, K//16]

    return w_packed, w_scale_fp8, global_scale


def _reference_gemm(
    x_fp8: torch.Tensor,
    x_scale: torch.Tensor,
    w_packed: torch.Tensor,
    w_scale: torch.Tensor,
    w_global_scale: torch.Tensor,
    expert_id: int,
) -> torch.Tensor:
    """Reference dequant + matmul for a single expert.

    Args:
        x_fp8: [M, K] fp8 activations
        x_scale: [M] per-row activation scales
        w_packed: [E, K//2, N] packed e2m1
        w_scale: [E, N, K//16] fp8 block scales
        w_global_scale: [E] float32 global scales (not the 0.5-corrected
            value passed to the kernel)
        expert_id: which expert

    Returns:
        [M, N] float32 result

    """
    M, K = x_fp8.shape
    N = w_packed.shape[2]

    # Dequant activations
    x_float = x_fp8.float() * x_scale[:, None]  # [M, K]

    # Dequant weights
    w_km = w_packed[expert_id]  # [K//2, N]
    w_float_km = _dequant_e2m1(w_km.T.contiguous(), (N, K)).T  # [K, N]

    # Apply block scales
    ws = w_scale[expert_id].float()  # [N, K//16]
    ws_exp = ws.unsqueeze(-1).expand(N, K // 16, 16).reshape(N, K)  # [N, K]
    w_scaled = w_float_km * ws_exp.T  # [K, N] element-wise

    w_scaled = w_scaled * w_global_scale[expert_id].item()

    # Matmul
    return x_float @ w_scaled  # [M, N]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def device():
    return torch.device("cuda:0")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def _skip_if_no_aiter():
    """Skip if AITER is not available."""
    try:
        from vllm._aiter_ops import is_aiter_found_and_supported

        if not is_aiter_found_and_supported():
            pytest.skip("AITER not available")
    except ImportError:
        pytest.skip("AITER not importable")


class TestNvfp4KernelCorrectness:
    """Test moe_gemm_a8w4_nvfp4 kernel correctness."""

    @pytest.mark.parametrize("K", [128, 256, 512])
    @pytest.mark.parametrize("N", [128, 256])
    @pytest.mark.parametrize("num_experts", [2, 4])
    def test_gemm1_gather_path(self, device, K, N, num_experts):
        """GEMM1 path: gather_indx is not None, simulating gate+up proj."""
        _skip_if_no_aiter()
        from aiter.ops.triton.moe.moe_routing.routing import routing

        from vllm.model_executor.layers.fused_moe.experts.aiter_nvfp4_moe import (
            _static_fp8_quant,
        )
        from vllm.model_executor.layers.fused_moe.experts.nvfp4_kernel import (
            moe_gemm_a8w4_nvfp4,
        )

        torch.manual_seed(42)
        M = 32
        topk = min(2, num_experts)
        x = torch.randn(M, K, device=device, dtype=torch.bfloat16)

        w_packed, w_scale, w_global_scale = _make_nvfp4_weights(
            num_experts, N, K, device
        )
        w_global_scale_corrected = w_global_scale * 0.5

        xq, x_static_scale = _static_fp8_quant(x)

        logits = torch.randn(M, num_experts, device=device, dtype=torch.float32)
        routing_data, gather_indx, scatter_indx = routing(logits, topk)

        result = moe_gemm_a8w4_nvfp4(
            xq,
            w_packed,
            w_scale,
            w_global_scale_corrected,
            x_static_scale=x_static_scale,
            routing_data=routing_data,
            gather_indx=gather_indx,
            out_dtype=torch.bfloat16,
        )

        assert result is not None
        assert result.shape[1] == N
        # Basic sanity: output should not be all zeros or NaN
        assert not torch.isnan(result).any(), "Output contains NaN"
        assert result.abs().sum() > 0, "Output is all zeros"

        # Row r of the sorted output is token gather_indx[r] // topk for the
        # expert given by the routing histogram.
        ref = torch.empty(result.shape, device=device)
        start = 0
        for e, n in enumerate(routing_data.expt_hist.tolist()):
            if n:
                tok = gather_indx[start : start + n].long() // topk
                ref[start : start + n] = _reference_gemm(
                    xq[tok], x_static_scale[tok], w_packed, w_scale, w_global_scale, e
                )
            start += n
        rel_err = (result.float() - ref).norm() / ref.norm()
        assert rel_err < 2e-2, f"GEMM1 gather path rel err {rel_err:.4f}"

    @pytest.mark.parametrize("K", [128, 256])
    @pytest.mark.parametrize("N", [128])
    @pytest.mark.parametrize("num_experts", [2, 4])
    def test_gemm2_scatter_path_scale_offset(self, device, K, N, num_experts):
        """GEMM2 path: gather_indx is None (scatter path).

        This specifically tests that per-row activation scales are correctly
        indexed when start_m != 0 (experts beyond the first). A bug where
        X_static_scale is not advanced by start_m would cause wrong scales
        for non-first experts.
        """
        _skip_if_no_aiter()
        from aiter.ops.triton.moe.moe_routing.routing import routing

        from vllm.model_executor.layers.fused_moe.experts.aiter_nvfp4_moe import (
            _static_fp8_quant,
        )
        from vllm.model_executor.layers.fused_moe.experts.nvfp4_kernel import (
            moe_gemm_a8w4_nvfp4,
        )

        torch.manual_seed(42)
        M = 64  # enough tokens to spread across experts
        topk = min(2, num_experts)

        # Create activations with distinct per-row scales to catch offset bugs
        x = torch.randn(M, K, device=device, dtype=torch.bfloat16)
        # Make rows have very different magnitudes
        row_mags = torch.logspace(-1, 2, M, device=device, dtype=torch.bfloat16)
        x = x * row_mags[:, None]

        w_packed, w_scale, w_global_scale = _make_nvfp4_weights(
            num_experts, N, K, device
        )
        w_global_scale_corrected = w_global_scale * 0.5

        # GEMM1 (with gather)
        logits = torch.randn(M, num_experts, device=device, dtype=torch.float32)
        routing_data, gather_indx, scatter_indx = routing(logits, topk)

        xq, x_static_scale = _static_fp8_quant(x)
        gate_up = moe_gemm_a8w4_nvfp4(
            xq,
            w_packed,
            w_scale,
            w_global_scale_corrected,
            x_static_scale=x_static_scale,
            routing_data=routing_data,
            gather_indx=gather_indx,
            out_dtype=torch.bfloat16,
        )

        # GEMM2 (no gather, with scatter) - this is the path that had the bug
        intermediate = gate_up  # pretend this is after activation
        iq, i_scale = _static_fp8_quant(intermediate)

        # GEMM2 input width is GEMM1's output width N; output width is K.
        w2_packed, w2_scale, w2_global_scale = _make_nvfp4_weights(
            num_experts, K, N, device
        )
        w2_global_scale_corrected = w2_global_scale * 0.5

        result = moe_gemm_a8w4_nvfp4(
            iq,
            w2_packed,
            w2_scale,
            w2_global_scale_corrected,
            x_static_scale=i_scale,
            routing_data=routing_data,
            scatter_indx=scatter_indx,
            out_dtype=torch.bfloat16,
        )

        assert result is not None
        assert not torch.isnan(result).any(), "GEMM2 output contains NaN"
        assert result.abs().sum() > 0, "GEMM2 output is all zeros"
        # Compare with a dequant reference: sorted row r belongs to the
        # expert given by the routing histogram, then rows are summed per
        # token over its topk slots.
        ref_rows = torch.empty(iq.shape[0], K, device=device)
        start = 0
        for e, n in enumerate(routing_data.expt_hist.tolist()):
            if n:
                ref_rows[start : start + n] = _reference_gemm(
                    iq[start : start + n],
                    i_scale[start : start + n],
                    w2_packed,
                    w2_scale,
                    w2_global_scale,
                    e,
                )
            start += n
        ref = ref_rows[scatter_indx.view(-1, topk).long()].sum(1)
        rel_err = (result.float() - ref).norm() / ref.norm()
        assert rel_err < 2e-2, f"GEMM2 scatter path rel err {rel_err:.4f}"

    @pytest.mark.parametrize("K", [192, 320])
    def test_k_not_multiple_of_128(self, device, K):
        """Test the tail/mask path when K is not a multiple of BLOCK_K=128."""
        _skip_if_no_aiter()
        from aiter.ops.triton.moe.moe_routing.routing import routing

        from vllm.model_executor.layers.fused_moe.experts.aiter_nvfp4_moe import (
            _static_fp8_quant,
        )
        from vllm.model_executor.layers.fused_moe.experts.nvfp4_kernel import (
            moe_gemm_a8w4_nvfp4,
        )

        torch.manual_seed(42)
        M, N, E = 16, 128, 2
        topk = 2
        # K must still be multiple of 16 (NVFP4 scale block)
        assert K % 16 == 0

        x = torch.randn(M, K, device=device, dtype=torch.bfloat16)
        w_packed, w_scale, w_global_scale = _make_nvfp4_weights(E, N, K, device)
        w_global_scale_corrected = w_global_scale * 0.5
        xq, x_static_scale = _static_fp8_quant(x)

        logits = torch.randn(M, E, device=device, dtype=torch.float32)
        routing_data, gather_indx, scatter_indx = routing(logits, topk)

        result = moe_gemm_a8w4_nvfp4(
            xq,
            w_packed,
            w_scale,
            w_global_scale_corrected,
            x_static_scale=x_static_scale,
            routing_data=routing_data,
            gather_indx=gather_indx,
            out_dtype=torch.bfloat16,
        )

        assert result is not None
        assert not torch.isnan(result).any(), "Tail-K output contains NaN"
        assert result.abs().sum() > 0, "Tail-K output is all zeros"

    def test_uneven_expert_tokens(self, device):
        """Test with deliberately uneven token distribution across experts."""
        _skip_if_no_aiter()
        from aiter.ops.triton.moe.moe_routing.routing import routing

        from vllm.model_executor.layers.fused_moe.experts.aiter_nvfp4_moe import (
            _static_fp8_quant,
        )
        from vllm.model_executor.layers.fused_moe.experts.nvfp4_kernel import (
            moe_gemm_a8w4_nvfp4,
        )

        torch.manual_seed(42)
        M, K, N, E = 48, 256, 128, 4
        topk = 2

        x = torch.randn(M, K, device=device, dtype=torch.bfloat16)
        w_packed, w_scale, w_global_scale = _make_nvfp4_weights(E, N, K, device)
        w_global_scale_corrected = w_global_scale * 0.5
        xq, x_static_scale = _static_fp8_quant(x)

        # Create biased logits so tokens cluster unevenly
        logits = torch.randn(M, E, device=device, dtype=torch.float32)
        logits[:, 0] += 5.0  # expert 0 gets most tokens
        logits[M - 3 :, 3] += 10.0  # expert 3 gets a few

        routing_data, gather_indx, scatter_indx = routing(logits, topk)

        result = moe_gemm_a8w4_nvfp4(
            xq,
            w_packed,
            w_scale,
            w_global_scale_corrected,
            x_static_scale=x_static_scale,
            routing_data=routing_data,
            gather_indx=gather_indx,
            out_dtype=torch.bfloat16,
        )

        assert result is not None
        assert not torch.isnan(result).any(), "Uneven output contains NaN"
        assert result.abs().sum() > 0, "Uneven output is all zeros"

    def test_fnuz_activation_encoding(self, device):
        """Verify that FNUZ-encoded fp8 activations produce correct results.

        On ROCm, _static_fp8_quant produces fp8_e4m3fnuz bytes. The kernel's
        tl.dot_scaled decodes them as OCP e4m3fn, which is exactly 2x the
        FNUZ value; the 0.5 global-scale correction compensates.
        """
        _skip_if_no_aiter()
        from aiter.ops.triton.moe.moe_routing.routing import routing

        from vllm.model_executor.layers.fused_moe.experts.aiter_nvfp4_moe import (
            _static_fp8_quant,
        )
        from vllm.model_executor.layers.fused_moe.experts.nvfp4_kernel import (
            moe_gemm_a8w4_nvfp4,
        )

        torch.manual_seed(42)
        M, K, N, E = 16, 128, 128, 2
        topk = 2

        # Use values that span a reasonable range (no subnormals in fp8)
        x = torch.randn(M, K, device=device, dtype=torch.bfloat16) * 10.0
        w_packed, w_scale, w_global_scale = _make_nvfp4_weights(E, N, K, device)
        w_global_scale_corrected = w_global_scale * 0.5

        xq, x_static_scale = _static_fp8_quant(x)

        # Verify quantized values are in normal fp8 range (not subnormal)
        xq_abs = xq.view(torch.uint8).float()
        # fp8 e4m3fnuz: exponent bits [6:3], subnormal when exponent == 0
        exponents = (xq_abs.to(torch.int32) >> 3) & 0x0F
        nonzero_mask = xq_abs > 0
        if nonzero_mask.any():
            subnormal_fraction = (exponents[nonzero_mask] == 0).float().mean().item()
            # Should be very few subnormals with our input range
            assert subnormal_fraction < 0.01, (
                f"Too many subnormal fp8 values: {subnormal_fraction:.2%}"
            )

        logits = torch.randn(M, E, device=device, dtype=torch.float32)
        routing_data, gather_indx, scatter_indx = routing(logits, topk)

        result = moe_gemm_a8w4_nvfp4(
            xq,
            w_packed,
            w_scale,
            w_global_scale_corrected,
            x_static_scale=x_static_scale,
            routing_data=routing_data,
            gather_indx=gather_indx,
            out_dtype=torch.bfloat16,
        )

        assert result is not None
        assert not torch.isnan(result).any(), "FNUZ path output contains NaN"
        assert result.abs().sum() > 0, "FNUZ path output is all zeros"


class TestAiterNvfp4Integration:
    """Integration tests for the full aiter_nvfp4_moe two-GEMM path."""

    def test_full_two_gemm_pipeline(self, device):
        """Test the full GEMM1 + activation + GEMM2 pipeline."""
        _skip_if_no_aiter()
        from aiter.ops.triton.moe.moe_routing.routing import routing

        from vllm.model_executor.layers.fused_moe.experts.aiter_nvfp4_moe import (
            _static_fp8_quant,
        )
        from vllm.model_executor.layers.fused_moe.experts.nvfp4_kernel import (
            moe_gemm_a8w4_nvfp4,
        )

        torch.manual_seed(42)
        M, K, N, E = 32, 256, 256, 4
        topk = 2
        down_N = 128  # down projection output dim

        x = torch.randn(M, K, device=device, dtype=torch.bfloat16)
        w13_packed, w13_scale, w13_gs = _make_nvfp4_weights(E, N, K, device)
        w2_packed, w2_scale, w2_gs = _make_nvfp4_weights(E, down_N, N // 2, device)

        logits = torch.randn(M, E, device=device, dtype=torch.float32)
        routing_data, gather_indx, scatter_indx = routing(logits, topk)

        # GEMM1
        xq, x_scale = _static_fp8_quant(x)
        gate_up = moe_gemm_a8w4_nvfp4(
            xq,
            w13_packed,
            w13_scale,
            w13_gs * 0.5,
            x_static_scale=x_scale,
            routing_data=routing_data,
            gather_indx=gather_indx,
            out_dtype=torch.bfloat16,
        )

        # SiLU activation (simplified - just take first half)
        half_n = gate_up.shape[1] // 2
        gate = gate_up[:, :half_n]
        up = gate_up[:, half_n:]
        intermediate = torch.nn.functional.silu(gate) * up

        # GEMM2
        iq, i_scale = _static_fp8_quant(intermediate)
        result = moe_gemm_a8w4_nvfp4(
            iq,
            w2_packed,
            w2_scale,
            w2_gs * 0.5,
            x_static_scale=i_scale,
            routing_data=routing_data,
            scatter_indx=scatter_indx,
            out_dtype=torch.bfloat16,
        )

        assert result is not None
        assert result.shape == (M, down_N)
        assert not torch.isnan(result).any(), "Pipeline output contains NaN"
        assert not torch.isinf(result).any(), "Pipeline output contains Inf"
