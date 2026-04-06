# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for fused BMM + FP8 quantization kernels (static & per-group)."""

import pytest
import torch

from vllm.model_executor.layers.quantization.utils.quant_utils import (
    get_fp8_min_max,
)
from vllm.platforms import current_platform
from vllm.utils.import_utils import has_helion

# DeepSeek-V3/R1 MLA dimensions
DEFAULT_N = 16  # num_heads
DEFAULT_L = 512  # kv_lora_rank
DEFAULT_V = 128  # v_head_dim


def _reference_bmm_fp8(input_tensor, weight, scale):
    """Reference implementation: torch.bmm then manual FP8 quantization."""
    fp8_dtype = current_platform.fp8_dtype()
    N, B, _L = input_tensor.shape
    V = weight.shape[2]

    ref_bmm = torch.bmm(input_tensor, weight)  # (N, B, V)
    ref_bf16 = ref_bmm.transpose(0, 1).reshape(B, N * V)
    _, fp8_max = get_fp8_min_max()
    ref_fp8 = (ref_bf16.float() * scale.item()).clamp(-fp8_max, fp8_max).to(fp8_dtype)
    return ref_fp8


# ── Triton kernel tests ──────────────────────────────────────────────────────


@pytest.mark.skipif(not current_platform.is_cuda_alike(), reason="CUDA only")
@pytest.mark.skipif(not current_platform.supports_fp8(), reason="Need FP8")
@pytest.mark.parametrize("B", [1, 7, 32, 256])
@pytest.mark.parametrize("dtype", [torch.bfloat16])
def test_triton_bmm_fp8_quant_correctness(B, dtype):
    """Test that Triton fused BMM+FP8 matches the reference."""
    from vllm.kernels.triton.ops.bmm_fp8_quant import bmm_fp8_quant

    N, L, V = DEFAULT_N, DEFAULT_L, DEFAULT_V
    device = torch.device("cuda:0")
    torch.manual_seed(42)

    input_tensor = torch.randn(N, B, L, dtype=dtype, device=device)
    weight = torch.randn(N, L, V, dtype=dtype, device=device)
    scale = torch.tensor([0.01], dtype=torch.float32, device=device)

    ref_fp8 = _reference_bmm_fp8(input_tensor, weight, scale)

    fp8_dtype = current_platform.fp8_dtype()
    fused_output = torch.empty(B, N * V, dtype=fp8_dtype, device=device)
    bmm_fp8_quant(input_tensor, weight, scale, fused_output)

    torch.testing.assert_close(
        fused_output.float(), ref_fp8.float(), atol=1.0, rtol=0.1
    )


@pytest.mark.skipif(not current_platform.is_cuda_alike(), reason="CUDA only")
@pytest.mark.skipif(not current_platform.supports_fp8(), reason="Need FP8")
def test_triton_bmm_fp8_quant_shapes():
    """Test various shapes work without errors (Triton)."""
    from vllm.kernels.triton.ops.bmm_fp8_quant import bmm_fp8_quant

    device = torch.device("cuda:0")
    fp8_dtype = current_platform.fp8_dtype()
    dtype = torch.bfloat16
    scale = torch.tensor([0.005], dtype=torch.float32, device=device)

    shapes = [
        (16, 1, 512, 128),  # single token
        (16, 128, 512, 128),  # medium batch
        (128, 1, 512, 128),  # many heads, single token
        (16, 1, 256, 64),  # smaller dims
    ]

    for N, B, L, V in shapes:
        inp = torch.randn(N, B, L, dtype=dtype, device=device)
        w = torch.randn(N, L, V, dtype=dtype, device=device)
        out = torch.empty(B, N * V, dtype=fp8_dtype, device=device)
        bmm_fp8_quant(inp, w, scale, out)
        assert out.shape == (B, N * V)
        assert out.dtype == fp8_dtype


# ── Helion kernel tests ──────────────────────────────────────────────────────


def _skip_if_helion_unavailable():
    """Skip test if Helion is not installed or configs are missing."""
    if not has_helion():
        pytest.skip("Helion not installed (pip install helion)")

    try:
        from vllm.kernels.helion.config_manager import ConfigManager
        from vllm.kernels.helion.utils import get_canonical_gpu_name

        platform = get_canonical_gpu_name()
        try:
            config_manager = ConfigManager.get_instance()
        except RuntimeError:
            config_manager = ConfigManager()

        configs = config_manager.get_platform_configs("bmm_fp8_quant_helion", platform)
        if len(configs) == 0:
            pytest.skip(
                f"No Helion configs for bmm_fp8_quant_helion on {platform}. "
                "Run: python scripts/autotune_helion_kernels.py "
                "--kernels bmm_fp8_quant_helion"
            )
    except (ImportError, RuntimeError, KeyError):
        pytest.skip("Error detecting Helion platform support")


@pytest.mark.skipif(not current_platform.is_cuda_alike(), reason="CUDA only")
@pytest.mark.skipif(not current_platform.supports_fp8(), reason="Need FP8")
@pytest.mark.parametrize("B", [1, 7, 32, 256])
@pytest.mark.parametrize("dtype", [torch.bfloat16])
def test_helion_bmm_fp8_quant_correctness(B, dtype):
    """Test that Helion fused BMM+FP8 matches the reference."""
    _skip_if_helion_unavailable()

    from vllm.kernels.helion.ops.bmm_fp8_quant import bmm_fp8_quant_helion

    N, L, V = DEFAULT_N, DEFAULT_L, DEFAULT_V
    device = torch.device("cuda:0")
    torch.manual_seed(42)

    input_tensor = torch.randn(N, B, L, dtype=dtype, device=device)
    weight = torch.randn(N, L, V, dtype=dtype, device=device)
    scale = torch.tensor([0.01], dtype=torch.float32, device=device)

    ref_fp8 = _reference_bmm_fp8(input_tensor, weight, scale)

    # Helion kernel returns (B, N, V), view as (B, N*V)
    helion_out = bmm_fp8_quant_helion(input_tensor, weight, scale)
    helion_flat = helion_out.reshape(B, N * V)

    torch.testing.assert_close(helion_flat.float(), ref_fp8.float(), atol=1.0, rtol=0.1)


@pytest.mark.skipif(not current_platform.is_cuda_alike(), reason="CUDA only")
@pytest.mark.skipif(not current_platform.supports_fp8(), reason="Need FP8")
def test_helion_bmm_fp8_quant_shapes():
    """Test various shapes work without errors (Helion)."""
    _skip_if_helion_unavailable()

    from vllm.kernels.helion.ops.bmm_fp8_quant import bmm_fp8_quant_helion

    device = torch.device("cuda:0")
    fp8_dtype = torch.float8_e4m3fn
    dtype = torch.bfloat16
    scale = torch.tensor([0.005], dtype=torch.float32, device=device)

    shapes = [
        (16, 1, 512, 128),
        (16, 128, 512, 128),
        (128, 1, 512, 128),
    ]

    for N, B, L, V in shapes:
        inp = torch.randn(N, B, L, dtype=dtype, device=device)
        w = torch.randn(N, L, V, dtype=dtype, device=device)
        out = bmm_fp8_quant_helion(inp, w, scale)
        assert out.reshape(B, N * V).shape == (B, N * V)
        assert out.dtype == fp8_dtype


# ── Cross-kernel consistency test ─────────────────────────────────────────────


@pytest.mark.skipif(not current_platform.is_cuda_alike(), reason="CUDA only")
@pytest.mark.skipif(not current_platform.supports_fp8(), reason="Need FP8")
@pytest.mark.parametrize("B", [1, 32, 128])
def test_triton_vs_helion_consistency(B):
    """Verify Triton and Helion kernels produce identical results."""
    _skip_if_helion_unavailable()

    from vllm.kernels.helion.ops.bmm_fp8_quant import bmm_fp8_quant_helion
    from vllm.kernels.triton.ops.bmm_fp8_quant import bmm_fp8_quant

    N, L, V = DEFAULT_N, DEFAULT_L, DEFAULT_V
    device = torch.device("cuda:0")
    fp8_dtype = current_platform.fp8_dtype()
    torch.manual_seed(42)

    input_tensor = torch.randn(N, B, L, dtype=torch.bfloat16, device=device)
    weight = torch.randn(N, L, V, dtype=torch.bfloat16, device=device)
    scale = torch.tensor([0.01], dtype=torch.float32, device=device)

    # Triton
    triton_out = torch.empty(B, N * V, dtype=fp8_dtype, device=device)
    bmm_fp8_quant(input_tensor, weight, scale, triton_out)

    # Helion
    helion_out = bmm_fp8_quant_helion(input_tensor, weight, scale)
    helion_flat = helion_out.reshape(B, N * V)

    torch.testing.assert_close(
        triton_out.float(), helion_flat.float(), atol=1.0, rtol=0.1
    )


# ── Per-group FP8 quantization tests ────────────────────────────────────────


def _reference_bmm_fp8_group(input_tensor, weight):
    """Reference: torch.bmm then per-group (per-head) dynamic FP8 quant.

    Uses fp32 BMM to match Triton kernel's fp32 accumulation, avoiding
    tf32 rounding differences on H100 that would cause scale mismatches.
    """
    fp8_dtype = current_platform.fp8_dtype()
    N, B, _L = input_tensor.shape
    V = weight.shape[2]

    # Use fp32 BMM to match Triton's fp32 accumulation
    ref_bmm = torch.bmm(input_tensor.float(), weight.float())  # (N, B, V)
    ref_bf16 = ref_bmm.transpose(0, 1)  # (B, N, V)

    _, fp8_max = get_fp8_min_max()

    # Per-group: one scale per (batch, head), computed from V elements
    abs_max = ref_bf16.float().abs().amax(dim=-1)  # (B, N)
    abs_max = abs_max.clamp(min=1e-12)
    scales = abs_max / fp8_max  # (B, N)
    inv_scales = fp8_max / abs_max  # (B, N)

    ref_scaled = (ref_bf16.float() * inv_scales.unsqueeze(-1)).clamp(-fp8_max, fp8_max)
    ref_fp8 = ref_scaled.reshape(B, N * V).to(fp8_dtype)
    return ref_fp8, scales


@pytest.mark.skipif(not current_platform.is_cuda_alike(), reason="CUDA only")
@pytest.mark.skipif(not current_platform.supports_fp8(), reason="Need FP8")
@pytest.mark.parametrize("B", [1, 7, 32, 256])
@pytest.mark.parametrize("dtype", [torch.bfloat16])
def test_triton_bmm_fp8_group_quant_correctness(B, dtype):
    """Test Triton fused BMM + per-group FP8 matches reference."""
    from vllm.kernels.triton.ops.bmm_fp8_quant import bmm_fp8_group_quant

    N, L, V = DEFAULT_N, DEFAULT_L, DEFAULT_V
    device = torch.device("cuda:0")
    torch.manual_seed(42)

    input_tensor = torch.randn(N, B, L, dtype=dtype, device=device)
    weight = torch.randn(N, L, V, dtype=dtype, device=device)

    ref_fp8, ref_scales = _reference_bmm_fp8_group(input_tensor, weight)

    fp8_dtype = current_platform.fp8_dtype()
    fused_output = torch.empty(B, N * V, dtype=fp8_dtype, device=device)
    fused_scales = torch.empty(B, N, dtype=torch.float32, device=device)
    bmm_fp8_group_quant(input_tensor, weight, fused_output, fused_scales)

    # Check scales match. Triton loads bf16 inputs into fp32 dot products,
    # which can differ slightly from pure fp32 torch.bmm, affecting abs_max.
    torch.testing.assert_close(fused_scales, ref_scales, atol=1e-3, rtol=1e-2)

    # Compare dequantized values rather than raw FP8.
    # FP8 e4m3fn has large gaps between representable values near the max
    # (up to 32), so raw FP8 comparison is unreliable. Dequantized values
    # (fp8 * scale) should be close to the original bf16 BMM result.
    fused_deq = fused_output.reshape(B, N, V).float() * fused_scales.unsqueeze(-1)
    ref_deq = ref_fp8.reshape(B, N, V).float() * ref_scales.unsqueeze(-1)
    torch.testing.assert_close(fused_deq, ref_deq, atol=4.0, rtol=0.12)


@pytest.mark.skipif(not current_platform.is_cuda_alike(), reason="CUDA only")
@pytest.mark.skipif(not current_platform.supports_fp8(), reason="Need FP8")
def test_triton_bmm_fp8_group_quant_shapes():
    """Test various shapes work without errors (per-group Triton)."""
    from vllm.kernels.triton.ops.bmm_fp8_quant import bmm_fp8_group_quant

    device = torch.device("cuda:0")
    fp8_dtype = current_platform.fp8_dtype()
    dtype = torch.bfloat16

    shapes = [
        (16, 1, 512, 128),
        (16, 128, 512, 128),
        (128, 1, 512, 128),
        (64, 32, 512, 128),
    ]

    for N, B, L, V in shapes:
        inp = torch.randn(N, B, L, dtype=dtype, device=device)
        w = torch.randn(N, L, V, dtype=dtype, device=device)
        out = torch.empty(B, N * V, dtype=fp8_dtype, device=device)
        scales = torch.empty(B, N, dtype=torch.float32, device=device)
        bmm_fp8_group_quant(inp, w, out, scales)
        assert out.shape == (B, N * V)
        assert out.dtype == fp8_dtype
        assert scales.shape == (B, N)
        assert scales.dtype == torch.float32
        # Scales should be positive
        assert (scales > 0).all()


# ── Helion per-group tests ──────────────────────────────────────────────────


def _skip_if_helion_group_unavailable():
    """Skip test if Helion is not installed or group-quant configs missing."""
    if not has_helion():
        pytest.skip("Helion not installed (pip install helion)")

    try:
        from vllm.kernels.helion.config_manager import ConfigManager
        from vllm.kernels.helion.utils import get_canonical_gpu_name

        platform = get_canonical_gpu_name()
        try:
            config_manager = ConfigManager.get_instance()
        except RuntimeError:
            config_manager = ConfigManager()

        configs = config_manager.get_platform_configs(
            "bmm_fp8_group_quant_helion", platform
        )
        if len(configs) == 0:
            pytest.skip(
                f"No Helion configs for bmm_fp8_group_quant_helion on "
                f"{platform}. Run: python scripts/autotune_helion_kernels.py "
                "--kernels bmm_fp8_group_quant_helion"
            )
    except (ImportError, RuntimeError, KeyError):
        pytest.skip("Error detecting Helion platform support")


@pytest.mark.skipif(not current_platform.is_cuda_alike(), reason="CUDA only")
@pytest.mark.skipif(not current_platform.supports_fp8(), reason="Need FP8")
@pytest.mark.parametrize("B", [1, 7, 32, 256])
@pytest.mark.parametrize("dtype", [torch.bfloat16])
def test_helion_bmm_fp8_group_quant_correctness(B, dtype):
    """Test Helion fused BMM + per-group FP8 matches reference."""
    _skip_if_helion_group_unavailable()

    from vllm.kernels.helion.ops.bmm_fp8_quant import (
        bmm_fp8_group_quant_helion,
    )

    N, L, V = DEFAULT_N, DEFAULT_L, DEFAULT_V
    device = torch.device("cuda:0")
    torch.manual_seed(42)

    input_tensor = torch.randn(N, B, L, dtype=dtype, device=device)
    weight = torch.randn(N, L, V, dtype=dtype, device=device)

    ref_fp8, ref_scales = _reference_bmm_fp8_group(input_tensor, weight)

    helion_out, helion_scales = bmm_fp8_group_quant_helion(input_tensor, weight)
    helion_flat = helion_out.reshape(B, N * V)

    # Compare scales (Helion also accumulates in fp32 but via different codegen)
    torch.testing.assert_close(helion_scales, ref_scales, atol=1e-3, rtol=1e-2)

    # Compare dequantized values (same reasoning as Triton per-group test)
    helion_deq = helion_flat.reshape(B, N, V).float() * helion_scales.unsqueeze(-1)
    ref_deq = ref_fp8.reshape(B, N, V).float() * ref_scales.unsqueeze(-1)
    torch.testing.assert_close(helion_deq, ref_deq, atol=4.0, rtol=0.12)


@pytest.mark.skipif(not current_platform.is_cuda_alike(), reason="CUDA only")
@pytest.mark.skipif(not current_platform.supports_fp8(), reason="Need FP8")
def test_helion_bmm_fp8_group_quant_shapes():
    """Test various shapes work without errors (per-group Helion)."""
    _skip_if_helion_group_unavailable()

    from vllm.kernels.helion.ops.bmm_fp8_quant import (
        bmm_fp8_group_quant_helion,
    )

    device = torch.device("cuda:0")
    fp8_dtype = torch.float8_e4m3fn
    dtype = torch.bfloat16

    shapes = [
        (16, 1, 512, 128),
        (16, 128, 512, 128),
        (128, 1, 512, 128),
    ]

    for N, B, L, V in shapes:
        inp = torch.randn(N, B, L, dtype=dtype, device=device)
        w = torch.randn(N, L, V, dtype=dtype, device=device)
        out, scales = bmm_fp8_group_quant_helion(inp, w)
        assert out.reshape(B, N * V).shape == (B, N * V)
        assert out.dtype == fp8_dtype
        assert scales.shape == (B, N)
        assert (scales > 0).all()


# ── Per-group cross-kernel consistency ──────────────────────────────────────


@pytest.mark.skipif(not current_platform.is_cuda_alike(), reason="CUDA only")
@pytest.mark.skipif(not current_platform.supports_fp8(), reason="Need FP8")
@pytest.mark.parametrize("B", [1, 32, 128])
def test_triton_vs_helion_group_quant_consistency(B):
    """Verify Triton and Helion per-group kernels produce identical results."""
    _skip_if_helion_group_unavailable()

    from vllm.kernels.helion.ops.bmm_fp8_quant import (
        bmm_fp8_group_quant_helion,
    )
    from vllm.kernels.triton.ops.bmm_fp8_quant import bmm_fp8_group_quant

    N, L, V = DEFAULT_N, DEFAULT_L, DEFAULT_V
    device = torch.device("cuda:0")
    fp8_dtype = current_platform.fp8_dtype()
    torch.manual_seed(42)

    input_tensor = torch.randn(N, B, L, dtype=torch.bfloat16, device=device)
    weight = torch.randn(N, L, V, dtype=torch.bfloat16, device=device)

    # Triton
    triton_out = torch.empty(B, N * V, dtype=fp8_dtype, device=device)
    triton_scales = torch.empty(B, N, dtype=torch.float32, device=device)
    bmm_fp8_group_quant(input_tensor, weight, triton_out, triton_scales)

    # Helion
    helion_out, helion_scales = bmm_fp8_group_quant_helion(input_tensor, weight)
    helion_flat = helion_out.reshape(B, N * V)

    # Compare scales
    torch.testing.assert_close(triton_scales, helion_scales, atol=1e-3, rtol=1e-2)

    # Compare dequantized values
    triton_deq = triton_out.reshape(B, N, V).float() * triton_scales.unsqueeze(-1)
    helion_deq = helion_flat.reshape(B, N, V).float() * helion_scales.unsqueeze(-1)
    torch.testing.assert_close(triton_deq, helion_deq, atol=4.0, rtol=0.12)
