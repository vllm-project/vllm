# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the fused BMM + FP8 static quantization kernels (Triton & Helion)."""

import pytest
import torch

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
    ref_fp8 = (ref_bf16.float() * scale.item()).clamp(-448.0, 448.0).to(fp8_dtype)
    return ref_fp8


# ── Triton kernel tests ──────────────────────────────────────────────────────


@pytest.mark.skipif(not current_platform.is_cuda_alike(), reason="CUDA only")
@pytest.mark.skipif(not current_platform.supports_fp8(), reason="Need FP8")
@pytest.mark.parametrize("B", [1, 7, 32, 256])
@pytest.mark.parametrize("dtype", [torch.bfloat16])
def test_triton_bmm_fp8_quant_correctness(B, dtype):
    """Test that Triton fused BMM+FP8 matches the reference."""
    from vllm.v1.attention.ops.triton_bmm_fp8 import bmm_fp8_quant

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
    from vllm.v1.attention.ops.triton_bmm_fp8 import bmm_fp8_quant

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
    from vllm.v1.attention.ops.triton_bmm_fp8 import bmm_fp8_quant

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
