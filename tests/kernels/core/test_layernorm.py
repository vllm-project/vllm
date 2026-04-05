# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from tests.kernels.quant_utils import FP8_DTYPE
from tests.kernels.utils import opcheck
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.platforms import current_platform
from vllm.utils.torch_utils import set_random_seed

_IS_GFX90A = False
if current_platform.is_rocm():
    from vllm.platforms.rocm import on_gfx90a

    _IS_GFX90A = on_gfx90a()

DTYPES = [torch.half, torch.bfloat16, torch.float]
NUM_TOKENS = [7, 83, 4096]  # Arbitrary values for testing
HIDDEN_SIZES = [8, 768, 769, 5120, 5125, 8192]  # Arbitrary values for testing
ADD_RESIDUAL = [False, True]
SEEDS = [0]
CUDA_DEVICES = [
    f"cuda:{i}" for i in range(1 if torch.accelerator.device_count() == 1 else 2)
]


@pytest.mark.parametrize("num_tokens", NUM_TOKENS)
@pytest.mark.parametrize("hidden_size", HIDDEN_SIZES)
@pytest.mark.parametrize("add_residual", ADD_RESIDUAL)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("device", CUDA_DEVICES)
@pytest.mark.parametrize("strided_input", [False, True])
@torch.inference_mode()
def test_rms_norm(
    default_vllm_config,
    num_tokens: int,
    hidden_size: int,
    add_residual: bool,
    dtype: torch.dtype,
    seed: int,
    device: str,
    strided_input: bool,
) -> None:
    set_random_seed(seed)
    torch.set_default_device(device)
    layer = RMSNorm(hidden_size).to(dtype=dtype)
    layer.weight.data.normal_(mean=1.0, std=0.1)
    scale = 1 / (2 * hidden_size)
    last_dim = 2 * hidden_size if strided_input else hidden_size
    x = torch.randn(num_tokens, last_dim, dtype=dtype)
    x = x[..., :hidden_size]
    assert x.is_contiguous() != strided_input
    x *= scale
    residual = torch.randn_like(x) * scale if add_residual else None

    # NOTE(woosuk): The reference implementation should be executed first
    # because the custom kernel is in-place.
    ref_out = layer.forward_native(x, residual)
    out = layer(x, residual)
    # NOTE(woosuk): LayerNorm operators (including RMS) typically have larger
    # numerical errors than other operators because they involve reductions.
    # Therefore, we use a larger tolerance.
    if add_residual:
        torch.testing.assert_close(out[0], ref_out[0], atol=1e-2, rtol=1e-2)
        torch.testing.assert_close(out[1], ref_out[1], atol=1e-2, rtol=1e-2)
    else:
        torch.testing.assert_close(out, ref_out, atol=1e-2, rtol=1e-2)

    if residual is not None:
        opcheck(
            torch.ops._C.fused_add_rms_norm,
            (x, residual, layer.weight.data, layer.variance_epsilon),
        )
    else:
        opcheck(
            torch.ops._C.rms_norm, (out, x, layer.weight.data, layer.variance_epsilon)
        )


@pytest.mark.parametrize("num_tokens", NUM_TOKENS)
@pytest.mark.parametrize("hidden_size", HIDDEN_SIZES)
@pytest.mark.parametrize("add_residual", ADD_RESIDUAL)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("quant_scale", [0.01, 1.0, 10.0])
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("device", CUDA_DEVICES)
@pytest.mark.parametrize("strided_input", [False, True])
def test_fused_rms_norm_quant(
    num_tokens: int,
    hidden_size: int,
    add_residual: bool,
    dtype: torch.dtype,
    quant_scale: float,
    seed: int,
    device: str,
    strided_input: bool,
) -> None:
    set_random_seed(seed)
    torch.set_default_device(device)

    weight = torch.empty(hidden_size, dtype=dtype).normal_(mean=1.0, std=0.1)
    scale = 1 / (2 * hidden_size)
    last_dim = 2 * hidden_size if strided_input else hidden_size
    x_base = torch.randn(num_tokens, last_dim, dtype=dtype)
    x = x_base[..., :hidden_size]
    assert x.is_contiguous() != strided_input

    x *= scale
    if add_residual:
        residual = torch.randn_like(x) * scale
        residual_fused = residual.clone()
    else:
        residual = residual_fused = None

    out_quant = torch.zeros_like(x, dtype=FP8_DTYPE)
    out_quant_fused = torch.zeros_like(out_quant)

    quant_scale_t = torch.tensor(quant_scale, dtype=torch.float32)

    if add_residual:
        torch.ops._C.fused_add_rms_norm_static_fp8_quant(
            out_quant_fused, x, residual_fused, weight, quant_scale_t, 1e-6
        )

        # Unfused kernel is in-place so it goes second
        # Also use a separate clone of x to avoid modifying the input
        x_unfused_base = x_base.clone()
        x_unfused = x_unfused_base[..., :hidden_size]
        assert x_unfused.is_contiguous() != strided_input
        torch.ops._C.fused_add_rms_norm(x_unfused, residual, weight, 1e-6)
        torch.ops._C.static_scaled_fp8_quant(
            out_quant, x_unfused.contiguous(), quant_scale_t
        )

        torch.accelerator.synchronize()
        torch.testing.assert_close(residual_fused, residual, atol=1e-2, rtol=1e-2)
        opcheck(
            torch.ops._C.fused_add_rms_norm_static_fp8_quant,
            (out_quant_fused, x, residual_fused, weight, quant_scale_t, 1e-6),
        )
    else:
        torch.ops._C.rms_norm_static_fp8_quant(
            out_quant_fused, x, weight, quant_scale_t, 1e-6
        )

        if _IS_GFX90A and dtype == torch.half:
            # On gfx90a the fused kernel computes the normalised value
            # entirely in float (hipcc -ffast-math elides half/bf16
            # round-trips). Build a float32 reference: compute
            # variance from the half inputs (same precision as the
            # kernel) then apply norm*weight in float32.
            x_f32 = x.float()
            variance = (x_f32**2).sum(dim=-1, keepdim=True)
            rms = torch.rsqrt(variance / hidden_size + 1e-6)
            ref_f32 = (x_f32 * rms * weight.float()).contiguous()
            torch.ops._C.static_scaled_fp8_quant(out_quant, ref_f32, quant_scale_t)
        else:
            out_norm = torch.zeros_like(x)
            torch.ops._C.rms_norm(out_norm, x, weight, 1e-6)
            torch.ops._C.static_scaled_fp8_quant(out_quant, out_norm, quant_scale_t)

        opcheck(
            torch.ops._C.rms_norm_static_fp8_quant,
            (out_quant_fused, x, weight, quant_scale_t, 1e-6),
        )

    if _IS_GFX90A and dtype == torch.half:
        # gfx90a fp8 is software-emulated; -ffast-math causes the
        # fused kernel to elide half round-trips that the unfused
        # reference forces through memory. The float32 reference
        # above eliminates >99.999% of mismatches, but the remaining
        # ~0.003% hit exact fp8 bin boundaries where a ~1-ULP
        # variance difference (CUB tree vs PyTorch reduction) shifts
        # the quantised value by exactly 1 fp8 step.
        diff = (out_quant.float() - out_quant_fused.float()).abs()
        n_total = diff.numel()
        n_exceed = int((diff > 1e-3).sum().item())
        pct_exceed = n_exceed / n_total * 100
        max_diff = diff.max().item()
        assert pct_exceed <= 0.01, (
            f"gfx90a fp8 quant: {pct_exceed:.4f}% elements exceed "
            f"atol=1e-3 (max allowed 0.01%)"
        )
        assert max_diff <= 9.0, (
            f"gfx90a fp8 quant: max_diff={max_diff} exceeds "
            f"relaxed limit 9.0 (1 fp8 step at highest magnitude)"
        )
    else:
        torch.testing.assert_close(
            out_quant.to(dtype=torch.float32),
            out_quant_fused.to(dtype=torch.float32),
            atol=1e-3,
            rtol=1e-3,
        )
