# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
import vllm._moe_C  # noqa: F401 — loads the _moe_C extension

from vllm.platforms import current_platform
from vllm.utils.torch_utils import set_random_seed

if not current_platform.has_device_capability(100):
    pytest.skip(
        reason="TMA persistent kernels require SM100+.",
        allow_module_level=True,
    )

FP8_DTYPE = current_platform.fp8_dtype()
E4M3_MAX = 448.0
GROUP_SIZE = 128


def quantize_fp8_per_group(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize a (N, D) float tensor to FP8 with per-128-group scales."""
    N, D = x.shape
    assert D % GROUP_SIZE == 0
    G = D // GROUP_SIZE
    x_grouped = x.reshape(N, G, GROUP_SIZE)
    amax = x_grouped.abs().amax(dim=-1).clamp(min=1e-12)
    scales = amax / E4M3_MAX
    inv_scales = 1.0 / scales
    x_scaled = x_grouped * inv_scales.unsqueeze(-1)
    x_fp8 = x_scaled.clamp(-E4M3_MAX, E4M3_MAX).to(FP8_DTYPE).reshape(N, D)
    return x_fp8, scales


def reference_silu_mul_fp8(
    input_fp8: torch.Tensor,
    gate_scales: torch.Tensor,
    up_scales: torch.Tensor,
    use_tanh_silu: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reference: dequant FP8 -> silu(up)*gate -> requant to FP8."""
    N = input_fp8.shape[0]
    H = input_fp8.shape[1] // 2
    G = H // GROUP_SIZE

    gate_fp8 = input_fp8[:, :H]
    up_fp8 = input_fp8[:, H:]

    gate_f32 = gate_fp8.to(torch.float32).reshape(N, G, GROUP_SIZE)
    up_f32 = up_fp8.to(torch.float32).reshape(N, G, GROUP_SIZE)
    gate_f32 = gate_f32 * gate_scales.unsqueeze(-1)
    up_f32 = up_f32 * up_scales.unsqueeze(-1)

    if use_tanh_silu:
        silu_up = up_f32 * (0.5 + 0.5 * torch.tanh(up_f32 * 0.5))
    else:
        silu_up = up_f32 * torch.sigmoid(up_f32)

    result = (silu_up * gate_f32).reshape(N, G, GROUP_SIZE)

    amax = result.abs().amax(dim=-1).clamp(min=1e-45)
    out_scales = amax / E4M3_MAX
    out_scales = out_scales.clamp(min=1e-45)
    inv_scales = 1.0 / out_scales
    result_scaled = result * inv_scales.unsqueeze(-1)
    result_fp8 = result_scaled.clamp(-E4M3_MAX, E4M3_MAX).to(FP8_DTYPE).reshape(N, H)

    return result_fp8, out_scales


def build_kernel_scale_tensor(
    gate_scales: torch.Tensor, up_scales: torch.Tensor
) -> torch.Tensor:
    """Build the column-major (2*G, N) scale tensor the kernel expects.

    Kernel indexes as: input_scales[tok + scale_stride * sb]
    where scale_stride = N, gate sb = 0..G-1, up sb = G..2G-1.
    A contiguous (2*G, N) tensor gives element [sb, tok] at offset sb*N + tok.
    """
    N, G = gate_scales.shape
    scales = torch.empty(2 * G, N, dtype=torch.float32, device=gate_scales.device)
    scales[:G] = gate_scales.t()
    scales[G:] = up_scales.t()
    return scales


CASES = [
    (128, 128, 1, 1, False),
    (256, 256, 1, 2, False),
    (1024, 7168, 7, 2, False),
    (1024, 7168, 7, 4, False),
    (1024, 7168, 7, 8, False),
    (4096, 7168, 7, 8, False),
    (1024, 7168, 7, 2, True),
    (4096, 7168, 7, 8, True),
]


@pytest.mark.parametrize("N,H,n_compute,batch_size,use_tanh_silu", CASES)
@torch.inference_mode()
def test_silu_mul_fp8_quant_tma_ws_persistent(
    N: int,
    H: int,
    n_compute: int,
    batch_size: int,
    use_tanh_silu: bool,
):
    set_random_seed(42)
    G = H // GROUP_SIZE

    bf16_data = torch.randn(N, 2 * H, dtype=torch.bfloat16, device="cuda")
    gate_bf16 = bf16_data[:, :H]
    up_bf16 = bf16_data[:, H:]

    gate_fp8, gate_scales = quantize_fp8_per_group(gate_bf16.float())
    up_fp8, up_scales = quantize_fp8_per_group(up_bf16.float())

    input_fp8 = torch.cat([gate_fp8, up_fp8], dim=1)
    input_scales = build_kernel_scale_tensor(gate_scales, up_scales)

    output = torch.empty(N, H, dtype=FP8_DTYPE, device="cuda")
    output_scales_buf = torch.empty(G, N, dtype=torch.float32, device="cuda")

    n_tokens = torch.tensor([N], dtype=torch.int32, device="cuda")

    torch.ops._moe_C.silu_mul_fp8_quant_tma_ws_persistent(
        input_fp8,
        input_scales,
        output,
        output_scales_buf,
        n_tokens,
        n_compute,
        batch_size,
        use_tanh_silu,
    )

    ref_fp8, ref_scales = reference_silu_mul_fp8(
        input_fp8, gate_scales, up_scales, use_tanh_silu
    )

    kernel_out_scales = output_scales_buf[:G, :N].t()

    torch.testing.assert_close(
        kernel_out_scales,
        ref_scales,
        atol=1e-5,
        rtol=1e-4,
    )

    output_f32 = output.to(torch.float32).reshape(N, G, GROUP_SIZE)
    output_dequant = output_f32 * kernel_out_scales.unsqueeze(-1)
    ref_f32 = ref_fp8.to(torch.float32).reshape(N, G, GROUP_SIZE)
    ref_dequant = ref_f32 * ref_scales.unsqueeze(-1)

    torch.testing.assert_close(
        output_dequant.reshape(N, H),
        ref_dequant.reshape(N, H),
        atol=0.5,
        rtol=0.05,
    )
