# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
import vllm._moe_C  # noqa: F401

from vllm.platforms import current_platform
from vllm.utils.torch_utils import set_random_seed

if not current_platform.has_device_capability(100):
    pytest.skip(
        reason="TMA persistent kernels require SM100+.",
        allow_module_level=True,
    )


def nvfp4_reference_silu_mul(
    input_bf16: torch.Tensor,
    global_scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reference: BF16 -> silu(gate)*up -> NVFP4 quantize.

    Returns dequantized output for comparison (since exact FP4
    bit patterns depend on rounding).
    """
    H = input_bf16.shape[1] // 2

    gate = input_bf16[:, :H].float()
    up = input_bf16[:, H:].float()
    result = torch.sigmoid(gate) * gate * up

    return result


def compute_sf_bytes(N: int, H: int) -> int:
    """Compute scale factor buffer size matching kernel's swizzled layout."""
    num_m_tiles = (N + 127) // 128
    num_k_tiles = (H + 63) // 64
    return num_m_tiles * num_k_tiles * 512


CASES = [
    (128, 7168, 7, 1, False),
    (256, 7168, 7, 2, False),
    (1024, 7168, 7, 2, False),
    (1024, 7168, 7, 4, False),
    (1024, 7168, 7, 8, False),
    (4096, 7168, 7, 8, False),
]


@pytest.mark.parametrize("N,H,n_compute,batch_size,use_tanh_silu", CASES)
@torch.inference_mode()
def test_nvfp4_persistent_vs_baseline(
    N: int,
    H: int,
    n_compute: int,
    batch_size: int,
    use_tanh_silu: bool,
):
    """Test persistent kernel matches baseline kernel output."""
    set_random_seed(42)

    input_bf16 = torch.randn(N, 2 * H, dtype=torch.bfloat16, device="cuda")
    global_scale = torch.ones(1, dtype=torch.float32, device="cuda")

    # Baseline outputs
    output_base = torch.empty(N, H // 2, dtype=torch.uint8, device="cuda")
    sf_bytes = compute_sf_bytes(N, H)
    sf_base = torch.zeros(sf_bytes, dtype=torch.uint8, device="cuda")
    mask = torch.tensor([N], dtype=torch.int32, device="cuda")

    torch.ops._moe_C.nvfp4_silu_mul_quant(
        output_base, sf_base, input_bf16, global_scale, mask, 1
    )

    # Persistent outputs
    output_persist = torch.empty(N, H // 2, dtype=torch.uint8, device="cuda")
    sf_persist = torch.zeros(sf_bytes, dtype=torch.uint8, device="cuda")
    n_tokens = torch.tensor([N], dtype=torch.int32, device="cuda")

    torch.ops._moe_C.silu_mul_nvfp4_quant_tma_ws_persistent_bf16(
        input_bf16,
        output_persist,
        sf_persist,
        global_scale,
        n_tokens,
        n_compute,
        batch_size,
        use_tanh_silu,
    )

    # Compare FP4 output bytes
    match_rate = (output_base == output_persist).float().mean().item()
    assert match_rate > 0.99, (
        f"FP4 output match rate {match_rate:.4f} < 0.99 "
        f"(N={N}, H={H}, nc={n_compute}, bs={batch_size})"
    )

    # Compare scale factors
    sf_match = (sf_base == sf_persist).float().mean().item()
    assert sf_match > 0.99, (
        f"Scale factor match rate {sf_match:.4f} < 0.99 "
        f"(N={N}, H={H}, nc={n_compute}, bs={batch_size})"
    )
