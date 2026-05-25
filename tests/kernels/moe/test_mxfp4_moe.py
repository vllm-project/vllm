# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Tests for SM100 CUTLASS MXFP4 x MXFP4 grouped MoE kernels."""

import random

import pytest
import torch

from tests.kernels.utils import torch_moe_single
from vllm import _custom_ops as ops
from vllm.platforms import current_platform
from vllm.utils.torch_utils import set_random_seed

random.seed(42)
set_random_seed(42)

MXFP4_BLOCK_SIZE = 32


def align(val: int, alignment: int = 128) -> int:
    return int((val + alignment - 1) // alignment * alignment)


def calc_diff(x, y):
    x, y = x.double(), y.double()
    denominator = (x * x + y * y).sum()
    sim = 2 * (x * y).sum() / denominator
    return 1 - sim


def is_sm100_supported() -> bool:
    return current_platform.is_cuda() and current_platform.is_device_capability_family(
        100
    )


def compute_ref_output(
    input_tensor: torch.Tensor,
    weight_list: list[torch.Tensor],
    expert_offsets: list[int],
    expert_offset: int,
    num_experts: int,
) -> torch.Tensor:
    """Reference output using torch_moe_single with top-1 routing."""
    score = torch.full(
        (expert_offset, num_experts),
        -1e9,
        device=input_tensor.device,
        dtype=torch.float32,
    )
    for g in range(num_experts):
        start = expert_offsets[g]
        end = expert_offsets[g + 1] if g + 1 < num_experts else expert_offset
        score[start:end, g] = 0.0

    return torch_moe_single(
        input_tensor, torch.stack(weight_list, dim=0), score, topk=1
    )


@pytest.mark.skipif(
    not is_sm100_supported(),
    reason="cutlass_mxfp4_group_mm requires CUDA SM100",
)
@pytest.mark.parametrize("num_experts", [8, 16, 32])
@pytest.mark.parametrize("out_dtype", [torch.bfloat16])
def test_cutlass_mxfp4_grouped_mm(num_experts, out_dtype):
    """
    Test the MXFP4 grouped GEMM kernel by:
    1. Creating random per-expert inputs and weights
    2. Quantizing both to MXFP4 using the CUDA kernel
    3. Running the CUTLASS grouped GEMM
    4. Comparing against BF16 reference
    """
    device = "cuda"
    alignment = 128
    # N and K must be multiples of 128 for clean swizzle layout
    n_g = random.randint(1, 16) * alignment
    k_g = random.randint(1, 16) * alignment

    expert_offset = 0
    expert_offsets_input = []
    problem_sizes = []
    input_list = []
    weight_list = []

    for g in range(num_experts):
        m_g = random.randint(1, 256)
        expert_offsets_input.append(expert_offset)
        expert_offset += m_g
        problem_sizes.append([m_g, n_g, k_g])

        input_list.append(
            torch.normal(0.0, std=0.5, size=(m_g, k_g), device=device, dtype=out_dtype)
        )
        weight_list.append(
            torch.normal(0.0, std=0.5, size=(n_g, k_g), device=device, dtype=out_dtype)
        )

    input_tensor = torch.concat(input_list, dim=0)  # [M_total, K]

    # --- Quantize INPUTS via mxfp4_experts_quant ---
    input_bs_offsets = []
    tot = 0
    for g in range(num_experts):
        input_bs_offsets.append(tot)
        tot += align(problem_sizes[g][0], 128)
    input_bs_offsets.append(tot)

    _inp_expert_offsets = torch.tensor(
        expert_offsets_input + [expert_offset], device=device, dtype=torch.int32
    )
    _inp_bs_offsets = torch.tensor(input_bs_offsets, device=device, dtype=torch.int32)

    input_quant, input_sf = ops.mxfp4_experts_quant(
        input_tensor,
        _inp_expert_offsets,
        _inp_bs_offsets,
        num_experts,
        topk=1,
    )

    # --- Quantize WEIGHTS via mxfp4_experts_quant ---
    # Treat each expert's N weight rows as an "expert" with N tokens
    weight_tensor = torch.concat(weight_list, dim=0)  # [E*N, K]
    weight_expert_offsets = [g * n_g for g in range(num_experts)] + [num_experts * n_g]
    # N is always multiple of 128, so blockscale offsets are clean
    weight_bs_offsets = [g * n_g for g in range(num_experts)] + [num_experts * n_g]

    _wt_expert_offsets = torch.tensor(
        weight_expert_offsets, device=device, dtype=torch.int32
    )
    _wt_bs_offsets = torch.tensor(weight_bs_offsets, device=device, dtype=torch.int32)

    weight_quant, weight_sf = ops.mxfp4_experts_quant(
        weight_tensor,
        _wt_expert_offsets,
        _wt_bs_offsets,
        num_experts,
        topk=1,
    )

    # Reshape weight quantized data to [E, N, K//2]
    weight_quant = weight_quant[: num_experts * n_g].view(num_experts, n_g, k_g // 2)

    # Reshape weight scale factors to [E, N, K//32]
    # The quant kernel produces uint8 SF buffer. Each row has K//32 SFs.
    scales_per_row = k_g // MXFP4_BLOCK_SIZE
    weight_sf_flat = weight_sf.view(-1)[: num_experts * n_g * scales_per_row]
    weight_sf_3d = weight_sf_flat.view(num_experts, n_g, scales_per_row)

    # Output
    output = torch.empty((expert_offset, n_g), device=device, dtype=out_dtype)

    _problem_sizes = torch.tensor(problem_sizes, device=device, dtype=torch.int32)
    _expert_offsets = torch.tensor(
        expert_offsets_input, device=device, dtype=torch.int32
    )
    _input_bs = torch.tensor(input_bs_offsets[:-1], device=device, dtype=torch.int32)

    # Run the MXFP4 grouped GEMM
    ops.cutlass_mxfp4_moe_mm(
        output,
        input_quant,
        weight_quant,
        input_sf,
        weight_sf_3d,
        _problem_sizes,
        _expert_offsets,
        _input_bs,
    )

    # Reference: BF16 matmul
    ref_output = compute_ref_output(
        input_tensor=input_tensor,
        weight_list=weight_list,
        expert_offsets=expert_offsets_input,
        expert_offset=expert_offset,
        num_experts=num_experts,
    )

    # Compare per-expert
    for g in range(num_experts):
        start = expert_offsets_input[g]
        end = expert_offsets_input[g + 1] if g + 1 < num_experts else expert_offset
        if start == end:
            continue
        baseline = ref_output[start:end]
        actual = output[start:end]
        diff = calc_diff(actual, baseline)
        print(
            f"m_g={end - start} n_g={n_g} k_g={k_g} "
            f"num_experts={num_experts}, "
            f"out_dtype={out_dtype}, diff={diff:.5f}"
        )
        # FP4 quantization is very lossy (~4 bits precision)
        # Comparing quantized vs full-precision gives cosine diff of 0.05-0.15
        assert diff < 0.15, f"Expert {g}: diff={diff:.5f} exceeds threshold"


@pytest.mark.skipif(
    not is_sm100_supported(),
    reason="mxfp4_experts_quant requires CUDA SM100",
)
def test_mxfp4_experts_quant_basic():
    """
    Basic smoke test for the MXFP4 experts quantization kernel.
    """
    device = "cuda"
    num_experts = 4
    k = 256
    tokens_per_expert = 16

    total_tokens = tokens_per_expert * num_experts
    input_tensor = torch.randn(total_tokens, k, device=device, dtype=torch.bfloat16) / 5

    expert_offsets = [i * tokens_per_expert for i in range(num_experts + 1)]
    blockscale_offsets = [
        align(i * tokens_per_expert, 128) for i in range(num_experts + 1)
    ]

    _expert_offsets = torch.tensor(expert_offsets, device=device, dtype=torch.int32)
    _blockscale_offsets = torch.tensor(
        blockscale_offsets, device=device, dtype=torch.int32
    )

    output, output_sf = ops.mxfp4_experts_quant(
        input_tensor,
        _expert_offsets,
        _blockscale_offsets,
        num_experts,
        topk=1,
    )

    assert output.shape == (total_tokens, k // 2)
    assert output.dtype == torch.uint8
    assert output_sf.dtype == torch.uint8
    assert output.any(), "Quantized output is all zeros"
    print(
        f"MXFP4 experts quant: output shape={output.shape}, sf shape={output_sf.shape}"
    )
    print("PASSED")


def untile_cutlass_scale(scale_raw: torch.Tensor, rows: int, K: int) -> torch.Tensor:
    """Convert CUTLASS tiled scale back to flat [M, K//32] layout.

    CUTLASS tiled layout: [numMTiles, numKTiles, 32(outerM), 4(innerM), 4(innerK)]
    Produced by: padded.reshape(numMTiles, 4, 32, numKTiles, 4).permute(0,3,2,1,4)
    To undo: tiled.permute(0, 3, 2, 1, 4).reshape(padded_M, padded_sK)
    """
    num_scale_cols = K // MXFP4_BLOCK_SIZE
    num_m_tiles = (rows + 127) // 128
    num_k_tiles = (num_scale_cols + 3) // 4
    padded_M = num_m_tiles * 128
    padded_sK = num_k_tiles * 4

    scale_bytes = scale_raw.view(torch.uint8).flatten()
    total_bytes = padded_M * padded_sK
    tiled = scale_bytes[:total_bytes].reshape(num_m_tiles, num_k_tiles, 32, 4, 4)
    undone = tiled.permute(0, 3, 2, 1, 4).contiguous()
    return undone.reshape(padded_M, padded_sK)[:rows, :num_scale_cols]


def compute_reference_e8m0_scale(block_max: float) -> int:
    """Compute the expected OCP MX spec E8M0 scale for a given block max.

    The CUTLASS kernel uses round-to-nearest on the mantissa:
      rounded_bits = (float_bits + (1 << 21)) & 0xFF800000
      biased_exp = (rounded_bits >> 23) & 0xFF
      scale_exp = max(biased_exp - 2, 0)

    This ensures max_val / scale <= 6.0 for most inputs.
    """
    import struct

    if block_max <= 0:
        return 0
    # Replicate the kernel's rounding logic in Python
    float_bytes = struct.pack("f", block_max)
    max_bits = struct.unpack("I", float_bytes)[0]
    rounded_bits = (max_bits + (1 << 21)) & 0xFF800000
    biased_exp = (rounded_bits >> 23) & 0xFF
    scale_exp = max(int(biased_exp) - 2, 0)
    scale_exp = min(scale_exp, 254)
    return scale_exp


@pytest.mark.skipif(
    not is_sm100_supported(),
    reason="mxfp4_experts_quant requires CUDA SM100",
)
@pytest.mark.parametrize("k", [256, 7168])
@pytest.mark.parametrize("m", [16, 64])
def test_mxfp4_experts_quant_e8m0_scale_correctness(m, k):
    """
    Test that mxfp4_experts_quant computes E8M0 block scales correctly
    per OCP MX spec (not the NVFP4 formula).

    The old buggy kernel used: floor(log2(max/6)) + 127
    The fixed kernel uses:     round_nearest_exp(max) - 2

    This test verifies:
    1. Scales match the expected OCP MX formula for all blocks
    2. No block max exceeds the representable range (no unexpected saturation)
    3. Reconstruction error is within expected bounds for MXFP4
    """
    device = "cuda"
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    # Generate input with controlled range
    input_tensor = torch.randn(m, k, device=device, dtype=torch.bfloat16) * 0.5

    # Quantize
    num_experts = 1
    expert_offsets = torch.tensor([0, m], device=device, dtype=torch.int32)
    num_k_tiles = (k // MXFP4_BLOCK_SIZE + 3) // 4
    blockscale_offsets = torch.tensor(
        [0, align(m, 128) * num_k_tiles], device=device, dtype=torch.int32
    )

    output_fp4, output_sf = ops.mxfp4_experts_quant(
        input_tensor, expert_offsets, blockscale_offsets, num_experts, topk=1
    )

    # Untile scale to flat layout for verification
    scale_flat = untile_cutlass_scale(output_sf, m, k)
    assert scale_flat.shape == (m, k // MXFP4_BLOCK_SIZE)

    # Verify each block's scale matches the OCP MX spec formula
    num_blocks = k // MXFP4_BLOCK_SIZE
    mismatches = 0
    buggy_pattern = 0  # count blocks where scale is 1-2 lower than expected

    for row in range(m):
        for blk in range(num_blocks):
            block_start = blk * MXFP4_BLOCK_SIZE
            block_end = block_start + MXFP4_BLOCK_SIZE
            block_max = input_tensor[row, block_start:block_end].float().abs().max().item()

            actual_scale = scale_flat[row, blk].item()
            expected_scale = compute_reference_e8m0_scale(block_max)

            if actual_scale != expected_scale:
                mismatches += 1
                if actual_scale < expected_scale:
                    buggy_pattern += 1

    total_blocks = m * num_blocks
    match_rate = (total_blocks - mismatches) / total_blocks

    print(f"  m={m}, k={k}: scale match rate = {match_rate*100:.2f}% "
          f"({mismatches}/{total_blocks} mismatches)")

    # The fixed kernel should match the reference formula exactly
    assert match_rate > 0.99, (
        f"E8M0 scale match rate too low: {match_rate*100:.2f}%. "
        f"Buggy pattern (scale too low): {buggy_pattern}/{mismatches}. "
        f"This suggests the NVFP4 formula bug is present."
    )

    # Extra check: if most mismatches show scale < expected, it's the old bug
    if mismatches > 0:
        assert buggy_pattern / mismatches < 0.5, (
            f"Most scale mismatches show scale too LOW ({buggy_pattern}/{mismatches}). "
            "This is the signature of the NVFP4 formula bug in nvfp4_utils.cuh."
        )

    # Verify reconstruction error is within MXFP4 expected bounds
    # Dequantize and check cosine similarity
    fp4_lut = torch.tensor(
        [0, 0.5, 1, 1.5, 2, 3, 4, 6, 0, -0.5, -1, -1.5, -2, -3, -4, -6],
        device=device, dtype=torch.float32,
    )
    lo = (output_fp4 & 0x0F).long()
    hi = ((output_fp4 >> 4) & 0x0F).long()
    unpacked = torch.stack([lo, hi], dim=-1).reshape(m, k)
    fp4_vals = fp4_lut[unpacked]

    scales_expanded = (2.0 ** (scale_flat.float() - 127.0))
    scales_expanded = scales_expanded.unsqueeze(-1).expand(-1, -1, MXFP4_BLOCK_SIZE)
    scales_expanded = scales_expanded.reshape(m, k)
    recon = (fp4_vals * scales_expanded).bfloat16()

    # Cosine similarity should be > 0.99 for well-behaved MXFP4 quantization
    cos_sim = torch.nn.functional.cosine_similarity(
        recon.float().flatten().unsqueeze(0),
        input_tensor.float().flatten().unsqueeze(0),
    ).item()
    max_abs_diff = (recon.float() - input_tensor.float()).abs().max().item()

    print(f"  Reconstruction: cosine_sim={cos_sim:.6f}, max_abs_diff={max_abs_diff:.4f}")

    assert cos_sim > 0.99, (
        f"Reconstruction cosine similarity too low: {cos_sim:.6f}. "
        f"Expected > 0.99 for correct MXFP4 quantization."
    )
    # With correct E8M0, max abs diff should be bounded by scale * 6
    # (worst case: value just below threshold rounds to wrong FP4 code)
    assert max_abs_diff < 1.0, (
        f"Max reconstruction error too large: {max_abs_diff:.4f}. "
        "Likely caused by incorrect E8M0 scale (values saturating to ±6)."
    )


@pytest.mark.skipif(
    not is_sm100_supported(),
    reason="mxfp4_experts_quant requires CUDA SM100",
)
def test_mxfp4_experts_quant_no_saturation():
    """
    Test that the E8M0 scale is large enough to avoid unexpected saturation.

    With the buggy NVFP4 formula, the scale was too small causing most values
    to saturate to ±6 in FP4. The fixed OCP MX formula should ensure that
    block_max / scale <= 6.0 (the max E2M1 value) in almost all cases.
    """
    device = "cuda"
    torch.manual_seed(123)
    torch.cuda.manual_seed(123)

    m, k = 128, 1024
    # Use inputs with known range to make saturation detectable
    input_tensor = torch.randn(m, k, device=device, dtype=torch.bfloat16) * 0.5

    num_experts = 1
    expert_offsets = torch.tensor([0, m], device=device, dtype=torch.int32)
    num_k_tiles = (k // MXFP4_BLOCK_SIZE + 3) // 4
    blockscale_offsets = torch.tensor(
        [0, align(m, 128) * num_k_tiles], device=device, dtype=torch.int32
    )

    output_fp4, output_sf = ops.mxfp4_experts_quant(
        input_tensor, expert_offsets, blockscale_offsets, num_experts, topk=1
    )

    # Check saturation rate: count FP4 values that are ±6 (codes 7 and 15)
    lo = output_fp4 & 0x0F
    hi = (output_fp4 >> 4) & 0x0F
    # Code 7 = +6.0, code 15 = -6.0
    saturated = ((lo == 7) | (lo == 15) | (hi == 7) | (hi == 15)).sum().item()
    total_values = m * k
    saturation_rate = saturated / total_values

    print(f"  Saturation rate: {saturation_rate*100:.2f}% "
          f"({saturated}/{total_values} values at ±6)")

    # For Gaussian input with std=0.5, saturation should be very rare
    # (±6 * scale is far from the typical range).
    # The buggy kernel had ~30-50% saturation; fixed should be < 5%.
    assert saturation_rate < 0.05, (
        f"FP4 saturation rate too high: {saturation_rate*100:.2f}%. "
        "This suggests the E8M0 scale is too small (NVFP4 formula bug). "
        "Expected < 5% for Gaussian(0, 0.5) input with correct OCP MX scale."
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
