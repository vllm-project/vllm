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


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
