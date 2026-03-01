# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Test comparing Marlin INT4 MoE vs FlashInfer TRT-LLM MXINT4 MoE."""

import pytest
import torch
from test_cutlass_w4a8_moe import cutlass_quantize

from vllm import _custom_ops as ops
from vllm.platforms import current_platform
from vllm.scalar_type import scalar_types


@pytest.mark.skipif(
    not current_platform.is_cuda() or not current_platform.is_device_capability(90),
    reason="CUTLASS W4A16 MoE is only supported on SM90 devices.",
)
@pytest.mark.parametrize("bs", [1, 64, 128, 384])
@pytest.mark.parametrize("M", [128, 1024, 2048])
@pytest.mark.parametrize("N", [128, 1024, 2048, 7168])
@pytest.mark.parametrize("K", [2048, 4096, 7168, 16384])
@pytest.mark.parametrize("group_size", [64, 128])
@pytest.mark.parametrize("maybe_schedule", [None])
def test_cutlass_w4a16_moe_mm(
    bs: int,
    M: int,
    N: int,
    K: int,
    group_size: int,
    maybe_schedule: str | None,
):
    torch.random.manual_seed(42)
    device = torch.device("cuda")
    dtype = torch.bfloat16
    Ms = torch.randint(0, M, (bs,))
    print(f"{Ms=}")
    M_full = torch.sum(Ms).item()
    A = torch.randn(M_full, K, device=device, dtype=dtype)
    B = torch.randn(bs, K, N, device=device, dtype=dtype)

    B_ref_list, B_int4_list, B_scales_list = [], [], []
    for i in range(bs):
        B_ref_, B_int4_, B_scales_, _ = cutlass_quantize(
            torch.bfloat16,
            B[i],
            scalar_types.int4,
            torch.bfloat16,
            group_size,
            zero_points=False,
        )
        B_ref_list.append(B_ref_)
        B_int4_list.append(B_int4_.view(torch.int32))
        B_scales_list.append(B_scales_)
    B_ref = torch.stack(B_ref_list)
    B_int4 = torch.stack(B_int4_list)
    B_scales = torch.stack(B_scales_list)

    out_tensors = torch.empty(M_full, N, device=device, dtype=dtype)

    # swap AB
    problem_sizes = torch.zeros(bs, 3, device=device, dtype=torch.int32)
    problem_sizes[:, 0] = N
    problem_sizes[:, 2] = K
    for i in range(bs):
        problem_sizes[i, 1] = Ms[i]

    # Strides for memory layout
    # A strides: [K, 1] for row-major [M, K]
    a_strides = torch.full((bs,), K, device=device, dtype=torch.int64)

    B_int4_cutlass, b_strides = ops.cutlass_reorder_int4b_grouped(B_int4)

    # C strides: [N, 1] for row-major [M, N]
    c_strides = torch.full((bs,), N, device=device, dtype=torch.int64)

    # sizeof(StrideS) = 16 bytes, so we need to use 2xint64 to encode it
    group_scale_strides = torch.zeros((bs, 2), device=device, dtype=torch.int64)
    group_scale_strides[:, 0] = N

    offsets = torch.cat(
        [
            torch.tensor([0], dtype=torch.int64),
            torch.cumsum(Ms, dim=0)[:-1],
        ]
    ).to(device=device)

    # Call the kernel
    ops.cutlass_w4a16_moe_mm(
        out_tensors,
        A,
        B_int4_cutlass,
        B_scales.to(torch.bfloat16),
        group_size,
        offsets,
        problem_sizes,
        a_strides,
        b_strides,
        c_strides,
        group_scale_strides,
        maybe_schedule=maybe_schedule,
    )

    # ========reference implementation========
    out_tensors_ref = torch.empty(M_full, N, device=device, dtype=dtype)
    offset = 0
    for i in range(bs):
        out_tensors_ref[offset : offset + Ms[i]] = A[offset : offset + Ms[i]] @ B_ref[i]
        offset += Ms[i]
    # print(f"{out_tensors=}")
    # print(f"{out_tensors_ref=}")
    torch.testing.assert_close(out_tensors, out_tensors_ref, atol=5, rtol=1e-2)
