# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Tests for the CUTLASS-based W4A8 grouped GEMM kernel and the full MoE layer.
"""

import random
from dataclasses import dataclass

import pytest
import torch

from vllm import _custom_ops as ops
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    pack_rows,
    quantize_weights,
)
from vllm.platforms import current_platform
from vllm.scalar_type import ScalarType, scalar_types
from vllm.utils.torch_utils import set_random_seed

IS_SUPPORTED_BY_GPU = (
    current_platform.is_cuda() and current_platform.get_device_capability()[0] >= 9
)


def to_fp8(tensor: torch.Tensor) -> torch.Tensor:
    finfo = torch.finfo(torch.float8_e4m3fn)
    return tensor.clamp(min=finfo.min, max=finfo.max).to(dtype=torch.float8_e4m3fn)


def cutlass_quantize(
    atype: torch.dtype,
    w: torch.Tensor,
    wtype: ScalarType,
    stype: torch.dtype | None,
    group_size: int | None,
    zero_points: bool = False,
):
    """
    Quantize weights into W4 and compute reference dequantized weights.

    Encoding/reordering of weights and packing of scales is deferred
    until after all experts are combined.
    """
    assert wtype.is_integer(), "TODO: support floating point weights"

    w_ref, w_q, w_s, w_zp = quantize_weights(
        w, wtype, group_size=group_size, zero_points=zero_points
    )

    # Since scales are later cast to fp8, recompute w_ref in atype here.
    w_ref = (
        w_q.to(torch.float32)
        * w_s.to(atype).to(torch.float32).repeat_interleave(group_size, dim=0)
    ).to(atype)

    # Bit mask prevents sign extension of int4 when packing.
    w_q = pack_rows(w_q & 0x0F, wtype.size_bits, *w_q.shape)
    # Make weights row-major (N, K).
    w_q = w_q.t().contiguous()

    return w_ref, w_q, w_s.to(atype), w_zp


def cutlass_preprocess(
    w_q_experts: list[torch.Tensor], w_s_experts: list[torch.Tensor]
):
    """
    Reorder/encode expert weights and pack scales.

    Returns:
        w_q_packed: Packed/encoded int4 weights for all experts.
        w_s_packed: Packed fp8 scales for all experts.
        packed_layout: Layout/stride metadata for grouped GEMM.
    """
    w_s_packed = ops.cutlass_pack_scale_fp8(torch.stack(w_s_experts))
    w_q_packed, packed_layout = ops.cutlass_encode_and_reorder_int4b_grouped(
        torch.stack(w_q_experts)
    )  # expects dim 3
    return w_q_packed, w_s_packed, packed_layout


GROUP_SIZE = 128
# (num_experts, N, K)
TEST_SHAPES = [
    (8, 512, 2048),
    (8, 2048, 2048),
    (64, 512, 1024),
    (64, 2048, 2048),
    (4, 2048, 768),
    (8, 768, 2048),
    (64, 1536, 2048),
    (128, 8192, 4096),  # test overflow int32
]
ALIGNMENT = 16  # torch._scaled_mm alignment for M, needed for reference check


@dataclass
class MoETestSetup:
    num_experts: int
    K: int
    N: int
    Ms: list[int]
    M_full: int
    a: torch.Tensor
    a_ref: torch.Tensor
    a_strides: torch.Tensor
    out: torch.Tensor
    c_strides: torch.Tensor
    per_tok_scales: torch.Tensor
    per_chan_scales: torch.Tensor
    w_refs: list[torch.Tensor]
    w_q_packed: torch.Tensor
    w_s_packed: torch.Tensor
    problem_sizes: torch.Tensor
    expert_offsets: torch.Tensor
    b_strides: torch.Tensor
    group_scale_strides: torch.Tensor


def make_moe_test_setup(
    num_experts: int,
    K: int,
    N: int,
    *,
    alignment: int = ALIGNMENT,
    max_blocks: int = 64,
    device: str = "cuda",
    random_zero: bool = False,
) -> MoETestSetup:
    """Create a full set of tensors for testing cutlass_w4a8_moe_mm."""

    assert K % GROUP_SIZE == 0
    # Token counts per expert (multiples of `alignment`).
    Ms = [alignment * random.randint(1, max_blocks) for _ in range(num_experts)]

    # set random experts to 0 tokens
    if random_zero and num_experts > 1:
        num_zero = max(1, num_experts // 8)
        zero_indices = random.sample(range(num_experts), k=num_zero)
        for idx in zero_indices:
            Ms[idx] = 0

    M_full = sum(Ms)
    assert M_full > 0

    # Activations.
    a = to_fp8(torch.randn((M_full, K), device=device))
    a_ref = a.to(torch.float32)
    a_strides = torch.full((num_experts,), K, dtype=torch.int64, device=device)

    # Output buffer.
    out = torch.empty((M_full, N), dtype=torch.bfloat16, device=device)
    c_strides = torch.full((num_experts,), N, dtype=torch.int64, device=device)

    # Channel/token scales.
    per_tok_scales = torch.randn((M_full, 1), dtype=torch.float32, device=device)
    per_chan_scales = torch.randn(
        (num_experts, N, 1), dtype=torch.float32, device=device
    )

    # Expert weights and scales.
    wtype = scalar_types.int4
    atype = stype = torch.float8_e4m3fn
    w_refs, w_qs, w_ss = [], [], []
    for _ in range(num_experts):
        b = to_fp8(torch.randn((K, N), device=device))
        w_ref, w_q, w_s, _ = cutlass_quantize(
            atype, b.to(torch.float16), wtype, stype, GROUP_SIZE, zero_points=False
        )
        w_refs.append(w_ref)
        w_qs.append(w_q)
        w_ss.append(w_s)

    w_q_packed, w_s_packed, packed_layout = cutlass_preprocess(w_qs, w_ss)

    problem_sizes = torch.tensor(
        [[N, M, K] for M in Ms], dtype=torch.int32, device=device
    )

    expert_offsets = torch.cat(
        [
            torch.tensor([0], dtype=torch.int64),
            torch.cumsum(torch.tensor(Ms, dtype=torch.int64), dim=0)[:-1],
        ]
    ).to(device=device)

    # B strides and group scale strides.
    b_strides = packed_layout
    group_scale_strides = torch.zeros(
        (num_experts, 2), dtype=torch.int64, device=device
    )
    group_scale_strides[:, 0] = N

    return MoETestSetup(
        num_experts=num_experts,
        K=K,
        N=N,
        Ms=Ms,
        M_full=M_full,
        a=a,
        a_ref=a_ref,
        a_strides=a_strides,
        out=out,
        c_strides=c_strides,
        per_tok_scales=per_tok_scales,
        per_chan_scales=per_chan_scales,
        w_refs=w_refs,
        w_q_packed=w_q_packed,
        w_s_packed=w_s_packed,
        problem_sizes=problem_sizes,
        expert_offsets=expert_offsets,
        b_strides=b_strides,
        group_scale_strides=group_scale_strides,
    )


def compute_moe_reference_output(setup: MoETestSetup) -> torch.Tensor:
    """Compute reference output using torch._scaled_mm per expert."""
    out_ref = torch.empty_like(setup.out)

    ends = torch.cumsum(torch.tensor(setup.Ms), 0).tolist()
    starts = setup.expert_offsets.cpu().tolist()

    for i in range(setup.num_experts):
        start, end = starts[i], ends[i]
        if start == end:
            continue

        out_ref_i = torch._scaled_mm(
            setup.a_ref[start:end].to(torch.float8_e4m3fn),
            setup.w_refs[i].to(torch.float8_e4m3fn).t().contiguous().t(),
            setup.per_tok_scales[start:end],  # (M, 1)
            setup.per_chan_scales[i].reshape(1, -1),  # (1, N)
            out_dtype=torch.bfloat16,
            use_fast_accum=True,
        )
        out_ref[start:end] = out_ref_i

    return out_ref


@pytest.mark.skipif(
    not IS_SUPPORTED_BY_GPU,
    reason="W4A8 Grouped GEMM is not supported on this GPU type.",
)
@pytest.mark.parametrize("shape", TEST_SHAPES)
@pytest.mark.parametrize("random_zero", [True, False])
def test_cutlass_w4a8_moe_mm_end_to_end(shape, random_zero):
    num_experts, N, K = shape
    set_random_seed(42)
    setup = make_moe_test_setup(
        num_experts=num_experts, K=K, N=N, max_blocks=64, random_zero=random_zero
    )

    ops.cutlass_w4a8_moe_mm(
        setup.out,
        setup.a,
        setup.w_q_packed,
        setup.per_tok_scales,
        setup.per_chan_scales,
        setup.w_s_packed,
        GROUP_SIZE,
        setup.expert_offsets,
        setup.problem_sizes,
        setup.a_strides,
        setup.b_strides,
        setup.c_strides,
        setup.group_scale_strides,
    )
    torch.cuda.synchronize()

    out_ref = compute_moe_reference_output(setup)
    torch.testing.assert_close(setup.out, out_ref, rtol=1e-2, atol=1e-2)


class W4A8MoELayer(torch.nn.Module):
    """
    Minimal wrapper module to test cuda graphs
    """

    def __init__(self, setup: MoETestSetup):
        super().__init__()
        self.setup = setup

    def forward(self, a: torch.Tensor) -> torch.Tensor:
        s = self.setup
        ops.cutlass_w4a8_moe_mm(
            s.out,
            a,
            s.w_q_packed,
            s.per_tok_scales,
            s.per_chan_scales,
            s.w_s_packed,
            GROUP_SIZE,
            s.expert_offsets,
            s.problem_sizes,
            s.a_strides,
            s.b_strides,
            s.c_strides,
            s.group_scale_strides,
        )
        return s.out


@pytest.mark.skipif(
    not IS_SUPPORTED_BY_GPU,
    reason="W4A8 Grouped GEMM is not supported on this GPU type.",
)
def test_cutlass_w4a8_moe_mm_cuda_graph():
    set_random_seed(42)
    # Fixed config for CUDA graph test (single parameter point).
    num_experts = 8
    K = 512
    N = 2048

    setup = make_moe_test_setup(
        num_experts=num_experts,
        K=K,
        N=N,
        max_blocks=32,
    )

    # Construct model that calls the grouped GEMM kernel.
    model = W4A8MoELayer(setup)

    # Build reference output once.
    out_ref = compute_moe_reference_output(setup)

    # Capture and run the model in a CUDA graph.
    a_static = setup.a.clone()  # static input tensor for graph replay

    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            out_static = model(a_static)

    out_static.zero_()
    g.replay()

    torch.testing.assert_close(out_static, out_ref, rtol=1e-2, atol=1e-2)
