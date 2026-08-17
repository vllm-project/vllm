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
from vllm.model_executor.layers.fused_moe.activation import (
    MoEActivation,
    apply_moe_activation,
)
from vllm.model_executor.layers.fused_moe.experts.cutlass_moe import (
    run_cutlass_moe_w4a8_fp8,
)
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
    torch.accelerator.synchronize()

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


def make_batched_pipeline_weight(
    num_experts: int,
    out_features: int,
    in_features: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    w_qs = []
    w_ss = []
    for _ in range(num_experts):
        weight = to_fp8(torch.randn((in_features, out_features), device="cuda"))
        _, w_q, w_s, _ = cutlass_quantize(
            torch.float8_e4m3fn,
            weight.to(torch.float16),
            scalar_types.int4,
            torch.float8_e4m3fn,
            GROUP_SIZE,
        )
        w_qs.append(w_q)
        w_ss.append(w_s)

    weight_q, weight_scale, b_strides = cutlass_preprocess(w_qs, w_ss)
    channel_scale = (
        torch.rand(
            (num_experts, out_features, 1),
            dtype=torch.float32,
            device="cuda",
        )
        .mul_(0.05)
        .add_(0.001)
    )
    return weight_q, weight_scale, channel_scale, b_strides


@pytest.mark.skipif(
    not IS_SUPPORTED_BY_GPU,
    reason="W4A8 Grouped GEMM is not supported on this GPU type.",
)
def test_cutlass_w4a8_batched_pipeline_matches_flatten_reference():
    set_random_seed(42)
    num_experts = 4
    padded_m = 8
    hidden = 512
    intermediate = 256
    counts = [0, 8, 1, 5]
    device = torch.device("cuda")

    w1, w1_scale, w1_chan_scale, b_strides1 = make_batched_pipeline_weight(
        num_experts,
        intermediate * 2,
        hidden,
    )
    w2, w2_scale, w2_chan_scale, b_strides2 = make_batched_pipeline_weight(
        num_experts,
        hidden,
        intermediate,
    )
    expert_num_tokens = torch.tensor(counts, dtype=torch.int32, device=device)
    hidden_states = torch.randn(
        (num_experts, padded_m, hidden),
        dtype=torch.bfloat16,
        device=device,
    )
    a1q, a1q_scale = ops.scaled_fp8_quant(
        hidden_states.view(-1, hidden),
        use_per_token_if_dynamic=True,
    )
    a1q = a1q.view_as(hidden_states)
    a1q_scale = a1q_scale.view(num_experts, padded_m, 1)

    a_strides1 = torch.full((num_experts,), hidden, dtype=torch.int64, device=device)
    a_strides2 = torch.full(
        (num_experts,), intermediate, dtype=torch.int64, device=device
    )
    c_strides1 = torch.full(
        (num_experts,), intermediate * 2, dtype=torch.int64, device=device
    )
    c_strides2 = a_strides1
    s_strides1 = torch.zeros((num_experts, 2), dtype=torch.int64, device=device)
    s_strides1[:, 0] = intermediate * 2
    s_strides2 = torch.zeros_like(s_strides1)
    s_strides2[:, 0] = hidden

    workspace13 = torch.empty(
        (num_experts, padded_m, hidden),
        dtype=torch.bfloat16,
        device=device,
    )
    workspace2 = torch.empty_like(workspace13)
    output = torch.empty_like(workspace13)
    topk_ids = torch.zeros((1, 2), dtype=torch.int64, device=device)
    run_cutlass_moe_w4a8_fp8(
        output=output,
        hidden_states=a1q,
        w1=w1,
        w2=w2,
        topk_ids=topk_ids,
        activation=MoEActivation.SILU,
        global_num_experts=num_experts,
        expert_map=None,
        w1_scale=w1_scale,
        w2_scale=w2_scale,
        a1q_scale=a1q_scale,
        a2_scale=None,
        w1_chan_scale=w1_chan_scale,
        w2_chan_scale=w2_chan_scale,
        a_strides1=a_strides1,
        a_strides2=a_strides2,
        b_strides1=b_strides1,
        b_strides2=b_strides2,
        c_strides1=c_strides1,
        c_strides2=c_strides2,
        s_strides1=s_strides1,
        s_strides2=s_strides2,
        workspace13=workspace13,
        workspace2=workspace2,
        expert_num_tokens=expert_num_tokens,
        out_dtype=torch.bfloat16,
        per_act_token=True,
        per_out_ch=True,
        use_batched_format=True,
        topk_weights=None,
        group_size=GROUP_SIZE,
        permute_scratch=None,
    )

    expert_offsets = (
        torch.arange(num_experts, dtype=torch.int64, device=device) * padded_m
    )
    problem_sizes1 = torch.tensor(
        [[intermediate * 2, count, hidden] for count in counts],
        dtype=torch.int32,
        device=device,
    )
    problem_sizes2 = torch.tensor(
        [[hidden, count, intermediate] for count in counts],
        dtype=torch.int32,
        device=device,
    )
    mm1 = torch.empty(
        (num_experts * padded_m, intermediate * 2),
        dtype=torch.bfloat16,
        device=device,
    )
    ops.cutlass_w4a8_moe_mm(
        mm1,
        a1q.view(-1, hidden),
        w1,
        a1q_scale.view(-1, 1),
        w1_chan_scale,
        w1_scale,
        GROUP_SIZE,
        expert_offsets,
        problem_sizes1,
        a_strides1,
        b_strides1,
        c_strides1,
        s_strides1,
    )
    activation = torch.empty(
        (num_experts * padded_m, intermediate),
        dtype=torch.bfloat16,
        device=device,
    )
    apply_moe_activation(MoEActivation.SILU, activation, mm1)
    a2q, a2q_scale = ops.scaled_fp8_quant(
        activation,
        use_per_token_if_dynamic=True,
    )
    reference = torch.empty_like(output)
    ops.cutlass_w4a8_moe_mm(
        reference.view(-1, hidden),
        a2q,
        w2,
        a2q_scale,
        w2_chan_scale,
        w2_scale,
        GROUP_SIZE,
        expert_offsets,
        problem_sizes2,
        a_strides2,
        b_strides2,
        c_strides2,
        s_strides2,
    )

    for expert, count in enumerate(counts):
        torch.testing.assert_close(
            output[expert, :count],
            reference[expert, :count],
            rtol=1e-2,
            atol=3e-2,
        )
