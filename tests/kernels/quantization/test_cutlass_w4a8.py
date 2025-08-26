# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the CUTLASS W4A8 kernel.

Run `pytest tests/kernels/test_cutlass_w4a8.py`.
"""

from dataclasses import dataclass
from typing import Optional

import pytest
import torch

from vllm import _custom_ops as ops
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    pack_rows, quantize_weights)
from vllm.platforms import current_platform
from vllm.scalar_type import ScalarType, scalar_types

# TODO: in future PR refactor this and `is_quant_method_supported` in the kernel
#  unit tests to a common utility function. Currently the use of
#  `is_quant_method_supported` conflates kernels with quantization methods
#  an assumption which is breaking down as quantizations methods can have
#  have kernels and some kernels support multiple quantization methods.
IS_SUPPORTED_BY_GPU = current_platform.get_device_capability()[0] >= 9

MNK_SHAPES = [(1, 128, 128), (1, 512, 1024), (1, 4096, 4096), (1, 8192, 28672),
              (13, 8192, 4096), (26, 4096, 8192), (64, 4096, 4096),
              (64, 8192, 28672), (257, 128, 4096), (257, 4096, 4096),
              (1024, 4096, 8192), (1024, 8192, 4096)]

# TODO(czhu): get supported schedules from fn
SCHEDULES = [
    '128x16_1x1x1', '256x16_1x1x1', '128x32_1x1x1', '256x32_1x1x1',
    '128x64_1x1x1', '256x64_1x1x1', '128x128_1x1x1', '256x128_1x1x1',
    '128x256_1x1x1', '128x256_2x1x1'
]


@dataclass
class TypeConfig:
    act_type: torch.dtype
    weight_type: ScalarType
    output_type: Optional[torch.dtype]
    group_scale_type: Optional[torch.dtype]
    channel_scale_type: Optional[torch.dtype]
    token_scale_type: Optional[torch.dtype]


@dataclass
class Tensors:
    w_ref: torch.Tensor
    a_ref: torch.Tensor
    a: torch.Tensor
    w_q: torch.Tensor
    w_g_s: torch.Tensor
    w_ch_s: torch.Tensor
    w_tok_s: torch.Tensor


# (Act Type, Weight Type, Output Type, Scale Type, ZeroPoints,
#  Ch Scales Type, Tok Scales Type)
TestTypeTuple = tuple[list[torch.dtype], ScalarType, Optional[torch.dtype],
                      Optional[torch.dtype], bool]
TEST_TYPES = [
    *(
        TypeConfig(act_type=torch.float8_e4m3fn,
                   weight_type=w_type,
                   output_type=o_type,
                   group_scale_type=torch.float8_e4m3fn,
                   channel_scale_type=torch.float32,
                   token_scale_type=torch.float32)
        for w_type in [scalar_types.int4]
        # TODO(czhu): fp16 out type
        for o_type in [torch.bfloat16]),
]

# TODO: in future PR refactor this and `is_quant_method_supported` in the kernel
#  unit tests to a common utility function. Currently the use of
#  `is_quant_method_supported` conflates kernels with quantization methods
#  an assumption which is breaking down as quantizations methods can have
#  have kernels and some kernels support multiple quantization methods.
IS_SUPPORTED_BY_GPU = current_platform.has_device_capability(90)


# For testing quantized linear kernels
def to_fp8(tensor: torch.Tensor):
    finfo = torch.finfo(torch.float8_e4m3fn)
    return tensor.clamp(min=finfo.min,
                        max=finfo.max).to(dtype=torch.float8_e4m3fn)


def cutlass_quantize_and_pack(atype: torch.dtype,
                              w: torch.Tensor,
                              wtype: ScalarType,
                              stype: Optional[torch.dtype],
                              group_size: Optional[int],
                              zero_points: bool = False):
    assert wtype.is_integer(), "TODO: support floating point weights"

    w_ref, w_q, w_s, w_zp = quantize_weights(w,
                                             wtype,
                                             group_size=group_size,
                                             zero_points=zero_points)

    # since scales are cast to fp8, we need to compute w_ref this way
    w_ref = ((w_q).to(torch.float32) * w_s.to(atype).to(
        torch.float32).repeat_interleave(group_size, dim=0)).to(atype)

    # bit mask prevents sign extending int4 when packing
    w_q = pack_rows(w_q & 0x0F, wtype.size_bits, *w_q.shape)
    w_q = w_q.t().contiguous().t()  # convert to col major

    w_q_packed = ops.cutlass_encode_and_reorder_int4b(w_q)
    w_s_packed = ops.cutlass_pack_scale_fp8(w_s.to(atype))

    return w_ref, w_q_packed, w_s_packed, w_zp


def create_test_tensors(shape: tuple[int, int, int], types: TypeConfig,
                        group_size: Optional[int]) -> Tensors:
    m, n, k = shape

    print("create_test_tensors, shape:", shape, "types:", types, "group_size:",
          group_size)

    a = to_fp8(torch.randn((m, k), device="cuda"))
    w = to_fp8(torch.randn((k, n), device="cuda"))

    if types.group_scale_type is not None:
        w = w.to(types.group_scale_type)
    if w.dtype.itemsize == 1:
        w = w.to(torch.float16)

    w_ref, w_q_packed, w_s, _ = cutlass_quantize_and_pack(
        a.dtype, w, types.weight_type, types.group_scale_type, group_size,
        False)

    a_ref = a.to(torch.float32)
    w_ref = w_ref.to(torch.float32)

    # for the practical use case we need per-tok scales for fp8 activations
    w_tok_s = torch.randn((m, ), device='cuda', dtype=types.token_scale_type)
    # weights are already per-group quantized, use placeholder here
    w_ch_s = torch.ones((n, ), device='cuda', dtype=types.channel_scale_type)

    return Tensors(w_ref=w_ref,
                   a_ref=a_ref,
                   a=a,
                   w_q=w_q_packed,
                   w_g_s=w_s,
                   w_ch_s=w_ch_s,
                   w_tok_s=w_tok_s)


def mm_test_helper(types: TypeConfig,
                   tensors: Tensors,
                   group_size: Optional[int] = None,
                   schedule: Optional[str] = None):
    # CUTLASS upstream uses fp8 with fastaccum as reference
    # https://github.com/NVIDIA/cutlass/blob/main/examples/55_hopper_mixed_dtype_gemm/55_hopper_int4_fp8_gemm.cu#L406
    output_ref = torch._scaled_mm(
        tensors.a_ref.to(types.act_type),
        tensors.w_ref.to(types.act_type).t().contiguous().t(),  # col major
        tensors.w_tok_s.unsqueeze(1),
        tensors.w_ch_s.unsqueeze(0),
        out_dtype=types.output_type,
        use_fast_accum=True)

    output = ops.cutlass_w4a8_mm(
        a=tensors.a,
        b_q=tensors.w_q,
        b_group_scales=tensors.w_g_s,
        b_group_size=group_size,
        b_channel_scales=tensors.w_ch_s,
        a_token_scales=tensors.w_tok_s,
    )

    print(output)
    print(output_ref)

    torch.testing.assert_close(output,
                               output_ref.to(output.dtype),
                               rtol=1e-3,
                               atol=1e-3)


@pytest.mark.skipif(not IS_SUPPORTED_BY_GPU,
                    reason="CUTLASS W4A8 is not supported on this GPU type.")
@pytest.mark.parametrize("shape",
                         MNK_SHAPES,
                         ids=lambda x: "x".join(str(v) for v in x))
@pytest.mark.parametrize("types", TEST_TYPES)
@pytest.mark.parametrize("schedule", SCHEDULES)
def test_cutlass_w4a8(shape, types: TypeConfig, schedule):
    group_sizes = [128]
    for group_size in group_sizes:
        tensors = create_test_tensors(shape, types, group_size)
        mm_test_helper(types, tensors, group_size, schedule)


# Test to make sure cuda graphs work
class W4A8Layer(torch.nn.Module):

    def __init__(self, **kwargs):
        super().__init__()
        self.kwargs = kwargs

    def forward(self, a):
        return ops.cutlass_w4a8_mm(a=a, **self.kwargs)


@pytest.mark.skipif(not IS_SUPPORTED_BY_GPU,
                    reason="CUTLASS W4A8 is not supported on this GPU type.")
def test_w4a8_cuda_graph():
    m, n, k = 512, 4096, 4096

    a = to_fp8(torch.randn((m, k), device="cuda"))
    b = to_fp8(torch.randn((k, n), device="cuda"))

    wtype = scalar_types.int4
    stype = torch.float8_e4m3fn
    group_size = 128
    zero_points = False

    w_ref, w_q_packed, w_s, _ = cutlass_quantize_and_pack(
        a.dtype, b.to(torch.float16), wtype, stype, group_size, zero_points)

    w_tok_s = torch.randn((m, ), device='cuda', dtype=torch.float32)
    w_ch_s = torch.ones((n, ), device='cuda', dtype=torch.float32)

    # Construct a trivial model with a single layer that calls the kernel
    model = W4A8Layer(
        b_q=w_q_packed,
        b_group_scales=w_s,
        b_group_size=group_size,
        b_channel_scales=w_ch_s,
        a_token_scales=w_tok_s,
    )

    output_ref = torch._scaled_mm(
        a,
        w_ref.to(a.dtype).t().contiguous().t(),  # col major
        w_tok_s.unsqueeze(1),
        w_ch_s.unsqueeze(0),
        out_dtype=torch.bfloat16,
        use_fast_accum=True)

    # Run the model with a cuda graph
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            output = model(a)

    output.zero_()
    g.replay()

    torch.testing.assert_close(output, output_ref, rtol=1e-3, atol=1e-3)
