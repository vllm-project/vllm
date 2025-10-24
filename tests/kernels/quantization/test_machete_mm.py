# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the machete kernel.

Run `pytest tests/kernels/quantization/test_machete_mm.py`.
"""

import math
from dataclasses import dataclass, fields

import pytest
import torch

from tests.kernels.utils import opcheck
from vllm import _custom_ops as ops
from vllm.model_executor.layers.quantization.utils.machete_utils import (
    query_machete_supported_group_sizes,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    pack_rows,
    quantize_weights,
)
from vllm.platforms import current_platform
from vllm.scalar_type import ScalarType, scalar_types

CUDA_DEVICES = [f"cuda:{i}" for i in range(1 if torch.cuda.device_count() == 1 else 2)]

# TODO: in future PR refactor this and `is_quant_method_supported` in the kernel
#  unit tests to a common utility function. Currently the use of
#  `is_quant_method_supported` conflates kernels with quantization methods
#  an assumption which is breaking down as quantizations methods can have
#  have kernels and some kernels support multiple quantization methods.
IS_SUPPORTED_BY_GPU = current_platform.get_device_capability()[0] >= 9

MNK_SHAPES = [
    (1, 128, 128),
    (1, 8192, 28672),
    (13, 8192, 4096),
    (26, 4096, 8192),
    (64, 4096, 4096),
    (64, 8192, 28672),
    (257, 128, 4096),
    (257, 4224, 4160),
    (1024, 8192, 4096),
]


@dataclass
class TypeConfig:
    act_type: torch.dtype
    weight_type: ScalarType
    output_type: torch.dtype | None
    group_scale_type: torch.dtype | None
    group_zero_type: torch.dtype | None
    channel_scale_type: torch.dtype | None
    token_scale_type: torch.dtype | None


@dataclass
class Tensors:
    w_ref: torch.Tensor
    a_ref: torch.Tensor
    a: torch.Tensor
    w_q: torch.Tensor
    w_g_s: torch.Tensor | None
    w_g_zp: torch.Tensor | None
    w_ch_s: torch.Tensor | None
    w_tok_s: torch.Tensor | None


# (Act Type, Weight Type, Output Type, Scale Type, ZeroPoints,
#  Ch Scales Type, Tok Scales Type)
# NOTE: None "Scale Type" means the act type is floating point
#       None "Output Type" means the output type is the same as the act type
TestTypeTuple = tuple[
    list[torch.dtype], ScalarType, torch.dtype | None, torch.dtype | None, bool
]
TEST_TYPES = [
    # GPTQ style
    *(
        TypeConfig(
            act_type=a_type,
            weight_type=w_type,
            output_type=None,
            group_scale_type=a_type,
            group_zero_type=None,
            channel_scale_type=None,
            token_scale_type=None,
        )
        for w_type in [scalar_types.uint4b8, scalar_types.uint8b128]
        for a_type in [torch.float16, torch.bfloat16]
    ),
    # AWQ style
    *(
        TypeConfig(
            act_type=a_type,
            weight_type=w_type,
            output_type=None,
            group_scale_type=a_type,
            group_zero_type=a_type,
            channel_scale_type=None,
            token_scale_type=None,
        )
        for w_type in [scalar_types.uint4, scalar_types.uint8]
        for a_type in [torch.float16, torch.bfloat16]
    ),
    # # QQQ style
    # *(TypeConfig(act_type=torch.int8,
    #              weight_type=scalar_types.uint4b8,
    #              output_type=torch.float16,
    #              group_scale_type=group_scale_type,
    #              group_zero_type=None,
    #              channel_scale_type=torch.float,
    #              token_scale_type=torch.float)
    #   for group_scale_type in [None, torch.float16]),
    # *(TypeConfig(act_type=torch.float8_e4m3fn,
    #              weight_type=scalar_types.uint4b8,
    #              output_type=torch.float16,
    #              group_scale_type=group_scale_type,
    #              group_zero_type=None,
    #              channel_scale_type=torch.float,
    #              token_scale_type=torch.float)
    #   for group_scale_type in [None, torch.float16]),
]

# TODO: in future PR refactor this and `is_quant_method_supported` in the kernel
#  unit tests to a common utility function. Currently the use of
#  `is_quant_method_supported` conflates kernels with quantization methods
#  an assumption which is breaking down as quantizations methods can have
#  have kernels and some kernels support multiple quantization methods.
IS_SUPPORTED_BY_GPU = current_platform.has_device_capability(90)


def rand_data(shape, dtype=torch.float16, scale=1, offset=0):
    if dtype.is_floating_point:
        return (scale * torch.rand(shape, device="cuda") - offset).to(dtype)
    else:
        return torch.randint(-8, 7, shape, dtype=dtype, device="cuda")


def maybe_convert_zeropoints(zps: torch.Tensor | None, s: torch.Tensor):
    return zps if zps is None else -1 * s * (zps.to(s.dtype))


def group_size_valid(shape: tuple[int, int, int], group_size: int | None) -> bool:
    return group_size is None or group_size == -1 or shape[2] % group_size == 0


def machete_quantize_and_pack(
    atype: torch.dtype,
    w: torch.Tensor,
    wtype: ScalarType,
    stype: torch.dtype | None,
    group_size: int | None,
    zero_points: bool = False,
):
    assert wtype.is_integer(), "TODO: support floating point weights"

    w_ref, w_q, w_s, w_zp = quantize_weights(
        w,
        wtype,
        group_size=group_size,
        zero_points=zero_points,
        # to match how the kernel applies zps
        ref_zero_points_after_scales=True,
    )

    w_q = pack_rows(w_q, wtype.size_bits, *w_q.shape)
    w_q = w_q.t().contiguous().t()  # convert to col major

    w_q_machete = ops.machete_prepack_B(w_q, atype, wtype, stype)
    opcheck(torch.ops._C.machete_prepack_B, (w_q, atype, wtype.id, stype))

    return w_ref, w_q_machete, w_s, w_zp


def create_test_tensors(
    shape: tuple[int, int, int],
    types: TypeConfig,
    group_size: int | None,
    subset_stride_factor: int | None = None,
) -> Tensors:
    m, n, k = shape
    factor = subset_stride_factor or 1

    print(
        "create_test_tensors, shape:", shape, "types:", types, "group_size:", group_size
    )

    a = rand_data((m * factor, k * factor), types.act_type, scale=3, offset=2)
    w = rand_data((k * factor, n * factor), types.act_type, scale=3, offset=1)

    if factor > 1:
        a = a[0:m, 0:k]
        w = w[0:k, 0:n]

    if types.group_scale_type is not None:
        w = w.to(types.group_scale_type)
    if w.dtype.itemsize == 1:
        w = w.to(torch.float16)

    w_ref, w_q_packed, w_s, w_zp = machete_quantize_and_pack(
        a.dtype,
        w,
        types.weight_type,
        types.group_scale_type,
        group_size,
        types.group_zero_type is not None,
    )

    if not a.dtype.is_floating_point:
        aiinfo = torch.iinfo(a.dtype)
        w_ref = w_ref.round().clamp(aiinfo.min, aiinfo.max)

    a_ref = a.to(torch.float32)
    w_ref = w_ref.to(torch.float32)

    w_ch_s = (
        None
        if types.channel_scale_type is None
        else rand_data((n,), types.channel_scale_type)
    )
    w_tok_s = (
        None
        if types.token_scale_type is None
        else rand_data((m,), types.token_scale_type)
    )

    return Tensors(
        w_ref=w_ref,
        a_ref=a_ref,
        a=a,
        w_q=w_q_packed,
        w_g_s=w_s,
        w_g_zp=maybe_convert_zeropoints(w_zp, w_s),
        w_ch_s=w_ch_s,
        w_tok_s=w_tok_s,
    )


# None stype means scales use the same dtype as a
def machete_mm_test_helper(
    types: TypeConfig,
    tensors: Tensors,
    group_size: int | None = None,
    schedule: str | None = None,
):
    output_ref = torch.matmul(tensors.a_ref, tensors.w_ref)
    output_ref_type = output_ref.dtype

    if tensors.w_ch_s is not None:
        output_ref = (
            output_ref.to(tensors.w_ch_s.dtype) * tensors.w_ch_s.unsqueeze(0)
        ).to(output_ref_type)
    if tensors.w_tok_s is not None:
        output_ref = (
            output_ref.to(tensors.w_tok_s.dtype) * tensors.w_tok_s.unsqueeze(1)
        ).to(output_ref_type)

    output = ops.machete_mm(
        a=tensors.a,
        b_q=tensors.w_q,
        b_type=types.weight_type,
        b_group_scales=tensors.w_g_s,
        b_group_zeros=tensors.w_g_zp,
        b_group_size=group_size,
        b_channel_scales=tensors.w_ch_s,
        a_token_scales=tensors.w_tok_s,
        out_type=types.output_type,
        schedule=schedule,
    )

    print(output)
    print(output_ref)

    # Relax atol as our reduction dim becomes larger (more rounding error)
    # Relax atol when we have zeropoints since the way machete applies
    #  zeropoints (after scales) causes noise around 0
    atol = (
        1
        if tensors.w_g_zp is not None
        else min(5e-2 * math.sqrt(tensors.a.shape[1]), 1)
    )
    rtol = 1e-1 if tensors.a.element_size() >= 2 else 2e-1
    torch.testing.assert_close(
        output, output_ref.to(output.dtype), rtol=rtol, atol=atol
    )


@pytest.mark.skipif(
    not IS_SUPPORTED_BY_GPU, reason="Machete is not supported on this GPU type."
)
@pytest.mark.parametrize("shape", MNK_SHAPES, ids=lambda x: "x".join(str(v) for v in x))
@pytest.mark.parametrize("types", TEST_TYPES)
def test_machete_all_schedules(shape, types: TypeConfig):
    group_sizes: list[int | None] = []
    if types.group_scale_type is None:
        group_sizes = [None]
    else:
        group_sizes = query_machete_supported_group_sizes(types.act_type)

    for group_size in group_sizes:
        if not group_size_valid(shape, group_size):
            continue

        tensors = create_test_tensors(shape, types, group_size)
        print(f"MNK = {shape}")
        for schedule in ops.machete_supported_schedules(
            types.act_type,
            types.weight_type,
            group_scales_type=types.group_scale_type,
            group_zeros_type=types.group_scale_type,
            out_type=types.output_type,
        ):
            print(f"Testing schedule {schedule}")
            machete_mm_test_helper(types, tensors, group_size, schedule)


@pytest.mark.skipif(
    not IS_SUPPORTED_BY_GPU, reason="Machete is not supported on this GPU type."
)
@pytest.mark.parametrize("shape", MNK_SHAPES, ids=lambda x: "x".join(str(v) for v in x))
@pytest.mark.parametrize("types", TEST_TYPES)
def test_machete_heuristic(shape, types: TypeConfig):
    group_sizes: list[int | None] = []
    if types.group_scale_type is None:
        group_sizes = [None]
    else:
        group_sizes = query_machete_supported_group_sizes(types.act_type)

    for group_size in group_sizes:
        if not group_size_valid(shape, group_size):
            continue

        tensors = create_test_tensors(shape, types, group_size)
        machete_mm_test_helper(types, tensors, group_size)


# Test working on other devices
@pytest.mark.skipif(
    not IS_SUPPORTED_BY_GPU, reason="Machete is not supported on this GPU type."
)
@pytest.mark.parametrize("device", CUDA_DEVICES)
def test_machete_devices(device: str):
    group_size = 128

    type_config = TypeConfig(
        act_type=torch.float16,
        weight_type=scalar_types.uint4b8,
        output_type=None,
        group_scale_type=torch.float16,
        group_zero_type=None,
        channel_scale_type=None,
        token_scale_type=None,
    )

    tensors = create_test_tensors((512, 4096, 4096), type_config, group_size)

    for field in fields(Tensors):
        tensor = getattr(tensors, field.name)
        if isinstance(tensor, torch.Tensor):
            setattr(tensors, field.name, tensor.to(device))

    machete_mm_test_helper(type_config, tensors, group_size)


# Test working with a subset of A and B
@pytest.mark.skipif(
    not IS_SUPPORTED_BY_GPU, reason="Machete is not supported on this GPU type."
)
def test_machete_subset():
    group_size = 128

    type_config = TypeConfig(
        act_type=torch.float16,
        weight_type=scalar_types.uint4b8,
        output_type=None,
        group_scale_type=torch.float16,
        group_zero_type=None,
        channel_scale_type=None,
        token_scale_type=None,
    )

    tensors = create_test_tensors(
        (512, 4096, 4096), type_config, group_size, subset_stride_factor=2
    )
    machete_mm_test_helper(type_config, tensors, group_size)


# Test to make sure cuda graphs work
class MacheteLayer(torch.nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.kwargs = kwargs

    def forward(self, a):
        return ops.machete_mm(a=a, **self.kwargs)


@pytest.mark.skipif(
    not IS_SUPPORTED_BY_GPU, reason="Machete is not supported on this GPU type."
)
def test_machete_cuda_graph():
    m, n, k = 512, 4096, 4096

    a = rand_data((m, k), torch.float16)
    b = rand_data((k, n), torch.float16)
    wtype = scalar_types.uint4b8
    stype = torch.float16
    group_size = 128
    zero_points = False

    w_ref, w_q_packed, w_s, w_zp = machete_quantize_and_pack(
        a.dtype, b, wtype, stype, group_size, zero_points
    )

    # Construct a trivial model with a single layer that calls a machete kernel
    model = MacheteLayer(
        b_q=w_q_packed,
        b_type=wtype,
        b_group_scales=w_s,
        b_group_zeros=maybe_convert_zeropoints(w_zp, w_s),
        b_group_size=group_size,
    )

    output_ref = torch.matmul(a, w_ref)

    # Run the model with a cuda graph
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            output = model(a)
    output.zero_()
    g.replay()

    # Relax atol as our reduction dim becomes larger (more rounding error)
    # Relax atol when we have zeropoints since the way machete applies
    #  zeropoints (after scales) causes noise around 0
    atol = 1 if zero_points else min(5e-2 * math.sqrt(k), 1)
    torch.testing.assert_close(output, output_ref, rtol=1e-1, atol=atol)
