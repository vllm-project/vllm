"""Tests for the machete kernel.

Run `pytest tests/kernels/test_machete_gemm.py`.
"""

import math
from typing import Optional, Tuple

import pytest
import torch

from tests.kernels.utils import opcheck
from vllm import _custom_ops as ops
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    pack_rows, quantize_weights)
from vllm.platforms import current_platform
from vllm.scalar_type import ScalarType, scalar_types

CUDA_DEVICES = [
    f"cuda:{i}" for i in range(1 if torch.cuda.device_count() == 1 else 2)
]

MNK_SHAPES = [
    (1, 128, 128),
    (1, 512, 1024),
    (1, 4096, 4096),
    (13, 8192, 4096),
    (26, 4096, 8192),
    (1, 4096, 4096),
    (257, 128, 4096),
    (257, 4224, 4160),
    (257, 4096, 4096),
    (64, 4096, 4096),
]

ACT_TYPES = [torch.float16, torch.bfloat16]
WTYPE_ZEROPOINTS = [
    # GPTQ style
    (scalar_types.uint4b8, False),
    (scalar_types.uint8b128, False),
    # AWQ style
    (scalar_types.uint4, True),
    (scalar_types.uint8, True),
]

# TODO: in future PR refactor this and `is_quant_method_supported` in the kernel
#  unit tests to a common utility function. Currently the use of
#  `is_quant_method_supported` conflates kernels with quantization methods
#  an assumption which is breaking down as quantizations methods can have
#  have kernels and some kernels support multiple quantization methods.
IS_SUPPORTED_BY_GPU = current_platform.get_device_capability()[0] >= 9


def rand_data(shape, dtype=torch.float16):
    return 10 * (torch.rand(shape, dtype=dtype, device="cuda") - 0.3)


def maybe_convert_zeropoints(zps: Optional[torch.Tensor], s: torch.Tensor):
    return zps if zps is None else -1 * s * (zps.to(s.dtype))


def machete_quantize_and_pack(w: torch.Tensor,
                              wtype: ScalarType,
                              group_size: int,
                              zero_points: bool = False):
    assert wtype.is_integer(), "TODO: support floating point weights"

    w_ref, w_q, w_s, w_zp = quantize_weights(
        w,
        wtype,
        group_size,
        zero_points=zero_points,
        # to match how the kernel applies zps
        ref_zero_points_after_scales=True)

    w_q = pack_rows(w_q, wtype.size_bits, *w_q.shape)
    w_q = w_q.t().contiguous().t()  # convert to col major
    w_q_machete = ops.machete_prepack_B(w_q, wtype)

    opcheck(torch.ops._C.machete_prepack_B, (w_q, wtype))

    return w_ref, w_q_machete, w_s, w_zp


def machete_gemm_test_helper(a: torch.Tensor, b: torch.Tensor,
                             wtype: ScalarType, group_size: int,
                             zero_points: bool):
    w_ref, w_q_packed, w_s, w_zp = machete_quantize_and_pack(
        b, wtype, group_size, zero_points)

    output_ref = torch.matmul(a, w_ref)

    output = ops.machete_gemm(
        a=a,
        b_q=w_q_packed,
        b_type=wtype,
        b_scales=w_s,
        b_zeros=maybe_convert_zeropoints(w_zp, w_s),
        b_group_size=group_size,
    )

    # Relax atol as our reduction dim becomes larger (more rounding error)
    # Relax atol when we have zeropoints since the way machete applies
    #  zeropoints (after scales) causes noise around 0
    atol = 1 if zero_points else min(5e-2 * math.sqrt(a.shape[1]), 1)
    torch.testing.assert_close(output, output_ref, rtol=1e-1, atol=atol)


@pytest.mark.skipif(not IS_SUPPORTED_BY_GPU,
                    reason="Machete is not supported on this GPU type.")
@pytest.mark.parametrize("shape",
                         MNK_SHAPES,
                         ids=lambda x: "x".join(str(v) for v in x))
@pytest.mark.parametrize("atype", ACT_TYPES, ids=lambda x: str(x))
@pytest.mark.parametrize("wtype_zeropoints", WTYPE_ZEROPOINTS)
@pytest.mark.parametrize("group_size", [128, None])
def test_machete_all_schedules(shape, atype: torch.dtype,
                               wtype_zeropoints: Tuple[ScalarType, bool],
                               group_size: Optional[int]):
    m, n, k = shape
    wtype, zero_points = wtype_zeropoints

    if group_size is not None and k % group_size != 0:
        return

    print(f"MNK = {m} {n} {k}")

    # Normalize group_size
    if group_size is None:
        group_size = k
    assert group_size <= k

    a = rand_data((m, k), atype)
    w = rand_data((k, n), atype)

    w_ref, w_q_machete, w_s, w_zp = machete_quantize_and_pack(
        w, wtype, group_size, zero_points)

    output_ref = torch.matmul(a, w_ref)

    for schedule in ops.machete_supported_schedules(wtype):
        output = ops.machete_gemm(
            a,
            b_q=w_q_machete,
            b_type=wtype,
            b_scales=w_s,
            b_zeros=maybe_convert_zeropoints(w_zp, w_s),
            b_group_size=group_size,
            schedule=schedule,
        )

        opcheck(torch.ops._C.machete_gemm,
                (a, w_q_machete, wtype, w_s, maybe_convert_zeropoints(
                    w_zp, w_s), group_size, None, None, None, schedule))

        # Relax atol as our reduction dim becomes larger (more rounding error)
        # Relax atol when we have zeropoints since the way machete applies
        #  zeropoints (after scales) causes noise around 0
        atol = 1 if zero_points else min(5e-2 * math.sqrt(k), 1)
        torch.testing.assert_close(output, output_ref, rtol=1e-1, atol=atol),\
               f"Schedule failed {schedule}"


@pytest.mark.skipif(not IS_SUPPORTED_BY_GPU,
                    reason="Machete is not supported on this GPU type.")
@pytest.mark.parametrize("shape",
                         MNK_SHAPES,
                         ids=lambda x: "x".join(str(v) for v in x))
@pytest.mark.parametrize("atype", ACT_TYPES, ids=lambda x: str(x))
@pytest.mark.parametrize("wtype_zeropoints", WTYPE_ZEROPOINTS)
@pytest.mark.parametrize("group_size", [128, None])
def test_machete_heuristic(shape, atype: torch.dtype,
                           wtype_zeropoints: Tuple[ScalarType, bool],
                           group_size: Optional[int]):
    m, n, k = shape
    wtype, zero_points = wtype_zeropoints

    if group_size is not None and k % group_size != 0:
        return

    # Normalize group_size
    if group_size is None:
        group_size = k
    assert group_size <= k

    a = rand_data((m, k), atype)
    b = rand_data((k, n), atype)

    machete_gemm_test_helper(a, b, wtype, group_size, zero_points)


# Test working on other devices
@pytest.mark.skipif(not IS_SUPPORTED_BY_GPU,
                    reason="Machete is not supported on this GPU type.")
@pytest.mark.parametrize("device", CUDA_DEVICES)
def test_machete_devices(device: str):
    m, n, k = 512, 4096, 4096
    wtype = scalar_types.uint4b8
    group_size = 128
    zero_points = False

    print(f"MNK = {m} {n} {k}, device = {device}")

    a = rand_data((m, k), torch.float16).to(device)
    b = rand_data((k, n), torch.float16).to(device)

    machete_gemm_test_helper(a, b, wtype, group_size, zero_points)


# Test working with a subset of A and B
@pytest.mark.skipif(not IS_SUPPORTED_BY_GPU,
                    reason="Machete is not supported on this GPU type.")
def test_machete_subset():
    big_m, big_n, big_k = 1024, 1024, 1024
    m, n, k = 512, 512, 512
    wtype = scalar_types.uint4b8
    group_size = 128
    zero_points = False

    whole_a = rand_data((big_m, big_k), torch.float16)
    whole_b = rand_data((big_k, big_n), torch.float16)

    a = whole_a[0:m, 0:k]
    b = whole_b[0:k, 0:n]

    machete_gemm_test_helper(a, b, wtype, group_size, zero_points)


# Test to make sure cuda graphs work
class MacheteLayer(torch.nn.Module):

    def __init__(self, **kwargs):
        super().__init__()
        self.kwargs = kwargs

    def forward(self, a):
        return ops.machete_gemm(**self.kwargs)


@pytest.mark.skipif(not IS_SUPPORTED_BY_GPU,
                    reason="Machete is not supported on this GPU type.")
def test_machete_cuda_graph():
    m, n, k = 512, 4096, 4096

    a = rand_data((m, k), torch.float16)
    b = rand_data((k, n), torch.float16)
    wtype = scalar_types.uint4b8
    group_size = 128
    zero_points = False

    w_ref, w_q_packed, w_s, w_zp = machete_quantize_and_pack(
        b, wtype, group_size, zero_points)

    # Construct a trivial model with a single layer that calls a machete kernel
    model = MacheteLayer(
        a=a,
        b_q=w_q_packed,
        b_type=wtype,
        b_scales=w_s,
        b_zeros=maybe_convert_zeropoints(w_zp, w_s),
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
