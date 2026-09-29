# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Scale kernels must preserve the PyTorch layout and E8M0 encoding bit for bit."""

from types import SimpleNamespace

import pytest
import torch

from tests.kernels.quantization.marlin_scales_utils import (
    prepare_scales,
    reference_permute_scales,
    reference_process_scales,
)
from tests.kernels.utils import opcheck
from vllm import _custom_ops as ops
from vllm.model_executor.layers.quantization.utils import marlin_utils_fp4 as fp4


@pytest.fixture(scope="module", autouse=True)
def require_gpu():
    if not torch.cuda.is_available():
        pytest.skip("GPU required")


def assert_bytes(actual, expected):
    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype
    assert actual.is_contiguous()
    assert torch.equal(
        actual.view(torch.uint8), expected.contiguous().view(torch.uint8)
    )


@pytest.mark.parametrize(
    "dtype", [torch.float16, torch.bfloat16, torch.float32, torch.uint8]
)
@pytest.mark.parametrize(
    "group_size,a8", [(-1, False), (32, False), (128, False), (32, True)]
)
@pytest.mark.parametrize("transpose", [False, True])
def test_permutation_preserves_bits_and_experts(dtype, group_size, a8, transpose):
    g, n = (1 if group_size == -1 else 256 // group_size), 128
    x = torch.arange(3 * g * n, device="cuda").reshape(3, g, n).to(dtype)
    if transpose:
        x = x.transpose(1, 2).contiguous().transpose(1, 2)
    for s in x:
        expected = reference_permute_scales(s, 256, n, group_size, a8)
        assert_bytes(ops.marlin_permute_scales(s, 256, n, group_size, a8), expected)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("a8", [False, True])
@pytest.mark.parametrize(
    "batched,padded", [(False, False), (True, False), (True, True)]
)
def test_scales_match_encoding_and_padding(dtype, a8, batched, padded):
    n, g = 128, 5
    high = (143 if dtype == torch.float16 else 250) if a8 else 256
    backing = (torch.arange(3 * n * g * 4, device="cuda") % high).to(torch.uint8)
    raw = backing.reshape(3, n * 2, g * 2)[:, ::2, ::2]
    raw.copy_((torch.arange(raw.numel(), device="cuda") % high).reshape(raw.shape))
    if not batched:
        raw = raw[0]
    before = raw.clone()
    act = torch.float8_e4m3fn if a8 else None
    kwargs = {"padded_k": 192, "padded_n": 192} if padded else {}
    expected = prepare_scales(raw, 32, dtype, act, reference=True, **kwargs)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        actual = prepare_scales(raw, 32, dtype, act, **kwargs)
        assert_bytes(actual, expected)
    stream.synchronize()
    assert torch.equal(raw, before)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_process_scales_rounding_and_special_values(dtype):
    x = torch.tensor(
        [
            0,
            -0.0,
            1,
            1.5,
            1.499,
            1.501,
            -1.5,
            float("inf"),
            float("nan"),
            2**-127,
            2**-24,
            2**16,
        ],
        device="cuda",
        dtype=dtype,
    ).reshape(3, 4)
    assert_bytes(ops.mxfp4_marlin_process_scales(x), reference_process_scales(x))


def test_fp8_rejects_overflow_but_keeps_249_boundary():
    raw = torch.full((2, 64, 1), 249, dtype=torch.uint8, device="cuda")
    act = torch.float8_e4m3fn
    good = prepare_scales(raw, 32, torch.bfloat16, act)
    assert torch.all(good.view(torch.uint8) == 255)
    raw[1, 63, 0] = 250
    with pytest.raises(AssertionError):
        prepare_scales(raw, 32, torch.bfloat16, act)


@pytest.mark.parametrize(
    "nvfp4,variant,input_dtype",
    [
        (False, variant, act)
        for variant in ("dense", "mixed_moe", "pure_moe")
        for act in (None, torch.float8_e4m3fn)
    ]
    + [(True, variant, None) for variant in ("dense", "mixed_moe")],
)
def test_preparation_call_sites_match_reference(
    monkeypatch, nvfp4, variant, input_dtype
):
    if not hasattr(torch.ops._C, "gptq_marlin_repack"):
        pytest.skip("Marlin weight repack is unavailable on this build")
    monkeypatch.setattr(fp4, "get_marlin_input_dtype", lambda: input_dtype)
    e, n, k = 2, 128, 128
    if variant == "dense":
        n, k = 200, 288
    group = 16 if nvfp4 else 32

    def weight(shape):
        return torch.randint(0, 256, shape, dtype=torch.uint8, device="cuda")

    def scale(shape):
        if nvfp4:
            return torch.ones(shape, dtype=torch.float8_e4m3fn, device="cuda")
        return torch.randint(110, 120, shape, dtype=torch.uint8, device="cuda")

    def run():
        torch.manual_seed(12)
        layer = torch.nn.Module()
        layer.params_dtype = torch.bfloat16
        if variant == "dense":
            layer.output_size_per_partition, layer.input_size_per_partition = n, k
            layer.weight, layer.weight_scale = (
                weight((n, k // 2)),
                scale((n, k // group)),
            )
            if nvfp4:
                layer.weight_global_scale = torch.ones(1, device="cuda")
            fp4.prepare_fp4_layer_for_marlin(layer, input_dtype)
            return layer.weight_scale, getattr(layer, "weight_global_scale", None)
        w13, w2 = weight((e, 2 * n, k // 2)), weight((e, k, n // 2))
        s13, s2 = scale((e, 2 * n, k // group)), scale((e, k, n // group))
        if variant == "pure_moe":
            result = fp4.prepare_moe_mxfp4_layer_for_marlin(
                layer, w13, w2, s13, s2, None, None
            )
            return result[2], result[3]
        layer.moe_config = SimpleNamespace(
            num_local_experts=e, hidden_dim=k, intermediate_size_per_partition=n
        )
        layer.w13_weight, layer.w2_weight = w13, w2
        layer.w13_weight_scale, layer.w2_weight_scale = s13, s2
        if nvfp4:
            layer.w13_weight_scale_2 = torch.ones(1, device="cuda")
            layer.w2_weight_scale_2 = torch.ones(1, device="cuda")
        fp4.prepare_moe_fp4_layer_for_marlin(layer, input_dtype)
        return layer.w13_weight_scale, layer.w2_weight_scale

    with monkeypatch.context() as reference:
        reference.setattr(ops, "marlin_permute_scales", reference_permute_scales)
        reference.setattr(ops, "mxfp4_marlin_process_scales", reference_process_scales)
        expected = run()
    for actual, reference in zip(run(), expected):
        if reference is not None:
            assert_bytes(actual, reference)


@pytest.mark.parametrize("a8", [False, True])
def test_single_group_uses_channelwise_permutation(a8):
    raw = (
        (torch.arange(3 * 128, device="cuda") % 30 + 110)
        .to(torch.uint8)
        .reshape(3, 128, 1)
    )
    act = torch.float8_e4m3fn if a8 else None
    expected = prepare_scales(raw, 32, torch.bfloat16, act, reference=True)
    assert_bytes(prepare_scales(raw, 32, torch.bfloat16, act), expected)


@pytest.mark.parametrize("nvfp4", [False, True])
def test_weight_test_helpers_keep_independent_dequantized_reference(monkeypatch, nvfp4):
    if not hasattr(torch.ops._C, "gptq_marlin_repack"):
        pytest.skip("Marlin weight repack is unavailable on this build")
    fn = (
        fp4.rand_marlin_weight_nvfp4_like
        if nvfp4
        else fp4.rand_marlin_weight_mxfp4_like
    )
    w = torch.ones((64, 128), dtype=torch.bfloat16, device="cuda")
    group = 16 if nvfp4 else 32
    torch.manual_seed(8)
    with monkeypatch.context() as reference:
        reference.setattr(ops, "marlin_permute_scales", reference_permute_scales)
        reference.setattr(ops, "mxfp4_marlin_process_scales", reference_process_scales)
        expected = fn(w, group)
    torch.manual_seed(8)
    for actual, reference in zip(fn(w, group), expected):
        assert torch.equal(
            actual.contiguous().reshape(-1).view(torch.uint8),
            reference.contiguous().reshape(-1).view(torch.uint8),
        )


def test_nvfp4_specialized_moe_keeps_global_scale_and_padding(monkeypatch):
    if not hasattr(torch.ops._C, "gptq_marlin_repack"):
        pytest.skip("Marlin weight repack is unavailable on this build")
    layer = SimpleNamespace(
        num_experts=2,
        hidden_size=128,
        intermediate_size_per_partition=96,
        params_dtype=torch.bfloat16,
    )
    w13 = torch.randint(0, 256, (2, 192, 64), dtype=torch.uint8, device="cuda")
    w2 = torch.randint(0, 256, (2, 128, 48), dtype=torch.uint8, device="cuda")
    s13 = (
        (torch.arange(2 * 192 * 8, device="cuda") % 8 + 1)
        .reshape(2, 192, 8)
        .to(torch.float8_e4m3fn)
    )
    s2 = (
        (torch.arange(2 * 128 * 6, device="cuda") % 4 + 1)
        .reshape(2, 128, 6)
        .to(torch.float8_e4m3fn)
    )
    global_scale = torch.ones(1, device="cuda")

    def run():
        return fp4.prepare_nvfp4_moe_layer_for_marlin(
            layer, w13, s13, global_scale, w2, s2, global_scale, True
        )

    with monkeypatch.context() as reference:
        reference.setattr(ops, "marlin_permute_scales", reference_permute_scales)
        expected = run()
    for actual, reference in zip(run(), expected):
        assert_bytes(actual, reference)


def test_scale_wrappers_replay_updated_weights():
    s = torch.ones((2, 64), dtype=torch.bfloat16, device="cuda")
    ops.mxfp4_marlin_process_scales(ops.marlin_permute_scales(s, 64, 64, 32))
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = ops.mxfp4_marlin_process_scales(ops.marlin_permute_scales(s, 64, 64, 32))
    ptr = out.data_ptr()
    for value, exponent in [(1, 127), (2, 128), (4, 129)]:
        s.fill_(value)
        graph.replay()
        torch.accelerator.synchronize()
        assert out.data_ptr() == ptr
        assert torch.all(out.view(torch.uint8) == exponent)


def test_out_buffer_validation_and_overlap_contract():
    empty = torch.empty((0, 64), dtype=torch.bfloat16, device="cuda")
    assert ops.marlin_permute_scales(empty, 32, 64, 32).shape == empty.shape
    s = torch.ones((2, 64), dtype=torch.bfloat16, device="cuda")
    with pytest.raises(RuntimeError, match="overlap"):
        torch.ops._C.marlin_permute_scales_out(s, s, False)
    invalid = s.view(torch.bool).reshape(-1)[:1]
    out = torch.empty(s.shape, dtype=torch.float8_e8m0fnu, device="cuda")
    with pytest.raises(RuntimeError, match="overlap"):
        torch.ops._C.mxfp4_marlin_process_scales_out(s, out, invalid, True)


def test_custom_op_contracts():
    s = torch.ones((2, 4, 64), dtype=torch.bfloat16, device="cuda")
    perm = torch.empty_like(s)
    out = torch.empty(s.shape, dtype=torch.float8_e8m0fnu, device="cuda")
    flags = torch.empty(2, dtype=torch.bool, device="cuda")
    opcheck(torch.ops._C.marlin_permute_scales_out, (s, perm, False, 32))
    opcheck(torch.ops._C.mxfp4_marlin_process_scales_out, (s, out, flags, True))
