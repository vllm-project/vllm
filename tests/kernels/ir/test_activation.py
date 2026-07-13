# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

import vllm.kernels  # noqa: F401
from tests.ir.ir_test_utils import assert_close, clone_args
from vllm import ir
from vllm.platforms import current_platform
from vllm.triton_utils import HAS_TRITON

gelu_and_mul_sparse_native = ir.ops.gelu_and_mul_sparse.impls["native"].impl_fn


def test_gelu_and_mul_sparse_registration():
    expected = {
        "native": True,
        "triton": HAS_TRITON and current_platform.is_cuda(),
    }
    actual = {
        provider: impl.supported
        for provider, impl in ir.ops.gelu_and_mul_sparse.impls.items()
    }
    assert actual == expected


@pytest.mark.skipif(not current_platform.is_cuda(), reason="CUDA-only provider")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("num_tokens", [1, 8, 64])
@pytest.mark.parametrize("intermediate_size", [2048, 4096, 8192, 13824, 16384, 32768])
def test_gelu_and_mul_sparse_triton(
    dtype: torch.dtype, num_tokens: int, intermediate_size: int
):
    torch.manual_seed(0)
    torch.set_default_device("cuda")
    args = ir.ops.gelu_and_mul_sparse.generate_inputs(
        num_tokens=num_tokens,
        hidden_size=intermediate_size,
        dtype=dtype,
    )
    impl = ir.ops.gelu_and_mul_sparse.impls["triton"]
    assert impl.supports_args(*args)

    expected = gelu_and_mul_sparse_native(*clone_args(args))
    actual = impl.impl_fn(*clone_args(args))
    assert_close(ir.ops.gelu_and_mul_sparse, actual, expected)

    mask_mismatch = torch.count_nonzero((actual == 0) != (expected == 0))
    assert mask_mismatch / actual.numel() < 1e-3


@pytest.mark.skipif(not current_platform.is_cuda(), reason="CUDA-only provider")
def test_gelu_and_mul_sparse_triton_second_seed():
    torch.manual_seed(1)
    torch.set_default_device("cuda")
    args = ir.ops.gelu_and_mul_sparse.generate_inputs(
        num_tokens=32,
        hidden_size=8192,
        dtype=torch.bfloat16,
    )

    actual = ir.ops.gelu_and_mul_sparse.impls["triton"].impl_fn(*clone_args(args))
    expected = gelu_and_mul_sparse_native(*clone_args(args))
    assert_close(ir.ops.gelu_and_mul_sparse, actual, expected)


@pytest.mark.skipif(not current_platform.is_cuda(), reason="CUDA-only provider")
@pytest.mark.parametrize("intermediate_size", [1, 31, 128])
def test_gelu_and_mul_sparse_triton_small_sizes(intermediate_size: int):
    torch.manual_seed(0)
    torch.set_default_device("cuda")
    args = ir.ops.gelu_and_mul_sparse.generate_inputs(
        num_tokens=5,
        hidden_size=intermediate_size,
        dtype=torch.bfloat16,
    )

    actual = ir.ops.gelu_and_mul_sparse.impls["triton"].impl_fn(*clone_args(args))
    expected = gelu_and_mul_sparse_native(*clone_args(args))
    assert_close(ir.ops.gelu_and_mul_sparse, actual, expected)


@pytest.mark.skipif(not current_platform.is_cuda(), reason="CUDA-only provider")
def test_gelu_and_mul_sparse_triton_nd_and_empty():
    torch.set_default_device("cuda")
    impl = ir.ops.gelu_and_mul_sparse.impls["triton"]

    x = torch.randn(2, 3, 16384, dtype=torch.bfloat16)
    args = (x, 1.6448536269514722, "tanh")
    actual = impl.impl_fn(*args)
    expected = gelu_and_mul_sparse_native(*args)
    assert actual.shape == (2, 3, 8192)
    assert_close(ir.ops.gelu_and_mul_sparse, actual, expected)

    empty = torch.empty(0, 16384, dtype=torch.bfloat16)
    empty_out = impl.impl_fn(empty, 1.6448536269514722, "tanh")
    assert empty_out.shape == (0, 8192)
    assert empty_out.dtype == empty.dtype


@pytest.mark.skipif(not current_platform.is_cuda(), reason="CUDA-only provider")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_gelu_and_mul_sparse_triton_constant_rows(dtype: torch.dtype):
    torch.set_default_device("cuda")
    gate = torch.full((4, 8192), 3.0, dtype=dtype)
    up = torch.randn_like(gate)
    x = torch.cat((gate, up), dim=-1)
    args = (x, 1.6448536269514722, "tanh")

    actual = ir.ops.gelu_and_mul_sparse.impls["triton"].impl_fn(*args)
    expected = gelu_and_mul_sparse_native(*args)
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)
    assert torch.count_nonzero(actual) == 0


@pytest.mark.skipif(not current_platform.is_cuda(), reason="CUDA-only provider")
@pytest.mark.parametrize("nonfinite", [float("nan"), float("inf")])
def test_gelu_and_mul_sparse_triton_nonfinite_gate(nonfinite: float):
    torch.set_default_device("cuda")
    x = torch.randn(2, 16384, dtype=torch.bfloat16)
    x[0, 0] = nonfinite
    args = (x, 1.6448536269514722, "tanh")

    actual = ir.ops.gelu_and_mul_sparse.impls["triton"].impl_fn(*args)
    expected = gelu_and_mul_sparse_native(*args)
    torch.testing.assert_close(actual, expected, equal_nan=True)


@pytest.mark.skipif(not current_platform.is_cuda(), reason="CUDA-only provider")
def test_gelu_and_mul_sparse_triton_fallback_conditions():
    torch.set_default_device("cuda")
    impl = ir.ops.gelu_and_mul_sparse.impls["triton"]
    x = torch.randn(4, 16384, dtype=torch.bfloat16)

    assert impl.supports_args(x, 1.0, "tanh")
    assert not impl.supports_args(x, 1.0, "none")
    assert not impl.supports_args(x[:, ::2], 1.0, "tanh")
    assert not impl.supports_args(x[:, :-1], 1.0, "tanh")
    assert not impl.supports_args(
        torch.randn(1, 65538, dtype=torch.bfloat16), 1.0, "tanh"
    )


@pytest.mark.skipif(not current_platform.is_cuda(), reason="CUDA-only provider")
def test_gelu_and_mul_sparse_triton_dispatch_and_opcheck():
    torch.set_default_device("cuda")
    args = ir.ops.gelu_and_mul_sparse.generate_inputs(
        num_tokens=8,
        hidden_size=4096,
        dtype=torch.bfloat16,
    )
    impl = ir.ops.gelu_and_mul_sparse.impls["triton"]

    with ir.ops.gelu_and_mul_sparse.set_priority(["triton", "native"]):
        dispatched = ir.ops.gelu_and_mul_sparse(*args)
        torch.library.opcheck(torch.ops.vllm_ir.gelu_and_mul_sparse, args)

    direct = impl.impl_fn(*args)
    torch.testing.assert_close(dispatched, direct, rtol=0.0, atol=0.0)


@pytest.mark.skipif(not current_platform.is_cuda(), reason="CUDA-only provider")
def test_gelu_and_mul_sparse_triton_cudagraph():
    torch.set_default_device("cuda")
    x = torch.randn(16, 32768, dtype=torch.bfloat16)
    impl = ir.ops.gelu_and_mul_sparse.impls["triton"]

    for _ in range(3):
        expected = impl.impl_fn(x, 1.6448536269514722, "tanh")
    torch.accelerator.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = impl.impl_fn(x, 1.6448536269514722, "tanh")
    graph.replay()
    torch.accelerator.synchronize()

    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)
