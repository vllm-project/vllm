# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright (c) 2026 MetaX Integrated Circuits (Shanghai) Co., Ltd.

"""Correctness checks for the checkpoint-backed Composite Hadamard kernel."""

import math

import pytest
import torch

from vllm.platforms import current_platform


def _fwht_cpu(x: torch.Tensor) -> torch.Tensor:
    """Apply an unnormalized Hadamard transform using CPU butterflies."""
    width = x.shape[-1]
    stride = 1
    while stride < width:
        pairs = x.reshape(*x.shape[:-1], width // (2 * stride), 2, stride)
        left, right = pairs[..., 0, :], pairs[..., 1, :]
        x = torch.stack((left + right, left - right), dim=-2).reshape(
            *x.shape[:-1], width
        )
        stride *= 2
    return x


def _reference(
    x: torch.Tensor,
    sign: torch.Tensor,
    hk: torch.Tensor,
    k: int,
    p: int,
    inverse: bool,
) -> torch.Tensor:
    """Compute the transform in FP32 on CPU, independent of the GPU kernel."""
    values = x.cpu().float().reshape(-1, k * p)
    sign = sign.cpu().float()
    hk = hk.cpu().float()
    if not inverse:
        values *= sign
    values = _fwht_cpu(values.reshape(-1, k, p))
    values = torch.einsum("bip,ij->bjp", values, hk.T if inverse else hk)
    values = values.reshape(x.shape) / math.sqrt(k * p)
    if inverse:
        values *= sign
    return values


@pytest.fixture(scope="module")
def rotation_api():
    if not current_platform.is_cuda() or not torch.cuda.is_available():
        pytest.skip("Composite Hadamard CUDA tests require an NVIDIA GPU")
    pytest.importorskip("tilelang")
    from vllm.model_executor.kernels.rotation.composite_hadamard import (
        composite_hadamard,
    )

    return composite_hadamard


@pytest.mark.parametrize(
    "k,p,rows,dtype,noncontiguous",
    [
        pytest.param(1, 4096, 1, torch.bfloat16, False, id="power-of-two"),
        pytest.param(12, 512, 4, torch.bfloat16, False, id="composite"),
        pytest.param(12, 512, 4, torch.float16, False, id="fp16"),
        pytest.param(12, 512, 685, torch.bfloat16, False, id="prefill-tail"),
        pytest.param(12, 32, 4, torch.bfloat16, False, id="width-32"),
        pytest.param(12, 32, 2, torch.bfloat16, False, id="partial-warp"),
        pytest.param(12, 32, 4, torch.bfloat16, True, id="noncontiguous"),
        pytest.param(1, 4096, 1, torch.float32, False, id="fp32"),
    ],
)
def test_forward_and_inverse_match_cpu_reference(
    rotation_api, k, p, rows, dtype, noncontiguous
):
    """Both directions match an independent reference at key kernel boundaries."""
    op = rotation_api
    generator = torch.Generator().manual_seed(20261015 + k * p)
    if noncontiguous:
        x = torch.randn((k * p, rows), generator=generator, dtype=dtype).T
        assert not x.is_contiguous()
    else:
        x = torch.randn((rows, k * p), generator=generator, dtype=dtype)
    x = x.cuda()
    sign = (2 * torch.randint(0, 2, (k * p,), generator=generator) - 1).to(
        device="cuda", dtype=torch.int8
    )
    if k == 1:
        hk = torch.ones((1, 1), dtype=torch.float32)
    else:
        orthogonal, _ = torch.linalg.qr(
            torch.randn((k, k), generator=generator, dtype=torch.float32)
        )
        hk = orthogonal * math.sqrt(k)
    hk = hk.cuda().contiguous()

    for inverse in (False, True):
        original = x.clone()
        actual = op(x, sign, hk, inverse=inverse, k=k, p=p)
        expected = _reference(x, sign, hk, k, p, inverse)
        relative_error = (
            (actual.cpu().float() - expected).norm() / expected.norm()
        ).item()
        tolerance = 0.005 if dtype == torch.float32 else 0.02
        assert actual.shape == x.shape
        assert actual.dtype == dtype
        assert actual.is_contiguous()
        assert relative_error < tolerance
        torch.testing.assert_close(x, original, rtol=0, atol=0)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = op(x, sign, hk, inverse=inverse, k=k, p=p)
        graph.replay()
        torch.testing.assert_close(captured, actual, rtol=0, atol=0)
        if noncontiguous:
            torch.library.opcheck(
                torch.ops.vllm.composite_hadamard_tilelang.default,
                (x, sign, hk, inverse, k, p),
                test_utils=("test_faketensor",),
            )


def test_empty_batch_does_not_launch_kernel(rotation_api):
    """An empty batch must return without launching a zero-grid kernel."""
    op = rotation_api
    x = torch.empty((0, 4096), device="cuda", dtype=torch.bfloat16)
    sign = torch.ones(4096, device="cuda", dtype=torch.int8)
    hk = torch.ones((1, 1), device="cuda", dtype=torch.float32)
    result = op(x, sign, hk, inverse=True, k=1, p=4096)
    assert result.shape == x.shape


def test_cpu_reference_preserves_empty_batch():
    """CPU butterflies accept empty inputs without ambiguous reshape dimensions."""
    from vllm.model_executor.kernels.rotation.composite_hadamard import (
        composite_hadamard_torch,
    )

    x = torch.empty((0, 8), device="cpu", dtype=torch.float32)
    sign = torch.ones(8, device="cpu", dtype=torch.int8)
    hk = torch.ones((1, 1), device="cpu", dtype=torch.float32)
    result = composite_hadamard_torch(x, sign, hk, inverse=True, k=1, p=8)
    torch.testing.assert_close(result, x)


def test_invalid_dimensions_are_rejected(rotation_api):
    """Invalid width or K/P factorization fails before compiling a GPU kernel."""
    op = rotation_api
    x = torch.zeros((1, 72), device="cuda", dtype=torch.bfloat16)
    sign = torch.ones(72, device="cuda", dtype=torch.int8)
    hk = torch.eye(12, device="cuda", dtype=torch.float32)
    with pytest.raises(ValueError, match="power of two"):
        op(x, sign, hk, inverse=False, k=12, p=6)
    with pytest.raises(ValueError, match="hidden_size must equal"):
        op(x, sign, hk, inverse=False, k=12, p=4)


@pytest.mark.parametrize("startup_warmup", [False, True])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_epilogue_cache_with_and_without_startup_warmup(
    rotation_api, monkeypatch, startup_warmup, dtype
):
    """Warmup or the first call compiles once; eager/graph reuse dynamic rows."""
    from vllm.model_executor.kernels.rotation.composite_hadamard import warmup
    from vllm.model_executor.warmup.jit_warmup import JitWarmupRegistry

    kernel = warmup.CompositeHadamardEpilogueKernel()
    monkeypatch.setattr(warmup, "COMPOSITE_HADAMARD_EPILOGUE", kernel)
    registry = JitWarmupRegistry(None)
    key = dict(k=12, p=512, inverse=True, dtype=dtype)
    with registry.activate():
        kernel.register_warmup(**key)

    compiled: list[warmup.CompositeHadamardEpilogueKernel.CompileKey] = []
    compile_kernel = kernel.compile

    def record_compile(compile_key):
        compiled.append(compile_key)
        compile_kernel(compile_key)

    monkeypatch.setattr(kernel, "compile", record_compile)
    if startup_warmup:
        registry.warmup()
    assert len(compiled) == int(startup_warmup)
    sign = torch.ones(6144, device="cuda", dtype=torch.int8)
    hk = torch.eye(12, device="cuda", dtype=torch.float32) * math.sqrt(12)
    for rows in (1, 32):
        x = torch.randn(rows, 6144, device="cuda", dtype=dtype)
        actual = rotation_api(x, sign, hk, inverse=True, k=12, p=512)
        expected = _reference(x, sign, hk, 12, 512, True)
        error = (actual.cpu().float() - expected).norm() / expected.norm()
        assert error < 0.02
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = rotation_api(x, sign, hk, inverse=True, k=12, p=512)
        graph.replay()
        torch.testing.assert_close(captured, actual, rtol=0, atol=0)
        assert compiled == [kernel.CompileKey(**key)]
