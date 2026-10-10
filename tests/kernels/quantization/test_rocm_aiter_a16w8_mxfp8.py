# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""RocmAiterA16W8Mxfp8LinearKernel (gfx942) against the BF16 emulation kernel."""

import pytest
import torch

from vllm.platforms import current_platform

if not current_platform.is_rocm():
    pytest.skip("ROCm only", allow_module_level=True)

from vllm.platforms.rocm import on_gfx942  # noqa: E402

if not on_gfx942():
    pytest.skip(
        "the AITER MXFP8 A16W8 asm kernels are built for gfx942",
        allow_module_level=True,
    )
pytest.importorskip("aiter.ops.gemm_op_a16w8_mxfp8")

from vllm.model_executor.kernels.linear.mxfp8 import (  # noqa: E402
    Mxfp8LinearLayerConfig,
)
from vllm.model_executor.kernels.linear.mxfp8.emulation import (  # noqa: E402
    EmulationMxfp8LinearKernel,
)
from vllm.model_executor.kernels.linear.mxfp8.rocm_aiter_a16w8 import (  # noqa: E402
    RocmAiterA16W8Mxfp8LinearKernel,
)

# Tuned in AITER's table on MI300X / MI325X (304 CUs); (N, K).
SHAPES = [(1792, 5120), (8192, 1280), (5120, 2048), (1152, 5120), (5120, 576)]


@pytest.fixture(autouse=True)
def _enable(monkeypatch):
    monkeypatch.setenv("VLLM_ROCM_USE_AITER", "1")
    monkeypatch.setenv("VLLM_ROCM_USE_AITER_MXFP8_ASM_GEMM", "1")
    from vllm._aiter_ops import rocm_aiter_ops

    rocm_aiter_ops.refresh_env_variables()
    yield
    monkeypatch.delenv("VLLM_ROCM_USE_AITER_MXFP8_ASM_GEMM")
    rocm_aiter_ops.refresh_env_variables()


def _layer(n, k, seed):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    g = k // 32
    w = torch.randn(n, k, device="cuda", generator=gen) * k**-0.5
    e = (
        torch.floor(torch.log2(w.view(n, g, 32).abs().amax(-1).clamp_min(2.0**-100)))
        - 8
    )
    q = (w.view(n, g, 32) / torch.exp2(e)[..., None]).clamp(-448, 448)
    layer = torch.nn.Module()
    layer.weight = torch.nn.Parameter(
        q.to(torch.float8_e4m3fn).view(n, k), requires_grad=False
    )
    layer.weight_scale = torch.nn.Parameter(
        (e + 127).to(torch.uint8), requires_grad=False
    )
    return layer


@pytest.mark.parametrize("n,k", SHAPES)
@pytest.mark.parametrize("rows", [1, 12, 48, 200])
def test_matches_emulation(n, k, rows):
    config = Mxfp8LinearLayerConfig(weight_shape=(n, k))
    ok, reason = RocmAiterA16W8Mxfp8LinearKernel.is_supported()
    assert ok, reason
    ok, reason = RocmAiterA16W8Mxfp8LinearKernel.can_implement(config)
    assert ok, reason
    ref_layer, layer = _layer(n, k, seed=n + k), _layer(n, k, seed=n + k)
    EmulationMxfp8LinearKernel(config).process_weights_after_loading(ref_layer)
    kernel = RocmAiterA16W8Mxfp8LinearKernel(config)
    kernel.process_weights_after_loading(layer)

    x = torch.randn(rows, k, device="cuda", dtype=torch.bfloat16)
    ref = EmulationMxfp8LinearKernel(config).apply_weights(ref_layer, x)
    out = kernel.apply_weights(layer, x)
    assert out.shape == ref.shape and out.dtype == torch.bfloat16
    if rows > 64:
        # Above 64 rows the op runs the same BF16 linear as the emulation kernel.
        torch.testing.assert_close(out, ref, rtol=0, atol=0)
    else:
        # The asm kernel and hipBLASLt both sum in fp32 and round to BF16.
        torch.testing.assert_close(out, ref, rtol=1.6e-2, atol=1e-3)


def test_kernel_selection(monkeypatch):
    from vllm._aiter_ops import rocm_aiter_ops
    from vllm.model_executor.kernels.linear import init_mxfp8_linear_kernel

    def selected(n, k):
        kernel = init_mxfp8_linear_kernel(weight_shape=(n, k))
        return isinstance(kernel, RocmAiterA16W8Mxfp8LinearKernel)

    assert selected(1792, 5120)
    assert not selected(4096, 4096)  # not in AITER's tuned table
    monkeypatch.setenv("VLLM_ROCM_USE_AITER_MXFP8_ASM_GEMM", "0")
    rocm_aiter_ops.refresh_env_variables()
    assert not selected(1792, 5120)


def test_cuda_graph_replay_equals_eager():
    n, k = 1152, 5120  # split-K kernel at 12 rows
    config = Mxfp8LinearLayerConfig(weight_shape=(n, k))
    layer = _layer(n, k, seed=3)
    kernel = RocmAiterA16W8Mxfp8LinearKernel(config)
    kernel.process_weights_after_loading(layer)
    x = torch.randn(12, k, device="cuda", dtype=torch.bfloat16)
    eager = kernel.apply_weights(layer, x).clone()
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        kernel.apply_weights(layer, x)
    torch.cuda.current_stream().wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = kernel.apply_weights(layer, x)
    for _ in range(3):
        graph.replay()
        torch.accelerator.synchronize()
        torch.testing.assert_close(out, eager, rtol=0, atol=0)
