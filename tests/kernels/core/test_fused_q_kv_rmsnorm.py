# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Correctness + large-token-count launch tests for fused_q_kv_rmsnorm.

Before the grid-dim fix the kernel used grid ``(2, num_tokens)``, which hit
CUDA's 65535 grid-y cap for ``num_tokens >= 65536`` and failed with
``Triton Error [CUDA]: invalid argument`` at every large chunked-prefill
profile run. These tests pin the new grid layout.
"""

from __future__ import annotations

import pytest
import torch

from vllm.models.common.ops.fused_qk_rmsnorm import (
    fused_q_kv_rmsnorm,
)
from vllm.platforms import current_platform

pytestmark = pytest.mark.skipif(
    not current_platform.is_cuda_alike(),
    reason="fused_q_kv_rmsnorm requires a CUDA/ROCm device",
)


def _ref_rmsnorm(x: torch.Tensor, w: torch.Tensor, eps: float) -> torch.Tensor:
    x_f32 = x.to(torch.float32)
    variance = x_f32.pow(2).mean(dim=-1, keepdim=True)
    y = x_f32 * torch.rsqrt(variance + eps) * w.to(torch.float32)
    return y.to(x.dtype)


@pytest.mark.parametrize("num_tokens", [1, 17, 1024, 8192])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_fused_q_kv_rmsnorm_correctness(num_tokens: int, dtype: torch.dtype):
    torch.manual_seed(0)
    device = "cuda"
    q_size, kv_size = 192, 576
    qr = torch.randn(num_tokens, q_size, dtype=dtype, device=device)
    kv = torch.randn(num_tokens, kv_size, dtype=dtype, device=device)
    qw = torch.randn(q_size, dtype=dtype, device=device)
    kvw = torch.randn(kv_size, dtype=dtype, device=device)
    eps = 1e-6

    qr_out, kv_out = fused_q_kv_rmsnorm(qr, kv, qw, kvw, eps)

    qr_ref = _ref_rmsnorm(qr, qw, eps)
    kv_ref = _ref_rmsnorm(kv, kvw, eps)

    tol = dict(rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(qr_out, qr_ref, **tol)
    torch.testing.assert_close(kv_out, kv_ref, **tol)


@pytest.mark.parametrize("num_tokens", [1, 16])
def test_fused_q_kv_rmsnorm_outputs_are_packed(num_tokens: int):
    """Regression guard: the outputs must be packed row-major even when the
    inputs are column slices of a wider fused-projection buffer.

    empty_like preserves the strides of size-1 dims, so with num_tokens == 1
    a [1, q_size] slice of a [1, q_size + kv_size] buffer used to produce a
    qr_out with row stride q_size + kv_size. Downstream dispatchers that
    require packed row-major inputs then reject the tensor and silently fall
    back to a slower GEMM path on every decode step."""
    device = "cuda"
    dtype = torch.bfloat16
    q_size, kv_size = 192, 576
    fused = torch.randn(num_tokens, q_size + kv_size, dtype=dtype, device=device)
    qr, kv = fused.split([q_size, kv_size], dim=-1)
    qw = torch.randn(q_size, dtype=dtype, device=device)
    kvw = torch.randn(kv_size, dtype=dtype, device=device)
    eps = 1e-6

    qr_out, kv_out = fused_q_kv_rmsnorm(qr, kv, qw, kvw, eps)

    assert qr_out.stride() == (q_size, 1)
    assert kv_out.stride() == (kv_size, 1)

    qr_ref = _ref_rmsnorm(qr, qw, eps)
    kv_ref = _ref_rmsnorm(kv, kvw, eps)
    tol = dict(rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(qr_out, qr_ref, **tol)
    torch.testing.assert_close(kv_out, kv_ref, **tol)


@pytest.mark.parametrize("num_tokens", [65535, 65536, 131072])
def test_fused_q_kv_rmsnorm_launches_past_grid_y_cap(num_tokens: int):
    """Regression guard: grid used to be (2, num_tokens), hitting CUDA's
    65535 grid-y cap at num_tokens >= 65536. The new grid (num_tokens, 2)
    lifts that bound to 2**31-1."""
    device = "cuda"
    dtype = torch.bfloat16
    q_size, kv_size = 192, 576
    qr = torch.randn(num_tokens, q_size, dtype=dtype, device=device)
    kv = torch.randn(num_tokens, kv_size, dtype=dtype, device=device)
    qw = torch.randn(q_size, dtype=dtype, device=device)
    kvw = torch.randn(kv_size, dtype=dtype, device=device)

    qr_out, kv_out = fused_q_kv_rmsnorm(qr, kv, qw, kvw, 1e-6)
    # spot-check a couple of rows against the torch reference
    for row in (0, num_tokens // 2, num_tokens - 1):
        torch.testing.assert_close(
            qr_out[row],
            _ref_rmsnorm(qr[row : row + 1], qw, 1e-6)[0],
            rtol=1e-2,
            atol=1e-2,
        )
        torch.testing.assert_close(
            kv_out[row],
            _ref_rmsnorm(kv[row : row + 1], kvw, 1e-6)[0],
            rtol=1e-2,
            atol=1e-2,
        )


@pytest.mark.skipif(
    not current_platform.has_device_capability(100), reason="MXFP8 needs Blackwell"
)
@pytest.mark.parametrize("num_tokens", [0, 1, 17, 128, 129, 1024])
@pytest.mark.parametrize("q_size", [384, 1280, 1344])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_fused_q_kv_rmsnorm_quant_matches_separate(num_tokens, q_size, dtype):
    """Preserve native MXFP8 rounding and all swizzled scale padding bytes."""
    from vllm.model_executor.layers.quantization.utils.mxfp8_utils import (
        mxfp8_e4m3_quantize,
    )
    from vllm.models.deepseek_v4_1.common.ops.query_quant import (
        fused_q_kv_rmsnorm_quant,
    )

    torch.manual_seed(0)
    kv_size = 512
    x = torch.randn(num_tokens, q_size + kv_size, device="cuda", dtype=dtype)
    x[:1] = 0
    x[1:2] *= 1e-30
    qr, kv = x.split([q_size, kv_size], -1)
    qw = torch.randn(q_size, device="cuda", dtype=x.dtype)
    kvw = torch.randn(kv_size, device="cuda", dtype=x.dtype)
    eps = 1e-20
    result = fused_q_kv_rmsnorm_quant(qr, kv, qw, kvw, eps)
    assert result[0].orig_dtype == dtype
    assert result[0].orig_shape == qr.shape
    if not num_tokens:
        assert result[0].data.shape == qr.shape
        assert result[0].scale.numel() == 0
        assert result[1].shape == kv.shape
        return
    qr_ref, kv_ref = fused_q_kv_rmsnorm(qr, kv, qw, kvw, eps)
    q_ref, scale_ref = mxfp8_e4m3_quantize(qr_ref, is_sf_swizzled_layout=True)
    for replay in (False, True):
        if replay:
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                result = fused_q_kv_rmsnorm_quant(qr, kv, qw, kvw, eps)
            graph.replay()
        torch.testing.assert_close(
            result[0].data.view(torch.uint8), q_ref.view(torch.uint8), rtol=0, atol=0
        )
        torch.testing.assert_close(result[0].scale, scale_ref, rtol=0, atol=0)
        torch.testing.assert_close(result[1], kv_ref, rtol=0, atol=0)


@pytest.mark.skipif(
    not current_platform.has_device_capability(100), reason="MXFP8 needs Blackwell"
)
@pytest.mark.parametrize("backend", ["cute-dsl", "cutlass"])
@pytest.mark.parametrize("num_tokens", [1, 17, 256])
@pytest.mark.parametrize("projection", ["attention", "indexer"])
@pytest.mark.parametrize("bias", [False, True])
def test_shared_query_quant_preserves_projection(
    num_tokens, backend, projection, bias, monkeypatch
):
    """The native-checkpoint projection keeps identical GEMM outputs."""
    from dataclasses import replace
    from types import SimpleNamespace
    from typing import Any, cast

    from vllm.config import (
        CompilationConfig,
        CompilationMode,
        VllmConfig,
        set_current_vllm_config,
    )
    from vllm.model_executor.kernels.linear.mxfp8 import Mxfp8LinearLayerConfig
    from vllm.model_executor.kernels.linear.mxfp8.flashinfer import (
        FlashInferCutedslMxfp8LinearKernel,
        FlashInferCutlassMxfp8LinearKernel,
    )
    from vllm.model_executor.layers.fusion.quant_activation import (
        expose_input_quant_key,
    )
    from vllm.model_executor.layers.linear import ColumnParallelLinear, ReplicatedLinear
    from vllm.model_executor.layers.quantization.modelopt import (
        ModelOptLinearMethod,
    )
    from vllm.model_executor.layers.quantization.utils.mxfp8_utils import (
        mxfp8_e4m3_quantize,
    )
    from vllm.model_executor.layers.quantization.utils.quant_utils import kNvfp4Dynamic
    from vllm.models.deepseek_v4_1.attention import (
        DeepseekV4Attention,
        DeepseekV4Indexer,
    )
    from vllm.models.deepseek_v4_1.common.ops.query_quant import (
        can_fuse_query_quant,
        fused_q_kv_rmsnorm_quant,
    )
    from vllm.models.deepseek_v4_1.quant_config import DeepseekV4FP8Config

    cls: (
        type[FlashInferCutedslMxfp8LinearKernel]
        | type[FlashInferCutlassMxfp8LinearKernel]
    )
    if backend == "cute-dsl":
        cls = FlashInferCutedslMxfp8LinearKernel
    else:
        cls = FlashInferCutlassMxfp8LinearKernel
    supported, reason = cls.is_supported()
    if not supported:
        pytest.skip(reason)
    torch.manual_seed(0)
    with set_current_vllm_config(
        VllmConfig(compilation_config=CompilationConfig(mode=CompilationMode.NONE))
    ):
        kernel = cls(Mxfp8LinearLayerConfig())
        # Match the checkpoint's TP4 attention and replicated indexer widths.
        if projection == "attention":
            linear = ColumnParallelLinear.__new__(ColumnParallelLinear)
            output_size = 8192
        else:
            linear = ReplicatedLinear.__new__(ReplicatedLinear)
            output_size = 4096
        torch.nn.Module.__init__(linear)
        linear.bias = (
            torch.nn.Parameter(
                torch.randn(output_size, device="cuda", dtype=torch.bfloat16)
            )
            if bias
            else None
        )
        linear.skip_bias_add = False
        linear.return_bias = projection == "indexer"
        linear.gather_output = False
        w, ws = mxfp8_e4m3_quantize(
            torch.randn(output_size, 1280, device="cuda", dtype=torch.bfloat16)
        )
        linear.weight = torch.nn.Parameter(w, requires_grad=False)
        linear.weight_scale = torch.nn.Parameter(ws, requires_grad=False)
        quant_config = DeepseekV4FP8Config.from_config(
            {
                "quant_method": "fp8",
                "activation_scheme": "dynamic",
                "weight_block_size": [32, 32],
                "scale_fmt": "ue8m0",
                "expert_dtype": "fp4",
            }
        )
        method = quant_config.get_quant_method(linear, "model.layers.2.attn.wq_b")
        assert type(method) is ModelOptLinearMethod
        method.out_dtype = torch.bfloat16
        method.kernel = kernel
        linear.quant_method = method
        kernel.process_weights_after_loading(linear)
        expose_input_quant_key(linear, kernel)
        assert can_fuse_query_quant([linear, linear])
        x = torch.randn(num_tokens, 1792, device="cuda", dtype=torch.bfloat16)
        qr, kv = x.split([1280, 512], -1)
        qw = torch.randn(1280, device="cuda", dtype=x.dtype)
        kvw = torch.randn(512, device="cuda", dtype=x.dtype)
        qr_ref, _ = fused_q_kv_rmsnorm(qr, kv, qw, kvw, 1e-20)
        q, _ = fused_q_kv_rmsnorm_quant(qr, kv, qw, kvw, 1e-20)
        owner = cast(Any, SimpleNamespace(wq_b=linear))
        project = (
            DeepseekV4Attention._wq_b_proj
            if projection == "attention"
            else DeepseekV4Indexer._wq_b_proj
        )
        expected = project(owner, qr_ref)

        def reject_requantization(*args, **kwargs):
            raise AssertionError("pre-quantized input must skip quantization")

        monkeypatch.setattr(
            "vllm.model_executor.kernels.linear.mxfp8.flashinfer.mxfp8_e4m3_quantize",
            reject_requantization,
        )
        actual = project(owner, q)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            actual = project(owner, q)
        graph.replay()
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        assert not can_fuse_query_quant(
            [linear, torch.nn.Linear(1280, 256, bias=False)]
        )
        reshaped = project(
            owner, replace(q, orig_shape=torch.Size((1, num_tokens, 1280)))
        )
        torch.testing.assert_close(reshaped, expected.unsqueeze(0), rtol=0, atol=0)
        with pytest.raises(AssertionError, match="consumer kernel"):
            linear(replace(q, quant_key=kNvfp4Dynamic))
