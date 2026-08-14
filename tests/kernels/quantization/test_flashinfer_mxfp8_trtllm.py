# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Generator

import pytest
import torch
import torch.nn.functional as F

from vllm.model_executor.kernels.linear import (
    FlashInferTrtllmMxfp8LinearKernel,
    Mxfp8LinearLayerConfig,
)
from vllm.model_executor.kernels.linear.mxfp8.flashinfer import (
    MXFP8_TRTLLM_LAYOUT_ENV,
    MXFP8_TRTLLM_SWITCH_M_ENV,
    _mxfp8_trtllm_layout_config,
)
from vllm.platforms import current_platform
from vllm.utils import flashinfer as vllm_flashinfer
from vllm.utils.flashinfer import has_flashinfer

if not (
    current_platform.is_cuda()
    and any(current_platform.is_device_capability(cc) for cc in (100, 103, 107))
    and has_flashinfer()
):
    pytest.skip(
        reason="FlashInfer TRTLLM MXFP8 requires SM100, SM103, or SM107",
        allow_module_level=True,
    )


def _make_layer(weight: torch.Tensor) -> torch.nn.Module:
    from flashinfer import SfLayout, mxfp8_quantize

    weight_mxfp8, weight_scale = mxfp8_quantize(
        weight,
        sf_swizzle_layout=SfLayout.layout_linear,
    )
    layer = torch.nn.Module()
    layer.weight = torch.nn.Parameter(weight_mxfp8, requires_grad=False)
    layer.weight_scale = torch.nn.Parameter(
        weight_scale.view(weight.shape[0], weight.shape[1] // 32),
        requires_grad=False,
    )
    return layer


@pytest.fixture(autouse=True)
def clear_mxfp8_trtllm_layout_config() -> Generator[None, None, None]:
    _mxfp8_trtllm_layout_config.cache_clear()
    yield
    _mxfp8_trtllm_layout_config.cache_clear()


@pytest.mark.parametrize("shape", [(1, 130, 256), (7, 256, 512), (128, 130, 768)])
@torch.inference_mode()
def test_flashinfer_trtllm_mxfp8_linear_numerics(
    shape: tuple[int, int, int],
) -> None:
    torch.manual_seed(0)
    m, n, k = shape
    x = torch.randn((m, k), dtype=torch.bfloat16, device="cuda")
    weight = torch.randn((n, k), dtype=torch.bfloat16, device="cuda")
    layer = _make_layer(weight)
    kernel = FlashInferTrtllmMxfp8LinearKernel(Mxfp8LinearLayerConfig())
    kernel.process_weights_after_loading(layer)

    output = kernel.apply_weights(layer, x)
    reference = torch.mm(x, weight.t())
    similarity = F.cosine_similarity(
        output.float().flatten(), reference.float().flatten(), dim=0
    )

    assert output.shape == (m, n)
    assert output.is_contiguous()
    assert similarity.item() > 0.98


@pytest.mark.parametrize(
    ("layout", "m"),
    [("128x4", 512), ("adaptive", 256), ("adaptive", 257)],
)
@torch.inference_mode()
def test_flashinfer_trtllm_mxfp8_alternate_layout_numerics(
    monkeypatch, layout: str, m: int
) -> None:
    monkeypatch.setenv(MXFP8_TRTLLM_LAYOUT_ENV, layout)
    monkeypatch.setenv(MXFP8_TRTLLM_SWITCH_M_ENV, "256")
    torch.manual_seed(0)
    n, k = 256, 512
    x = torch.randn((m, k), dtype=torch.bfloat16, device="cuda")
    weight = torch.randn((n, k), dtype=torch.bfloat16, device="cuda")
    layer = _make_layer(weight)
    kernel = FlashInferTrtllmMxfp8LinearKernel(Mxfp8LinearLayerConfig())
    kernel.process_weights_after_loading(layer)

    output = kernel.apply_weights(layer, x)
    reference = torch.mm(x, weight.t())
    similarity = F.cosine_similarity(
        output.float().flatten(), reference.float().flatten(), dim=0
    )

    assert output.shape == (m, n)
    assert output.is_contiguous()
    assert similarity.item() > 0.98


@torch.inference_mode()
def test_flashinfer_trtllm_mxfp8_custom_ops() -> None:
    x = torch.randn((7, 512), dtype=torch.bfloat16, device="cuda")
    weight = torch.randn((256, 512), dtype=torch.bfloat16, device="cuda")
    layer = _make_layer(weight)
    kernel = FlashInferTrtllmMxfp8LinearKernel(Mxfp8LinearLayerConfig())
    kernel.process_weights_after_loading(layer)

    torch.library.opcheck(
        torch.ops.vllm.flashinfer_mxfp8_quantize_8x4.default,
        (x,),
    )
    torch.library.opcheck(
        torch.ops.vllm.flashinfer_mxfp8_quantize_128x4.default,
        (x,),
    )
    x_mxfp8, x_scale = vllm_flashinfer.flashinfer_mxfp8_quantize_8x4(x)
    # SchemaCheckMode compares inputs with allclose, which CUDA does not
    # implement for float8. The numerical tests above guard input mutation.
    torch.library.opcheck(
        torch.ops.vllm.mm_mxfp8.default,
        (
            x_mxfp8,
            layer.weight.t(),
            x_scale,
            layer.weight_scale,
            torch.bfloat16,
            "trtllm",
            True,
        ),
        test_utils=(
            "test_autograd_registration",
            "test_faketensor",
            "test_aot_dispatch_dynamic",
        ),
    )
    torch.library.opcheck(
        torch.ops.vllm.mxfp8_trtllm_adaptive_linear.default,
        (x, layer.weight, layer.weight_scale, 256),
        test_utils=(
            "test_autograd_registration",
            "test_faketensor",
            "test_aot_dispatch_dynamic",
        ),
    )


@torch.inference_mode()
@pytest.mark.parametrize(
    ("layout", "m"),
    [("8x4", 7), ("128x4", 512), ("adaptive", 7), ("adaptive", 512)],
)
def test_flashinfer_trtllm_mxfp8_linear_cuda_graph(
    monkeypatch, layout: str, m: int
) -> None:
    monkeypatch.setenv(MXFP8_TRTLLM_LAYOUT_ENV, layout)
    monkeypatch.setenv(MXFP8_TRTLLM_SWITCH_M_ENV, "256")
    torch.manual_seed(0)
    n, k = 130, 512
    weight = torch.randn((n, k), dtype=torch.bfloat16, device="cuda")
    layer = _make_layer(weight)
    kernel = FlashInferTrtllmMxfp8LinearKernel(Mxfp8LinearLayerConfig())
    kernel.process_weights_after_loading(layer)

    static_x = torch.randn((m, k), dtype=torch.bfloat16, device="cuda")
    kernel.apply_weights(layer, static_x)
    torch.accelerator.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        graph_output = kernel.apply_weights(layer, static_x)

    new_x = torch.randn_like(static_x)
    static_x.copy_(new_x)
    graph.replay()
    eager_output = kernel.apply_weights(layer, new_x)

    torch.testing.assert_close(graph_output, eager_output, rtol=0, atol=0)
