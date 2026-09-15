# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from vllm.model_executor.layers.internvl_shuffle_layer_norm import (
    can_use_internvl_shuffle_layer_norm,
    internvl_shuffle_layer_norm,
)

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA is required"
)


def _reference(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    batch, height, width, channels = x.shape
    shuffled = x.view(batch, height, width // 2, channels * 2)
    shuffled = shuffled.permute(0, 2, 1, 3).contiguous()
    shuffled = shuffled.view(batch, width // 2, height // 2, channels * 4)
    shuffled = shuffled.permute(0, 2, 1, 3).contiguous()
    return F.layer_norm(
        shuffled,
        (shuffled.shape[-1],),
        weight,
        bias,
        eps,
    )


@requires_cuda
def test_v2_shuffle_channel_order() -> None:
    x = torch.arange(1 * 4 * 6 * 8, device="cuda", dtype=torch.float32)
    x = x.reshape(1, 4, 6, 8).to(torch.float16)
    weight = torch.linspace(0.5, 1.5, 32, device="cuda", dtype=torch.float16)
    bias = torch.linspace(-0.5, 0.5, 32, device="cuda", dtype=torch.float16)

    actual = internvl_shuffle_layer_norm(x, weight, bias, 1e-5)

    torch.testing.assert_close(actual, _reference(x, weight, bias, 1e-5))


@requires_cuda
def test_dispatch_accepts_supported_input() -> None:
    x = torch.empty(2, 14, 14, 64, device="cuda", dtype=torch.bfloat16)
    weight = torch.empty(256, device="cuda", dtype=torch.bfloat16)
    bias = torch.empty_like(weight)

    assert can_use_internvl_shuffle_layer_norm(x, weight, bias, "v2", 0.5)


def test_dispatch_rejects_cpu_input() -> None:
    x = torch.empty(1, 4, 6, 8, dtype=torch.float16)
    weight = torch.empty(32, dtype=torch.float16)
    bias = torch.empty_like(weight)

    assert not can_use_internvl_shuffle_layer_norm(x, weight, bias, "v2", 0.5)


@pytest.mark.parametrize(
    ("shape", "strided"),
    [
        ((1, 4, 6, 8), False),
        ((2, 14, 14, 64), True),
        ((2, 10, 8, 80), True),
        ((2, 32, 24, 128), True),
        ((1, 32, 32, 1024), False),
    ],
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@requires_cuda
def test_matches_reference(
    shape: tuple[int, int, int, int],
    strided: bool,
    dtype: torch.dtype,
) -> None:
    torch.manual_seed(0)
    batch, height, width, channels = shape
    if strided:
        with_class_token = torch.randn(
            batch,
            height * width + 1,
            channels,
            device="cuda",
            dtype=dtype,
        )
        x = with_class_token[:, 1:].view(batch, height, width, channels)
        assert not x.is_contiguous()
    else:
        x = torch.randn(shape, device="cuda", dtype=dtype)

    output_dim = 4 * channels
    weight = torch.randn(output_dim, device="cuda", dtype=dtype)
    bias = torch.randn(output_dim, device="cuda", dtype=dtype)
    expected = _reference(x, weight, bias, 1e-5)
    actual = internvl_shuffle_layer_norm(x, weight, bias, 1e-5)

    tolerance = 1e-3 if dtype == torch.float16 else 1e-2
    torch.testing.assert_close(actual, expected, rtol=tolerance, atol=tolerance)


@requires_cuda
def test_supports_torch_compile() -> None:
    x = torch.randn(2, 14, 14, 64, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(256, device="cuda", dtype=torch.bfloat16)
    bias = torch.randn_like(weight)
    compiled = torch.compile(internvl_shuffle_layer_norm, fullgraph=True)

    expected = internvl_shuffle_layer_norm(x, weight, bias, 1e-5)
    actual = compiled(x, weight, bias, 1e-5)

    torch.testing.assert_close(actual, expected)


@requires_cuda
def test_supports_cuda_graph_replay() -> None:
    x = torch.randn(2, 14, 14, 64, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(256, device="cuda", dtype=torch.bfloat16)
    bias = torch.randn_like(weight)
    for _ in range(3):
        internvl_shuffle_layer_norm(x, weight, bias, 1e-5)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = internvl_shuffle_layer_norm(x, weight, bias, 1e-5)

    x.copy_(torch.randn_like(x))
    expected = internvl_shuffle_layer_norm(x, weight, bias, 1e-5)
    graph.replay()

    torch.testing.assert_close(captured, expected)


@requires_cuda
def test_extract_feature_uses_fused_path(monkeypatch: pytest.MonkeyPatch) -> None:
    from vllm.model_executor.models.internvl import InternVLChatModel

    channels = 64
    hidden_size = 96
    batch = 2
    height = width = 14
    vision_output = torch.randn(
        batch,
        height * width + 1,
        channels,
        device="cuda",
        dtype=torch.bfloat16,
    )

    class VisionStub(nn.Module):
        def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
            return vision_output

    model = InternVLChatModel.__new__(InternVLChatModel)
    nn.Module.__init__(model)
    model.vision_model = VisionStub()
    model.downsample_ratio = 0.5
    model.ps_version = "v2"
    model.mlp1 = nn.Sequential(
        nn.LayerNorm(4 * channels, device="cuda", dtype=torch.bfloat16),
        nn.Linear(
            4 * channels,
            hidden_size,
            device="cuda",
            dtype=torch.bfloat16,
        ),
        nn.GELU(),
        nn.Linear(
            hidden_size,
            hidden_size,
            device="cuda",
            dtype=torch.bfloat16,
        ),
    )

    def fail_unfused_path(*args, **kwargs):
        raise AssertionError("unfused pixel shuffle was called")

    monkeypatch.setattr(model, "pixel_shuffle", fail_unfused_path)
    output = model.extract_feature(
        torch.empty(batch, 3, 1, 1, device="cuda", dtype=torch.bfloat16)
    )

    assert output.shape == (batch, height * width // 4, hidden_size)
