# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
import torch.nn.functional as F

import vllm.model_executor.models.deepencoder as deepencoder
from vllm.model_executor.kernels.deepencoder_attention import (
    deepencoder_rel_pos_attention,
)


@pytest.mark.parametrize(
    ("repeats", "expected"),
    [
        (1, [1]),
        (5, [1, 1, 1, 1, 1]),
        (7, [2, 2, 1, 1, 1]),
        (51, [11, 10, 10, 10, 10]),
    ],
)
def test_benchmark_distributes_all_repeats(repeats: int, expected: list[int]) -> None:
    from benchmarks.kernels.benchmark_deepencoder_rel_pos_attention import (
        repeat_counts,
    )

    assert repeat_counts(repeats) == expected


@pytest.mark.parametrize("repeats", [0, -1])
def test_benchmark_rejects_non_positive_repeats(repeats: int) -> None:
    from benchmarks.kernels.benchmark_deepencoder_rel_pos_attention import (
        repeat_counts,
    )

    with pytest.raises(ValueError, match="repeats must be positive"):
        repeat_counts(repeats)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize(("height", "width"), [(2, 3), (5, 7)])
def test_triton_attention_matches_dense_decomposed_bias(
    dtype: torch.dtype,
    height: int,
    width: int,
) -> None:
    torch.manual_seed(0)
    batch, heads, dim = 1, 2, 64
    tokens = height * width
    q = torch.randn(batch, heads, tokens, dim, device="cuda", dtype=dtype)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    rel_h = torch.randn(batch, heads, tokens, height, device="cuda", dtype=dtype)
    rel_w = torch.randn(batch, heads, tokens, width, device="cuda", dtype=dtype)
    expected = F.scaled_dot_product_attention(
        q,
        k,
        v,
        attn_mask=(rel_h.unsqueeze(-1) + rel_w.unsqueeze(-2)).flatten(-2),
    )

    actual = deepencoder_rel_pos_attention(
        q,
        k,
        v,
        rel_h,
        rel_w,
        height,
        width,
        dim**-0.5,
    )

    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, expected, atol=2e-2, rtol=2e-2)


def test_rel_pos_attention_cpu_fallback() -> None:
    from vllm.model_executor.models.deepencoder import RelPosAttention

    torch.manual_seed(0)
    layer = RelPosAttention(
        dim=32,
        num_heads=2,
        use_rel_pos=True,
        use_triton_attention=True,
        input_size=(4, 4),
    ).eval()
    layer.rel_pos_h.data.normal_()
    layer.rel_pos_w.data.normal_()

    output = layer(torch.randn(1, 4, 4, 32))

    assert output.shape == (1, 4, 4, 32)
    assert torch.isfinite(output).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_global_block_dispatches_to_triton(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    called = False

    def fake_attention(q, k, v, rel_h, rel_w, height, width, scale):
        nonlocal called
        called = True
        return torch.zeros_like(q)

    monkeypatch.setattr(
        deepencoder,
        "deepencoder_rel_pos_attention",
        fake_attention,
    )
    attention = (
        deepencoder.RelPosAttention(
            dim=128,
            num_heads=2,
            use_rel_pos=True,
            use_triton_attention=True,
            input_size=(4, 4),
        )
        .cuda()
        .to(torch.bfloat16)
        .eval()
    )
    inputs = torch.randn(1, 4, 4, 128, device="cuda", dtype=torch.bfloat16)

    attention(inputs)

    assert called


def test_block_enables_triton_attention_only_for_global_attention() -> None:
    from vllm.model_executor.models.deepencoder import Block

    global_block = Block(
        dim=32,
        num_heads=2,
        use_rel_pos=True,
        window_size=0,
        input_size=(64, 64),
    )
    window_block = Block(
        dim=32,
        num_heads=2,
        use_rel_pos=True,
        window_size=14,
        input_size=(64, 64),
    )

    assert global_block.attn.use_triton_attention
    assert not window_block.attn.use_triton_attention


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.parametrize("spatial_size", [14, 64])
def test_rel_pos_attention_cuda_graph(spatial_size: int) -> None:
    from vllm.model_executor.models.deepencoder import RelPosAttention

    torch.manual_seed(0)
    layer = (
        RelPosAttention(
            dim=768,
            num_heads=12,
            use_rel_pos=True,
            use_triton_attention=True,
            input_size=(spatial_size, spatial_size),
        )
        .cuda()
        .to(torch.bfloat16)
        .eval()
    )
    layer.rel_pos_h.data.normal_(std=0.02)
    layer.rel_pos_w.data.normal_(std=0.02)
    inputs = torch.randn(
        1,
        spatial_size,
        spatial_size,
        768,
        device="cuda",
        dtype=torch.bfloat16,
    )
    for _ in range(3):
        eager = layer(inputs)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = layer(inputs)

    graph.replay()
    torch.cuda.synchronize()

    torch.testing.assert_close(captured, eager, atol=2e-2, rtol=2e-2)
