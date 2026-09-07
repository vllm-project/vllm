# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
import torch.nn.functional as F

from vllm.model_executor.models.deepencoder import (
    _flex_attention_with_decomposed_rel_pos,
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
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_flex_attention_matches_dense_decomposed_bias(dtype: torch.dtype) -> None:
    torch.manual_seed(0)
    batch, heads, height, width, dim = 1, 2, 4, 4, 16
    tokens = height * width
    q = torch.randn(batch, heads, tokens, dim, device="cuda", dtype=dtype)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    rel_h = torch.randn(batch, heads, tokens, height, 1, device="cuda", dtype=dtype)
    rel_w = torch.randn(batch, heads, tokens, 1, width, device="cuda", dtype=dtype)
    expected = F.scaled_dot_product_attention(
        q, k, v, attn_mask=(rel_h + rel_w).flatten(-2)
    )

    actual = _flex_attention_with_decomposed_rel_pos(q, k, v, rel_h, rel_w, width)

    tolerance = 1e-4 if dtype == torch.float32 else 2e-2
    torch.testing.assert_close(actual, expected, atol=tolerance, rtol=tolerance)


def test_rel_pos_attention_cpu_fallback() -> None:
    from vllm.model_executor.models.deepencoder import RelPosAttention

    torch.manual_seed(0)
    layer = RelPosAttention(
        dim=32,
        num_heads=2,
        use_rel_pos=True,
        use_flex_attention=True,
        input_size=(4, 4),
    ).eval()
    layer.rel_pos_h.data.normal_()
    layer.rel_pos_w.data.normal_()

    output = layer(torch.randn(1, 4, 4, 32))

    assert output.shape == (1, 4, 4, 32)
    assert torch.isfinite(output).all()


def test_block_enables_flex_attention_only_for_global_attention() -> None:
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

    assert global_block.attn.use_flex_attention
    assert not window_block.attn.use_flex_attention


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
            use_flex_attention=True,
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
