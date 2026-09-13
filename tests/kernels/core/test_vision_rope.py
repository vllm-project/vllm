# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for fused vision rotary embeddings."""

import pytest
import torch

from vllm.model_executor.layers.rotary_embedding.vision import (
    apply_fused_qk_complex_rope,
    can_use_fused_qk_complex_rope,
)
from vllm.platforms import current_platform


def _complex_rope_reference(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    output = torch.view_as_real(x_complex * freqs_cis.unsqueeze(-2)).flatten(-2)
    return output.to(x.dtype)


def _make_packed_inputs(
    num_tokens: int,
    num_heads: int,
    head_dim: int,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    qkv = torch.randn(
        num_tokens,
        3,
        num_heads,
        head_dim,
        dtype=dtype,
        device="cuda",
    )
    query, key, _ = torch.unbind(qkv, dim=1)
    angles = torch.randn(num_tokens, head_dim // 2, device="cuda")
    freqs_cis = torch.polar(torch.ones_like(angles), angles)
    return qkv, query, key, freqs_cis


requires_sm90 = pytest.mark.skipif(
    not (current_platform.is_cuda() and current_platform.has_device_capability(90)),
    reason="The fused vision RoPE kernel requires NVIDIA SM90 or newer.",
)


@requires_sm90
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("num_tokens", [1, 257, 4096])
def test_fused_qk_complex_rope_matches_reference(
    dtype: torch.dtype,
    num_tokens: int,
) -> None:
    """The fused path matches FP32 complex multiplication for packed QKV."""
    torch.manual_seed(0)
    _, query, key, freqs_cis = _make_packed_inputs(
        num_tokens, num_heads=12, head_dim=128, dtype=dtype
    )

    assert query.stride(0) == 3 * query.shape[1] * query.shape[2]
    query_out, key_out = apply_fused_qk_complex_rope(query, key, freqs_cis)

    atol = 2 * torch.finfo(dtype).eps
    torch.testing.assert_close(
        query_out,
        _complex_rope_reference(query, freqs_cis),
        atol=atol,
        rtol=0,
    )
    torch.testing.assert_close(
        key_out,
        _complex_rope_reference(key, freqs_cis),
        atol=atol,
        rtol=0,
    )
    assert query_out.is_contiguous()
    assert key_out.is_contiguous()


@requires_sm90
def test_fused_qk_complex_rope_cuda_graph_replay() -> None:
    """Captured execution reads updated packed-QKV inputs on replay."""
    qkv, query, key, freqs_cis = _make_packed_inputs(
        257, num_heads=12, head_dim=128, dtype=torch.bfloat16
    )

    warmup_stream = torch.cuda.Stream()
    warmup_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(warmup_stream):
        apply_fused_qk_complex_rope(query, key, freqs_cis)
    torch.cuda.current_stream().wait_stream(warmup_stream)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        query_out, key_out = apply_fused_qk_complex_rope(query, key, freqs_cis)

    qkv.copy_(torch.randn_like(qkv))
    graph.replay()

    atol = 2 * torch.finfo(query.dtype).eps
    torch.testing.assert_close(
        query_out,
        _complex_rope_reference(query, freqs_cis),
        atol=atol,
        rtol=0,
    )
    torch.testing.assert_close(
        key_out,
        _complex_rope_reference(key, freqs_cis),
        atol=atol,
        rtol=0,
    )


def test_fused_qk_complex_rope_rejects_unsupported_inputs() -> None:
    query = torch.empty(2, 4, 8, dtype=torch.float32)
    key = torch.empty_like(query)
    freqs_cis = torch.empty(2, 4, dtype=torch.complex64)

    assert not can_use_fused_qk_complex_rope(query, key, freqs_cis)
    with pytest.raises(ValueError, match="Unsupported fused vision RoPE inputs"):
        apply_fused_qk_complex_rope(query, key, freqs_cis)
