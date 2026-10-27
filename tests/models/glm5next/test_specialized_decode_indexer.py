# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The specialized decode's fused DSA indexer kernels against PyTorch and the
stock Hadamard quant kernel."""

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from vllm.platforms import current_platform

HIDDEN, KEY, GATE, DIM = 4096, 128, 128, 128

pytestmark = pytest.mark.skipif(
    not current_platform.is_cuda(), reason="the indexer kernels are CUDA kernels"
)


@pytest.mark.parametrize("tokens", [1, 3, 16, 40])
@pytest.mark.parametrize("heads", [16, 32])
def test_projection_matches_linear(tokens, heads):
    from vllm.models.glm5next.nvidia.specialized_decode.indexer import (
        IndexerProjection,
    )

    g = torch.Generator(device="cuda").manual_seed(tokens)
    wk = (
        torch.randn(KEY + heads, HIDDEN, generator=g, device="cuda") * 0.02
    ).bfloat16()
    gate = (torch.randn(GATE, HIDDEN, generator=g, device="cuda") * 0.02).bfloat16()
    indexer = SimpleNamespace(
        head_dim=KEY,
        n_head=heads,
        index_kpool_compress_gate=SimpleNamespace(data=gate),
        wk_weights_proj=SimpleNamespace(weight=SimpleNamespace(data=wk)),
    )
    x = torch.randn(tokens, HIDDEN, generator=g, device="cuda").bfloat16()
    key, weights, gate_score = IndexerProjection(indexer)(x)

    # BF16 products are exact in FP32: only the summation order differs.
    full = x.float() @ torch.cat([wk, gate]).float().t()
    key_ref = full[:, :KEY].bfloat16()
    torch.testing.assert_close(key, key_ref, atol=2**-8, rtol=2**-7)
    torch.testing.assert_close(
        weights, full[:, KEY : KEY + heads], atol=1e-5, rtol=1e-5
    )
    torch.testing.assert_close(
        gate_score, full[:, KEY + heads :].bfloat16(), atol=2**-8, rtol=2**-7
    )


@pytest.mark.parametrize("tokens", [1, 5])
@pytest.mark.parametrize("heads", [16, 32])
def test_query_epilogue_matches_unfused(tokens, heads):
    from vllm.models.glm5next.nvidia.ops.kpool_compress import fwht128_quant_fp8
    from vllm.models.glm5next.nvidia.specialized_decode.indexer_query import (
        indexer_query,
    )

    g = torch.Generator(device="cuda").manual_seed(tokens)
    q = torch.randn(tokens, heads * DIM, generator=g, device="cuda").bfloat16()
    k = torch.randn(tokens, DIM, generator=g, device="cuda").bfloat16()
    w = torch.randn(tokens, heads, generator=g, device="cuda")
    norm = SimpleNamespace(
        weight=1 + 0.1 * torch.randn(DIM, generator=g, device="cuda"),
        bias=0.1 * torch.randn(DIM, generator=g, device="cuda"),
        eps=1e-6,
    )
    scale = DIM**-0.5 * heads**-0.5
    q_fp8, weights, key = indexer_query(q, k, w, norm, scale, 32)

    ref_q, ref_scale = fwht128_quant_fp8(q.view(-1, DIM))
    ref_scale = ref_scale.view(tokens, heads)
    assert torch.equal(
        q_fp8[:, :heads].view(torch.uint8),
        ref_q.view(tokens, heads, DIM).view(torch.uint8),
    )
    assert not q_fp8[:, heads:].float().any()
    assert torch.equal(weights[:, :heads], w * ref_scale * scale)
    assert not weights[:, heads:].any()
    ref_key = F.layer_norm(k.float(), (DIM,), norm.weight, norm.bias, norm.eps)
    torch.testing.assert_close(key, ref_key.bfloat16(), atol=2**-7, rtol=2**-7)


@pytest.mark.parametrize("tokens", [1, 5, 16])
@pytest.mark.parametrize("heads", [16, 32])
def test_fused_query_matches_pair(tokens, heads):
    """The projection kernel with the query epilogue inside writes what the
    projection and ``indexer_query`` write; repeated calls advance its grid
    barrier."""
    from vllm.models.glm5next.nvidia.specialized_decode.indexer import (
        IndexerProjection,
    )
    from vllm.models.glm5next.nvidia.specialized_decode.indexer_query import (
        indexer_query,
    )

    g = torch.Generator(device="cuda").manual_seed(tokens)
    wk = (
        torch.randn(KEY + heads, HIDDEN, generator=g, device="cuda") * 0.02
    ).bfloat16()
    gate = (torch.randn(GATE, HIDDEN, generator=g, device="cuda") * 0.02).bfloat16()
    projection = IndexerProjection(
        SimpleNamespace(
            head_dim=KEY,
            n_head=heads,
            index_kpool_compress_gate=SimpleNamespace(data=gate),
            wk_weights_proj=SimpleNamespace(weight=SimpleNamespace(data=wk)),
        )
    )
    norm = SimpleNamespace(
        weight=1 + 0.1 * torch.randn(DIM, generator=g, device="cuda"),
        bias=0.1 * torch.randn(DIM, generator=g, device="cuda"),
        eps=1e-6,
    )
    scale = DIM**-0.5 * heads**-0.5
    for _ in range(3):
        x = torch.randn(tokens, HIDDEN, generator=g, device="cuda").bfloat16()
        q = torch.randn(tokens, heads * DIM, generator=g, device="cuda").bfloat16()
        k, w, gate_score = projection(x)
        ref_q, ref_w, ref_key = indexer_query(q, k, w, norm, scale, 32)
        q_fp8, weights, key, fused_gate = projection.fused(x, q, norm, scale, 32)
        assert torch.equal(q_fp8.view(torch.uint8), ref_q.view(torch.uint8))
        assert torch.equal(weights, ref_w)
        assert torch.equal(fused_gate, gate_score)
        torch.testing.assert_close(key, ref_key, atol=2**-7, rtol=2**-7)


@pytest.mark.parametrize("fused", [False, True])
def test_bf16_head_weights_round_fp32(fused):
    """BF16 head weights (for the SM100 MQA logits) are the FP32 ones rounded
    as ``.to(torch.bfloat16)`` rounds them, padding included."""
    from vllm.models.glm5next.nvidia.specialized_decode.indexer import (
        IndexerProjection,
    )
    from vllm.models.glm5next.nvidia.specialized_decode.indexer_query import (
        indexer_query,
    )

    tokens, heads = 2, 16
    g = torch.Generator(device="cuda").manual_seed(7)
    wk = (
        torch.randn(KEY + heads, HIDDEN, generator=g, device="cuda") * 0.02
    ).bfloat16()
    gate = (torch.randn(GATE, HIDDEN, generator=g, device="cuda") * 0.02).bfloat16()
    projection = IndexerProjection(
        SimpleNamespace(
            head_dim=KEY,
            n_head=heads,
            index_kpool_compress_gate=SimpleNamespace(data=gate),
            wk_weights_proj=SimpleNamespace(weight=SimpleNamespace(data=wk)),
        )
    )
    norm = SimpleNamespace(
        weight=1 + 0.1 * torch.randn(DIM, generator=g, device="cuda"),
        bias=0.1 * torch.randn(DIM, generator=g, device="cuda"),
        eps=1e-6,
    )
    scale = DIM**-0.5 * heads**-0.5
    x = torch.randn(tokens, HIDDEN, generator=g, device="cuda").bfloat16()
    q = torch.randn(tokens, heads * DIM, generator=g, device="cuda").bfloat16()
    outputs = []
    for dtype in (torch.float32, torch.bfloat16):
        if fused:
            outputs.append(projection.fused(x, q, norm, scale, 32, dtype)[1])
        else:
            k, w, _ = projection(x)
            outputs.append(indexer_query(q, k, w, norm, scale, 32, dtype)[1])
    fp32, bf16 = outputs
    assert bf16.dtype == torch.bfloat16
    assert torch.equal(bf16.view(torch.int16), fp32.bfloat16().view(torch.int16))
