# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Step5 preparation, paged attention and summary contracts on CUDA/HIP."""

import pytest
import torch
from vllm.platforms import current_platform

pytestmark = pytest.mark.skipif(
    not (current_platform.is_cuda() or current_platform.is_rocm()),
    reason="Step5 Triton tests require a CUDA or ROCm GPU",
)


@pytest.mark.parametrize("rotary_dim", [64, 192])
@pytest.mark.parametrize("fused_summary", [False, True])
def test_joint_and_standalone_qkv_preserve_paged_cache(rotary_dim, fused_summary):
    from vllm.models.step5.nvidia.ops.indexer_norm_rope import indexer_norm_rope
    from vllm.models.step5.nvidia.ops.qkv_indexer_norm_rope import (
        qkv_indexer_norm_rope,
    )
    from vllm.models.step5.nvidia.ops.qkv_norm_rope_cache import qkv_norm_rope_cache

    torch.manual_seed(904)
    device, dtype = "cuda", torch.bfloat16
    n, qh, kh, dim, block = 4, 16, 1, 192, 16
    qkv = torch.randn(n, (qh + 2 * kh) * dim, device=device, dtype=dtype)
    qw = torch.randn(dim, device=device) * 0.1
    kw = torch.randn_like(qw) * 0.1
    angles = torch.randn(32, rotary_dim // 2, device=device)
    rope = torch.cat([angles.cos(), angles.sin()], -1).to(dtype)
    positions = torch.tensor([0, 8, 16, 24], device=device)
    slots = torch.tensor([0, 24, 32, -1], device=device)
    cache = torch.full((4, block, kh, 2 * dim), 13, device=device, dtype=dtype)
    kc, vc = cache.split(dim, -1)
    actual = qkv_norm_rope_cache(
        qkv,
        qw,
        kw,
        rope,
        positions,
        slots,
        kc,
        vc,
        qh,
        kh,
        rotary_dim,
        1e-5,
        True,
    )

    def reference(x, weight):
        x = x.reshape(n, -1, dim).float()
        y = (
            (x * torch.rsqrt(x.square().mean(-1, keepdim=True) + 1e-5) * (1 + weight))
            .to(dtype)
            .float()
        )
        half = rotary_dim // 2
        cos = rope[positions, :half].float()[:, None]
        sin = rope[positions, half:].float()[:, None]
        low, high = y[..., :half], y[..., half:rotary_dim]
        return (
            torch.cat(
                [low * cos - high * sin, high * cos + low * sin, y[..., rotary_dim:]],
                -1,
            )
            .to(dtype)
            .flatten(1)
        )

    q, k, v = qkv.split([qh * dim, kh * dim, kh * dim], -1)
    for got, ref in zip(actual, [reference(q, qw), reference(k, kw), v]):
        torch.testing.assert_close(got, ref, atol=0.03125, rtol=0.02)
    expected_cache = torch.full_like(cache, 13)
    for row, slot in enumerate([0, 24, 32]):
        expected_cache[slot // block, slot % block, :, :dim] = actual[1][row].view(
            kh, dim
        )
        expected_cache[slot // block, slot % block, :, dim:] = v[row].view(kh, dim)
    torch.testing.assert_close(cache, expected_cache, atol=0, rtol=0)

    iq = torch.randn(n, 4 * 256, device=device, dtype=dtype)
    ik = torch.randn(n, 256, device=device, dtype=dtype)
    weights = torch.randn(n, 4, device=device)
    iqw = torch.randn(256, device=device) * 0.1
    ikw, ikb = torch.randn_like(iqw) * 0.1, torch.randn_like(iqw) * 0.1
    angle = torch.randn(32, 16, device=device)
    irope = torch.cat([angle.cos(), angle.sin()], -1).to(dtype)
    args = (iq, ik, weights, iqw, ikw, ikb, irope, positions, 1e-5, True)
    index_ref = indexer_norm_rope(*args)
    cache.fill_(13)
    extra = {}
    if fused_summary:
        from vllm.models.step5.nvidia.sparse_summary_cache import (
            Step5SparseSummaryCache,
        )

        summary = Step5SparseSummaryCache(4, block, torch.device(device), n)
        summary_ref = Step5SparseSummaryCache(4, block, torch.device(device), n)
        z = torch.randn(n, 1, 256, device=device, dtype=dtype)
        extra = dict(decode_cache=summary, decode_z=z, decode_slots=slots)
        summary_ref.update_decode(index_ref[1], z, slots)
    joint = qkv_indexer_norm_rope(
        *args,
        main_inputs=(
            qkv,
            qw,
            kw,
            rope,
            slots,
            kc,
            vc,
            qh,
            kh,
            rotary_dim,
            1e-5,
            1.0,
            None,
            None,
        ),
        **extra,
    )
    if fused_summary:
        torch.testing.assert_close(summary.mean, summary_ref.mean, atol=0, rtol=0)
        torch.testing.assert_close(
            summary.numerator, summary_ref.numerator, atol=0, rtol=0
        )
    for got, ref in zip(joint[:3], index_ref):
        torch.testing.assert_close(got.float(), ref.float(), atol=0, rtol=0)
    for got, ref in zip(joint[3:], actual):
        torch.testing.assert_close(got, ref, atol=0, rtol=0)
    torch.testing.assert_close(cache, expected_cache, atol=0, rtol=0)


@pytest.mark.parametrize("splits", [1, 2])
def test_gated_swa_and_sparse_decode_match_causal_reference(splits):
    from vllm.models.step5.nvidia.ops.sparse_decode_attention import (
        sparse_decode_attention,
    )
    from vllm.models.step5.nvidia.ops.swa_decode import swa_decode_gated

    torch.manual_seed(7)
    device, dtype = "cuda", torch.bfloat16
    block, heads, dim = 16, 4, 192
    # Non-contiguous physical pages, a partial page, and an inactive request.
    cache = torch.randn(4, 1, block, 2 * dim, device=device, dtype=dtype)
    kc, vc = cache.transpose(1, 2).split(dim, -1)
    table = torch.tensor([[2, 0], [3, 1]], device=device, dtype=torch.int32)
    lens = torch.tensor([23, 0], device=device, dtype=torch.int32)
    query = torch.randn(2, heads, dim, device=device, dtype=dtype)
    gate = torch.randn(2, heads, device=device, dtype=dtype)
    swa = torch.empty_like(query)
    swa_decode_gated(
        query,
        kc,
        vc,
        table,
        lens,
        gate,
        swa,
        scale=dim**-0.5,
        window=16,
        query_len=1,
        splits=splits,
    )

    def reference(first):
        key = torch.cat([kc[2, :, 0], kc[0, :7, 0]])[first:].float()
        value = torch.cat([vc[2, :, 0], vc[0, :7, 0]])[first:].float()
        probs = (query[0].float() @ key.T * dim**-0.5).softmax(-1)
        attention = (probs @ value).to(dtype)
        return attention * gate[0].sigmoid()[:, None]

    torch.testing.assert_close(swa[0], reference(7), atol=0.02, rtol=0.02)
    torch.testing.assert_close(swa[1], torch.zeros_like(swa[1]), atol=0, rtol=0)
    regions = torch.tensor([[0, 1, 2], [-1, -1, -1]], device=device)
    positions = torch.tensor([22, 0], device=device)
    active = torch.tensor([True, False], device=device)
    sparse = torch.empty_like(query)
    sparse_decode_attention(
        query,
        cache,
        regions,
        table,
        positions,
        active,
        sparse,
        scale=dim**-0.5,
        splits=splits,
        gate=gate,
    )
    torch.testing.assert_close(sparse[0], reference(0), atol=0.02, rtol=0.02)
    torch.testing.assert_close(sparse[1], torch.zeros_like(sparse[1]), atol=0, rtol=0)


@pytest.mark.parametrize("heads", [4, 8, 16])
def test_summary_scores_use_all_local_heads(heads):
    from vllm.models.step5.nvidia.ops.sparse_indexer_scores import decode_summary_scores

    torch.manual_seed(91)
    query = torch.randn(2, 1, heads, 256, device="cuda").to(torch.float8_e4m3fn)
    weights = torch.randn(2, 1, heads, device="cuda")
    keys = torch.randn(3, 2, 256, device="cuda").to(torch.float8_e4m3fn)
    table = torch.tensor([[2, 0, 1], [1, 2, 0]], device="cuda", dtype=torch.int32)
    slots = torch.tensor([0, 16], device="cuda")
    lens = torch.tensor([17, 17], device="cuda", dtype=torch.int32)
    scores, metadata = decode_summary_scores(
        query,
        weights,
        keys.view(torch.uint8),
        table,
        slots,
        lens,
        6,
        num_sms=8,
    )
    for row, pages in enumerate([[2, 0, 1], [1, 2, 0]]):
        key = keys.float()[pages].flatten(0, 1)[:3]
        ref = ((key @ query[row, 0].float().T).relu() * weights[row, 0]).sum(-1)
        torch.testing.assert_close(scores[row, :3], ref, atol=1e-4, rtol=1e-5)
    torch.testing.assert_close(metadata.current, torch.full_like(metadata.current, 2))


def test_prefill_summary_then_decode_continues_partial_region():
    from vllm.models.step5.nvidia.ops.summary_update import (
        build_prefill_groups,
        update_prefill,
    )
    from vllm.models.step5.nvidia.sparse_summary_cache import Step5SparseSummaryCache

    torch.manual_seed(71)
    device = torch.device("cuda")
    cache = Step5SparseSummaryCache(2, 16, device, 2)
    key = torch.randn(2, 10, 1, 256, device=device, dtype=torch.bfloat16)
    z = torch.randn_like(key)
    slots = torch.cat(
        [torch.arange(9, device=device), torch.arange(16, 25, device=device)]
    )
    positions = torch.arange(9, device=device).repeat(2)
    owners = torch.arange(2, device=device).repeat_interleave(9)
    starts = torch.tensor([0, 9, 18], device=device)
    groups = build_prefill_groups(slots, positions, owners, starts, 16)
    update_prefill(
        key[:, :9].reshape(18, 1, 256),
        z[:, :9].reshape(18, 1, 256),
        groups,
        cache.mean,
        cache.numerator,
        cache.denominator,
        cache.maximum,
        cache.last_slot,
        16,
    )
    cache.update_decode(key[:, 9], z[:, 9], torch.tensor([9, 25], device=device))
    for region, start, end in [(0, 0, 8), (1, 8, 10)]:
        ref = (z[:, start:end].float().softmax(1) * key[:, start:end].float()).sum(1)[
            :, 0
        ]
        ref = ref.bfloat16().to(torch.float8_e4m3fn).float()
        actual = cache.mean[:, region].view(torch.float8_e4m3fn).float()
        # Allow adjacent FP8 bins when FP32 reduction orders straddle a midpoint.
        torch.testing.assert_close(actual, ref, atol=0.015625, rtol=0.125)
    torch.testing.assert_close(
        cache.last_slot[:2], torch.tensor([9, 25], device=device)
    )


def test_headwise_gate_preserves_bf16_rounding_and_inactive_rows():
    from vllm.models.step5.nvidia.ops.attention_gate import headwise_sigmoid_gate

    torch.manual_seed(12)
    x = torch.randn(3, 4 * 192, device="cuda", dtype=torch.bfloat16)
    gate = torch.randn(3, 4, device="cuda", dtype=torch.bfloat16)
    active = torch.tensor([True, False, True], device="cuda")
    actual = headwise_sigmoid_gate(x, gate, active)
    expected = (x.view(3, 4, 192) * gate.sigmoid()[..., None]).flatten(1)
    expected[1].zero_()
    torch.testing.assert_close(actual, expected, atol=0.015625, rtol=0.01)
