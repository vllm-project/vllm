# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""SoA verification against an independent decoded-cache FP32 reference."""

import math
from itertools import accumulate

import pytest
import torch

from vllm.platforms import current_platform

if not current_platform.is_rocm():
    pytest.skip("ROCm TurboQuant verification", allow_module_level=True)

from vllm.v1.attention.ops.turboquant_soa.triton_turboquant_unified_attention import (  # noqa: E402
    triton_turboquant_decode_attention_soa,
    triton_turboquant_unified_attention,
)


def _make_case(d, qlens, lengths, gqa=8, hk=1, sink=False, rotate=False):
    torch.manual_seed(61264)
    bs = 16
    batch = len(qlens)
    blocks = math.ceil(max(lengths) / bs)
    L = blocks * bs
    slot = d + 6
    idx = torch.randint(
        0, 16, (batch, blocks, bs, hk, d), device="cuda", dtype=torch.uint8
    )
    vi = torch.randint(0, 16, idx.shape, device="cuda", dtype=torch.uint8)
    norm = torch.ones((batch, blocks, hk, bs), device="cuda", dtype=torch.float16)
    vs = torch.full_like(norm, 0.125)
    vz = torch.full_like(norm, -1.0)
    cent = torch.linspace(-1, 1, 16, device="cuda")
    cache = torch.zeros(
        (batch * blocks, bs, hk, slot), device="cuda", dtype=torch.uint8
    )
    raw = cache.view(batch, blocks, -1)
    payload = torch.cat(
        (idx[..., 0::2] | idx[..., 1::2] << 4, vi[..., 0::2] | vi[..., 1::2] << 4), -1
    )
    raw[:, :, : bs * hk * d] = payload.reshape(batch, blocks, -1)
    meta = torch.stack((norm, vs, vz), -2).contiguous().view(torch.uint8)
    raw[:, :, bs * hk * d :] = meta.reshape(batch, blocks, -1)
    kr = cent[idx.long()].reshape(batch, L, hk, d).float()
    vr = (vi.float() * 0.125 - 1).reshape(batch, L, hk, d)
    bt = torch.arange(batch * blocks, device="cuda", dtype=torch.int32).reshape(
        batch, blocks
    )
    q = (
        torch.randn((sum(qlens), hk * gqa, d), device="cuda", dtype=torch.bfloat16)
        * 0.3
    )
    pi = torch.eye(d, device="cuda")
    if rotate:
        pi = torch.linalg.qr(torch.randn((d, d), device="cuda"))[0].contiguous()
    qsl = torch.tensor([0] + list(accumulate(qlens)), device="cuda", dtype=torch.int32)
    sinks = torch.linspace(-1, 1, hk * gqa, device="cuda") if sink else None
    args = dict(
        query=q,
        kv_cache=cache,
        block_table=bt,
        seq_lens=torch.tensor(lengths, device="cuda", dtype=torch.int32),
        query_start_loc=qsl,
        Pi=pi,
        PiT=pi.T.contiguous(),
        centroids=cent,
        scale=d**-0.5,
        mse_bits=4,
        key_packed_size=d // 2 + 2,
        value_quant_bits=4,
        value_packed_size=d // 2 + 4,
        max_query_len=max(qlens),
        max_seq_len=max(lengths),
        sinks=sinks,
    )
    qr = (q.float() @ pi.T).to(q.dtype).float()
    refs = []
    off = 0
    for i, (r, length) in enumerate(zip(qlens, lengths)):
        k = kr[i, :length].repeat_interleave(gqa, dim=1)
        v = vr[i, :length].repeat_interleave(gqa, dim=1)
        scores = torch.einsum("rhd,lhd->hrl", qr[off : off + r], k) * d**-0.5
        mask = (
            torch.arange(length, device="cuda")[None, :]
            <= (length - r + torch.arange(r, device="cuda"))[:, None]
        )
        scores = scores.masked_fill(~mask[None, :, :], -torch.inf)
        if sinks is not None:
            probs = torch.softmax(
                torch.cat((scores, sinks[:, None, None].expand(-1, r, 1)), dim=-1), -1
            )[..., :length]
        else:
            probs = torch.softmax(scores, -1)
        refs.append(torch.einsum("hrl,lhd->rhd", probs, v))
        off += r
    return args, torch.cat(refs)


@pytest.mark.parametrize("d", [64, 128, 256])
@pytest.mark.parametrize("fused", [False, True])
@pytest.mark.parametrize(
    "rows,lengths,gqa,hk",
    [
        ([1, 1], [1024, 4097], 8, 1),
        ([2, 4, 3], [1024, 4097, 511], 8, 1),
        ([4, 4], [4096, 4097], 8, 2),
        ([8], [8192], 4, 1),
        ([16], [2048], 8, 1),
        ([32], [2048], 8, 1),
        ([33], [2048], 8, 1),
        ([4], [127], 8, 1),
        ([4], [65536], 8, 1),
        ([4], [4096], 6, 1),
    ],
)
def test_verify_reference(d, fused, rows, lengths, gqa, hk):
    args, reference = _make_case(
        d, rows, lengths, gqa, hk, sink=max(rows) == 8, rotate=max(rows) == 4
    )
    output = triton_turboquant_unified_attention(**args, fuse_q_rot=fused)
    assert torch.isfinite(output).all()
    relative_error = (output.float() - reference).norm() / reference.norm()
    assert relative_error < 8e-3


@pytest.mark.parametrize("d", [128, 256])
def test_verify_wrapper_graph_and_poisoned_future(d):
    args, reference = _make_case(d, [4], [4096], rotate=True)
    wrapper_args = dict(args)
    wrapper_args.pop("output", None)
    wrapper_args.pop("force_2d", None)
    output = triton_turboquant_decode_attention_soa(**wrapper_args)
    torch.testing.assert_close(output.float(), reference, atol=5e-3, rtol=3e-2)
    torch.accelerator.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = triton_turboquant_decode_attention_soa(**wrapper_args)
    graph.replay()
    torch.accelerator.synchronize()
    torch.testing.assert_close(captured, output, atol=0, rtol=0)

    # Poison the last token, which the first three queries cannot see.
    # One KV head, 16-token pages, packed 4-bit K/V with three fp16 scales.
    cache = args["kv_cache"].view(args["kv_cache"].shape[0], -1)
    data_offset = 15 * d + d // 2
    cache[-1, data_offset : data_offset + d // 2] = 255
    metadata = cache.view(torch.float16)
    metadata[-1, 16 * d // 2 + 16 + 15] = 100.0
    graph.replay()
    torch.accelerator.synchronize()
    torch.testing.assert_close(captured[:3], output[:3], atol=0, rtol=0)
    assert not torch.equal(captured[3], output[3])


def test_backend_keeps_short_query_blocks(monkeypatch):
    from types import SimpleNamespace

    from vllm.v1.attention.backends.turboquant_attn import (
        TurboQuantAttentionImpl,
        TurboQuantMetadata,
    )

    args, reference = _make_case(128, [1, 4, 33], [4096] * 3)
    impl = object.__new__(TurboQuantAttentionImpl)
    impl._soa_store = True
    impl.sliding_window = None
    impl.sinks = None
    impl.scale = args["scale"]
    impl.tq_config = SimpleNamespace(
        key_mse_bits=4,
        key_packed_size=66,
        effective_value_quant_bits=4,
        value_packed_size=68,
        key_fp8=False,
        norm_correction=False,
    )
    metadata = TurboQuantMetadata(
        seq_lens=args["seq_lens"],
        slot_mapping=torch.empty(38, dtype=torch.int64, device="cuda"),
        block_table=args["block_table"],
        query_start_loc=args["query_start_loc"],
        max_query_len=33,
        max_seq_len=4096,
        query_start_loc_cpu=args["query_start_loc"].cpu(),
        seq_lens_cpu=args["seq_lens"].cpu(),
    )
    dispatches = []
    original = impl._dispatch_decode_soa

    def record(**kwargs):
        dispatches.append((kwargs["max_query_len"], kwargs["block_table"].shape[0]))
        return original(**kwargs)

    monkeypatch.setattr(impl, "_dispatch_decode_soa", record)
    raw_kv = torch.empty((38, 1, 128), dtype=torch.bfloat16, device="cuda")
    output = impl._prefill_attention(
        args["query"],
        raw_kv,
        raw_kv,
        args["kv_cache"],
        metadata,
        args["Pi"],
        args["centroids"],
        args["PiT"],
    )
    assert dispatches == [(1, 1), (4, 1), (1, 33)]
    torch.testing.assert_close(output.float(), reference, atol=5e-3, rtol=3e-2)
