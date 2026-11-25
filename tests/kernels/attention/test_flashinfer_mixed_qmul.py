# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Verify native mixed FMHA against the real vLLM writer, not a simulated pack."""

import pytest
import torch


def test_staged_prefill_budget_falls_back_before_allocation():
    from vllm.v1.attention.ops.flashinfer_mixed_kv_prefill import maybe_staged_prefill

    k = torch.empty(1, 2, 64, 128)
    v, sf = torch.empty(1, 2, 64, 64), torch.empty(1, 2, 64, 8)
    table = torch.empty(1, 1, dtype=torch.int32).expand(32, 160)
    sentinel = torch.tensor(17.0)
    assert not maybe_staged_prefill(
        sentinel,
        k,
        v,
        sf,
        table,
        sentinel,
        sentinel,
        sentinel,
        max_q_len=8,
        num_decodes=0,
        num_decode_tokens=0,
        qk_scale=1.0,
        v_scale=1.0,
        get_workspace=lambda: pytest.fail("must not allocate workspace"),
    )
    assert sentinel.item() == 17.0


@pytest.mark.parametrize("dim,layout", [(128, "HND"), (256, "NHD")])
def test_vllm_packed_writer_to_qmul(dim, layout):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() not in (
        (10, 0),
        (10, 3),
    ):
        pytest.skip("requires Blackwell")
    native = pytest.importorskip("flashinfer.attention.mixed_kv")
    from vllm import _custom_ops  # noqa: F401 (register the compiled cache ops)
    from vllm.utils.torch_utils import fp8_k_nvfp4_v_cache_split_views
    from vllm.v1.attention.ops.flashinfer_mixed_kv_prefill import maybe_staged_prefill

    device = "cuda"
    pages, heads, page, tokens = 3, 2, 64, 69
    width = dim + dim // 2 + dim // 16
    shape = (pages, heads, page, width)
    if layout == "HND":
        writer_cache = torch.zeros(shape, device=device, dtype=torch.uint8).transpose(
            1, 2
        )
    else:
        writer_cache = torch.zeros(
            (pages, page, heads, width), device=device, dtype=torch.uint8
        )
    # Unwritten K and scale bytes may encode NaNs, including within a live page.
    writer_cache.fill_(0x7F)
    # Scale variation across tokens AND channel groups exposes a linear/swizzled
    # mismatch. Six times a power of two is exactly representable by the writer.
    rng = torch.Generator(device=device).manual_seed(23)
    exponents = torch.randint(
        -3, 4, (tokens, heads, dim // 16), device=device, generator=rng
    )
    v = (6 * (2.0**exponents)).repeat_interleave(16, -1).bfloat16()
    k = torch.zeros_like(v)
    slots = torch.arange(tokens, device=device)
    slots = torch.where(slots < page, 2 * page + slots, slots - page)
    scale = torch.tensor(0.5, device=device)
    torch.ops._C_cache_ops.reshape_and_cache_flash(
        k, v, writer_cache, writer_cache, slots, "fp8_k_nvfp4_v", scale, scale
    )
    views = fp8_k_nvfp4_v_cache_split_views(writer_cache, dim)
    k_cache, v_cache, sf = (x.transpose(1, 2) for x in views)
    q = torch.ones((tokens, heads * 4, dim), device=device).to(torch.float8_e4m3fn)
    out = torch.empty_like(q, dtype=torch.bfloat16)
    table = torch.tensor([[2, 0]], device=device, dtype=torch.int32)
    length = torch.tensor([tokens], device=device, dtype=torch.int32)
    for n in (tokens, 1):
        ptr = torch.tensor([0, n], device=device, dtype=torch.int32)
        native.paged_fp8_k_nvfp4_v_fmha(
            q[:n],
            k_cache,
            v_cache,
            sf,
            table,
            length,
            ptr,
            max_q_len=n,
            out=out[:n],
            qk_scale=0.5 / dim**0.5,
            v_scale=0.5,
        )
        # K=0 makes causal attention a uniform average. Compute the answer
        # directly from the original values, without reading any packed bytes.
        denom = torch.arange(1, tokens + 1, device=device)[:, None, None]
        expected = (v.float().cumsum(0) / denom).repeat_interleave(4, dim=1)
        torch.testing.assert_close(
            out[:n].float(), expected[-n:], rtol=0.006, atol=0.01
        )
        if n == tokens:
            out.fill_(float("nan"))
            staged = maybe_staged_prefill(
                q,
                k_cache,
                v_cache,
                sf,
                table,
                length,
                ptr,
                out,
                max_q_len=n,
                num_decodes=0,
                num_decode_tokens=0,
                qk_scale=0.5 / dim**0.5,
                v_scale=0.5,
                get_workspace=lambda: torch.empty(
                    256 << 20, device=device, dtype=torch.uint8
                ),
            )
            assert staged == (layout == "HND")
            if staged:
                torch.testing.assert_close(out.float(), expected, rtol=0.006, atol=0.01)


def test_staged_mixed_batch_graph():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 3):
        pytest.skip("requires SM103a")
    pytest.importorskip("flashinfer.attention.mixed_kv")
    from vllm.v1.attention.ops.flashinfer_mixed_kv_prefill import maybe_staged_prefill

    dev, fp8 = "cuda", torch.float8_e4m3fn
    q = torch.ones(10, 8, 128, device=dev, dtype=fp8)
    k = torch.zeros(3, 2, 64, 128, device=dev, dtype=fp8)
    v = torch.empty(3, 2, 64, 64, device=dev, dtype=torch.uint8)
    for page, code in enumerate((0x11, 0x22, 0x44)):
        v[page].fill_(code)
    sf = torch.ones(3, 2, 64, 8, device=dev, dtype=fp8)
    tab = (
        (torch.arange(10, device=dev, dtype=torch.int32) % 3)[:, None]
        .expand(-1, 40)
        .contiguous()
    )
    lens = torch.full((10,), 65, device=dev, dtype=torch.int32)
    ptr = torch.tensor(
        [0, 1, 1, 1, 1, 2, 2, 2, 2, 5, 10], device=dev, dtype=torch.int32
    )
    ws = torch.empty(256 << 20, device=dev, dtype=torch.uint8)
    out = torch.empty_like(q, dtype=torch.bfloat16)

    def run():
        assert maybe_staged_prefill(
            q,
            k,
            v,
            sf,
            tab,
            lens,
            ptr,
            out,
            max_q_len=5,
            num_decodes=8,
            num_decode_tokens=2,
            qk_scale=0.125,
            v_scale=1.0,
            get_workspace=lambda: ws,
        )

    run()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    for counts in ([1, 0, 0, 0, 1, 0, 0, 0, 3, 5], [0, 1, 1, 0, 0, 0, 0, 0, 5, 3]):
        c = torch.tensor(counts, device=dev, dtype=torch.int32)
        torch.cumsum(c, 0, out=ptr[1:])
        lens.copy_(torch.where(c > 0, 65, 0))
        out.fill_(float("nan"))
        graph.replay()
        vals = torch.tensor([0.5, 1.0, 2.0], device=dev)[
            torch.arange(10, device=dev) % 3
        ]
        want = vals.repeat_interleave(c.long())[:, None, None].expand_as(out)
        torch.testing.assert_close(out.float(), want, rtol=0.01, atol=0.003)
