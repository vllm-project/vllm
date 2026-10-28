# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.model_executor.layers.unlimited_ocr2 import (
    dash_attention,
    page_coordinates,
    route_blocks,
)


def test_page_coordinates_continue_across_prompt_and_response():
    roles = torch.tensor([0, 2, 1, 1, 2, 1, 1, 1, 0, 2, 0, 0, 2, 0])
    tok, page, image_page, query_page = page_coordinates(roles)
    assert page.tolist() == [0, 1, 1, 1, 2, 2, 2, 2, 0, 1, 1, 1, 2, 2]
    assert tok.tolist() == [0, 0, 1, 2, 0, 1, 2, 3, 1, 3, 4, 5, 4, 5]
    assert image_page.tolist() == [-1, -1, 0, 0, -1, 1, 1, 1, -1, -1, -1, -1, -1, -1]
    assert query_page.tolist() == [-1, 0, 0, 0, 1, 1, 1, 1, -1, 0, 0, 0, 1, 1]


def test_zero_prior_blocks_still_consume_the_hard_read_budget():
    scores = torch.arange(20).float()[None]
    sp = {
        "sparse_attention_freq": [1],
        "sparse_index_dim": 128,
        "dash_topk_blocks": 12,
    }
    indices, bias, valid = route_blocks(scores, torch.ones_like(scores).bool(), sp, 0)
    assert indices.tolist() == [list(range(19, 7, -1))]
    assert valid.all()
    assert torch.isfinite(bias).all()
    torch.testing.assert_close(bias.sum(-1), torch.zeros(1), atol=1e-5, rtol=0)
    assert bias[0, 0] > bias[0, 1]


def test_extra_response_marker_restarts_final_page_offset():
    roles = torch.tensor([0, 2, 1, 1, 0, 2, 0, 2, 0])
    tok, page, _, query_page = page_coordinates(roles)
    assert page.tolist() == [0, 1, 1, 1, 0, 1, 1, 1, 1]
    assert tok.tolist() == [0, 0, 1, 2, 1, 3, 4, 3, 4]
    # Routing follows response markers even when RoPE has reached its last bucket.
    assert query_page.tolist() == [-1, 0, 0, 0, -1, 0, 0, 1, 1]


@pytest.mark.parametrize("nq", [1, 17, 200])
def test_selected_remote_and_local_overlap_is_counted_once(nq):
    """Zero Q/K and all remote blocks selected give causal prefix means."""
    length, dim = 200, 128
    roles = torch.tensor([2] + [1] * (length - 1))
    q = torch.zeros(nq, 1, dim)
    k = torch.zeros(length, 1, dim)
    values = (torch.arange(length).float() / length)[:, None, None].expand_as(k)
    sp = {
        "sparse_block_size": 32,
        "sparse_local_block": 4,
        "sparse_index_dim": dim,
        "sparse_attention_freq": [1],
        "dash_topk_blocks": 12,
    }
    result = dash_attention(
        q, k, values, q[:, 0], k[:, 0], roles, torch.zeros(dim), sp, 0
    )
    expected = torch.arange(length - nq, length).float() / (2 * length)
    torch.testing.assert_close(result[:, 0, 0], expected, atol=4e-7, rtol=0)


def test_page_filter_excludes_other_page_remote_values():
    """Page-two queries read page-two image blocks plus their local window."""
    length, dim = 576, 128
    roles = torch.tensor([2] + [1] * 287 + [2] + [1] * 286 + [2])
    q = torch.zeros(1, 1, dim)
    k = torch.zeros(length, 1, dim)
    v = torch.zeros_like(k)
    v[:256] = 1000
    sp = {
        "sparse_block_size": 32,
        "sparse_local_block": 4,
        "sparse_index_dim": dim,
        "sparse_attention_freq": [1],
        "dash_topk_blocks": 12,
    }
    result = dash_attention(
        q, k[:-1], v[:-1], q[:, 0], k[:-1, 0], roles[:-1], torch.zeros(dim), sp, 0
    )
    assert torch.count_nonzero(result) == 0


@pytest.mark.parametrize("rswa", [False, True])
@pytest.mark.parametrize(
    "dtype,backend",
    [
        (torch.bfloat16, "FLASH_ATTN"),
        (torch.bfloat16, "FLASHINFER"),
    ],
)
@pytest.mark.parametrize("block_size", [16, 32])
def test_paged_decode_graph_replay_tracks_requests_and_page_markers(
    dtype, backend, block_size, rswa
):
    """Replay changes lengths, page markers and KV while keeping captured addresses."""
    _check_paged_decode_graph(dtype, backend, block_size, rswa, high_cache=False)


@pytest.mark.parametrize("backend", ["FLASH_ATTN", "FLASHINFER"])
def test_paged_decode_large_physical_cache_offsets(backend):
    """Read/write offsets beyond int32 while replaying the real paged kernels."""
    _check_paged_decode_graph(torch.bfloat16, backend, 16, True, high_cache=True)


def _check_paged_decode_graph(dtype, backend, block_size, rswa, high_cache):
    from vllm.v1.attention.ops.unlimited_ocr2 import (
        cache_dash_prefill,
        paged_dash_decode,
    )

    device = "cuda"
    torch.manual_seed(42)
    n, dim, heads, max_len = 3, 128, 2, 2048
    blocks_per_request = max_len // block_size
    table = (
        torch.randperm(n * blocks_per_request, device=device, dtype=torch.int32).view(
            n, -1
        )
        + 1
    )
    # Match the backend's logical shape and non-contiguous physical layout.
    leading_blocks = (
        (2**31 // (block_size * (heads + 1) * 2 * dim) + 1) if high_cache else 0
    )
    table += leading_blocks
    allocate = torch.empty if high_cache else torch.zeros
    cache = allocate(
        leading_blocks + n * blocks_per_request + 1,
        block_size,
        heads + 1,
        2 * dim,
        dtype=dtype,
        device=device,
    ).permute(0, 2, 1, 3)
    sp = {
        "sparse_block_size": 32,
        "sparse_local_block": 4,
        "sparse_index_dim": dim,
        "sparse_attention_freq": [1],
        "dash_topk_blocks": 12,
        "sparse_rope_page_mod": 2,
    }
    history_roles = [
        torch.tensor(
            [0, 2]
            + [1] * 273
            + [2]
            + [1] * 273
            + [0] * 3
            + [2]
            + [0] * 500
            + [2]
            + [0] * 499,
            device=device,
        ),
        torch.tensor([0, 2] + [1] * 273 + [0] * 3 + [2] + [0] * 762, device=device),
    ]
    # Null / evicted storage is poisoned: masked gaps must never be read.
    cache[0] = float("nan")
    prefix_lens = torch.tensor([552, 278, 0], device=device, dtype=torch.int32)
    cls = torch.randn(dim, device=device, dtype=dtype)
    histories = []
    for row, roles in enumerate(history_roles):
        length = len(roles)
        k = torch.randn(length, heads, dim, device=device, dtype=dtype)
        v = torch.randn_like(k)
        ik = torch.randn(length, dim, device=device, dtype=dtype)
        ar = torch.arange(length, device=device)
        slots = table[row, ar // block_size] * block_size + ar % block_size
        cache_dash_prefill(k, v, ik, roles, slots, cache, sp, cls)
        histories.append([k, v, ik, roles])

    q = torch.randn(n, heads, dim, device=device, dtype=dtype)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    iq = torch.randn(n, dim, device=device, dtype=dtype)
    ik = torch.randn_like(iq)
    roles = torch.tensor([0, 2, 0], device=device)
    lengths = torch.tensor(
        [len(r) + 1 for r in history_roles] + [0], device=device, dtype=torch.int32
    )
    pos = (lengths - 1).long()
    slots = table[torch.arange(n, device=device), pos.clamp(min=0) // block_size]
    slots = slots * block_size + pos % block_size
    slots[-1] = -1

    def decode(cached_summaries=True):
        return paged_dash_decode(
            q,
            k,
            v,
            iq,
            ik,
            roles,
            cache,
            table,
            lengths,
            slots,
            cls,
            sp,
            0,
            max_len,
            prefix_lens if rswa else None,
            backend=backend,
            cached_summaries=cached_summaries,
        )

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        decode()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = decode()

    for step in range(3):
        if rswa:
            for row in range(2):
                first = (int(prefix_lens[row]) + block_size - 1) // block_size
                last = (int(lengths[row]) - 1 - 128) // block_size
                freed = table[row, first:last].clone()
                table[row, first:last] = 0
                cache[freed] = float("nan")
        graph.replay()
        uncached = decode(cached_summaries=False)
        torch.testing.assert_close(actual, uncached, atol=0, rtol=0)
        for row, history in enumerate(histories):
            ck, cv, ci, cr = history
            ck = torch.cat((ck, k[row : row + 1]))
            cv = torch.cat((cv, v[row : row + 1]))
            ci = torch.cat((ci, ik[row : row + 1]))
            cr = torch.cat((cr, roles[row : row + 1]))
            expected = dash_attention(
                q[row : row + 1],
                ck,
                cv,
                iq[row : row + 1],
                ci,
                cr,
                cls,
                sp,
                0,
                prompt_length=int(prefix_lens[row]) if rswa else None,
            )
            torch.testing.assert_close(
                actual[row : row + 1],
                expected,
                atol=2e-6 if dtype == torch.float32 else 0.00390625,
                rtol=1e-5 if dtype == torch.float32 else 0.01,
            )
            histories[row] = [ck, cv, ci, cr]
        assert torch.count_nonzero(actual[-1]) == 0
        if step < 2:
            lengths[:2] += 1
            pos = (lengths[:2] - 1).long()
            slots[:2] = table[torch.arange(2, device=device), pos // block_size]
            slots[:2] = slots[:2] * block_size + pos % block_size
            roles[:2] = torch.tensor([2, 0], device=device)
            for tensor in (q, k, v, iq, ik):
                tensor.normal_()
            if step == 1:
                # Scheduler row order may change while graph addresses stay fixed.
                histories.reverse()
                prefix_lens[:2].copy_(prefix_lens[:2].flip(0))
                table[:2].copy_(table[:2].flip(0))
                lengths[:2].copy_(lengths[:2].flip(0))
                slots[:2].copy_(slots[:2].flip(0))
                roles[:2].copy_(roles[:2].flip(0))


@pytest.mark.parametrize("block_size", [16, 32])
def test_rswa_keeps_input_and_bounds_1000_output_tokens(block_size):
    """The existing R-SWA manager recycles output blocks in a bounded pool."""
    from vllm.v1.core.block_pool import BlockPool
    from vllm.v1.core.single_type_kv_cache_manager import RSWAManager
    from vllm.v1.kv_cache_interface import RSWASpec

    prefix, window = 1000, 128
    prefix_blocks = (prefix + block_size - 1) // block_size
    bound = prefix_blocks + window // block_size + 1
    pool = BlockPool(
        num_gpu_blocks=bound + 2, enable_caching=False, hash_block_size=block_size
    )
    manager = RSWAManager(
        RSWASpec(
            block_size=block_size,
            num_kv_heads=1,
            head_size=128,
            dtype=torch.bfloat16,
            rswa_window=window,
        ),
        block_pool=pool,
        enable_caching=False,
        kv_cache_group_id=0,
        scheduler_block_size=block_size,
    )
    manager.allocate_new_blocks("doc", prefix, prefix)
    original = [b.block_id for b in manager.req_to_blocks["doc"][:prefix_blocks]]
    peak = 0
    for processed in range(prefix, prefix + 1000):
        manager.remove_skipped_blocks("doc", processed, num_prompt_tokens=prefix)
        manager.allocate_new_blocks("doc", processed + 1, processed + 1)
        blocks = manager.req_to_blocks["doc"]
        assert [b.block_id for b in blocks[:prefix_blocks]] == original
        live = sum(not b.is_null for b in blocks)
        peak = max(peak, live)
        assert live <= bound
    assert peak == bound
    assert any(b.is_null for b in manager.req_to_blocks["doc"][prefix_blocks:])


@pytest.mark.parametrize("backend", ["FLASH_ATTN", "FLASHINFER"])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_flash_prefill_preserves_route_priors_and_gqa(backend, dtype):
    """Compare real kernels with DASH math across page and partial-block edges."""
    torch.manual_seed(23)
    length, dim = 577, 128
    q = torch.randn(length, 4, dim, device="cuda", dtype=dtype) * 0.3
    k = torch.randn(length, 2, dim, device="cuda", dtype=dtype) * 0.3
    v = torch.randn_like(k)
    iq = torch.randn(length, dim, device="cuda", dtype=dtype)
    ik = torch.randn_like(iq)
    roles = torch.ones(length, device="cuda", dtype=torch.long)
    roles[0] = 2
    roles[288] = 2
    roles[-17:] = 0
    roles[-16] = 2
    sp = {
        "sparse_block_size": 32,
        "sparse_local_block": 4,
        "sparse_index_dim": dim,
        "sparse_attention_freq": [1],
        "dash_topk_blocks": 12,
    }
    cls = torch.randn(dim, device="cuda")
    expected = dash_attention(q, k, v, iq, ik, roles, cls, sp, 0)
    result = dash_attention(q, k, v, iq, ik, roles, cls, sp, 0, backend=backend)
    atol = 0.00390625 if dtype == torch.bfloat16 else 0.00048828125
    torch.testing.assert_close(result, expected, atol=atol, rtol=0.01)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("main", [False, True])
def test_decode_rope_preserves_long_position_rounding(dtype, main):
    from vllm.model_executor.layers.unlimited_ocr2 import page_rope
    from vllm.v1.attention.ops.unlimited_ocr2 import decode_page_rope

    torch.manual_seed(81)
    coords = torch.tensor(
        [[0, 0], [1001, 17], [235929, 299], [786431, 300]], device="cuda"
    )
    x = torch.randn(4, 2, 128, device="cuda", dtype=dtype)
    sp = {
        "sparse_rope_page_pairs": 2,
        "sparse_rope_page_mod": 256,
        "sparse_main_rope_tok_freq_mode": "truncate",
    }
    expected = page_rope(x, coords[:, 0], coords[:, 1], sp, main=main)
    actual = decode_page_rope(x, coords[:, 0], coords[:, 1], sp, main=main)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


@pytest.mark.parametrize("backend", ["FLASH_ATTN", "FLASHINFER"])
def test_chunked_prefill_matches_full_prompt_across_page_and_block_boundaries(backend):
    """Chunks can end before an image, inside it, or at the instruction tail."""
    from vllm.model_executor.layers.unlimited_ocr2 import prompt_page_positions
    from vllm.v1.attention.ops.unlimited_ocr2 import (
        paged_dash_decode,
        paged_dash_prefill,
    )

    torch.manual_seed(42)
    device, dtype = "cuda", torch.bfloat16
    roles = torch.tensor(
        [0, 2] + [1] * 1043 + [0, 2] + [1] * 1061 + [0] * 5, device=device
    )
    length, h, d, block_size = len(roles), 2, 128, 16
    sp = {
        "sparse_block_size": 32,
        "sparse_local_block": 1,
        "sparse_index_dim": d,
        "sparse_attention_freq": [1],
        "dash_topk_blocks": 12,
        "image_token_id": 1,
        "page_token_id": 2,
    }
    positions = prompt_page_positions(roles.tolist(), sp).to(device)
    q = torch.randn(length + 2, 4, d, device=device, dtype=dtype)
    k = torch.randn(length + 2, h, d, device=device, dtype=dtype)
    v = torch.randn_like(k)
    iq = torch.randn(length + 2, d, device=device, dtype=dtype)
    ik = torch.randn_like(iq)
    cls = torch.randn(d, device=device, dtype=dtype)
    table = torch.randperm(256, device=device, dtype=torch.int32)[None]
    cache = torch.full(
        (256, block_size, h + 1, 2 * d), float("nan"), device=device, dtype=dtype
    ).permute(0, 2, 1, 3)
    ar = torch.arange(length + 2, device=device)
    slots = table[0, ar // block_size] * block_size + ar % block_size
    expected = dash_attention(
        q[:length],
        k[:length],
        v[:length],
        iq[:length],
        ik[:length],
        roles,
        cls,
        sp,
        0,
        backend=backend,
    )
    chunks = [0, 1, 2, 17, 32, 1024, 1045, 1047, 1500, 2048, 2109, length]
    for start, end in zip(chunks, chunks[1:]):
        args = (
            q[start:end],
            k[start:end],
            v[start:end],
            iq[start:end],
            ik[start:end],
            roles[start:end],
        )
        if end - start == 1:
            actual = paged_dash_decode(
                *args,
                cache,
                table,
                torch.tensor([end], device=device, dtype=torch.int32),
                slots[start:end],
                cls,
                sp,
                0,
                4096,
                backend=backend,
                cached_summaries=True,
                positions=positions[:, start:end],
            )
        else:
            actual = paged_dash_prefill(
                *args,
                positions[:, start:end],
                cache,
                table,
                end,
                slots[start:end],
                cls,
                sp,
                0,
                backend,
            )
        torch.testing.assert_close(
            actual, expected[start:end], atol=0.00390625, rtol=0.01
        )
    # Both pages' decode offsets must survive the chunk boundaries.
    for step in range(2):
        pos = length + step
        roles = torch.cat((roles, torch.tensor([2], device=device)))
        actual = paged_dash_decode(
            q[pos : pos + 1],
            k[pos : pos + 1],
            v[pos : pos + 1],
            iq[pos : pos + 1],
            ik[pos : pos + 1],
            roles[-1:],
            cache,
            table,
            torch.tensor([pos + 1], device=device, dtype=torch.int32),
            slots[pos : pos + 1],
            cls,
            sp,
            0,
            4096,
            torch.tensor([length], device=device, dtype=torch.int32),
            backend=backend,
            cached_summaries=True,
        )
        expected = dash_attention(
            q[pos : pos + 1],
            k[: pos + 1],
            v[: pos + 1],
            iq[pos : pos + 1],
            ik[: pos + 1],
            roles,
            cls,
            sp,
            0,
            prompt_length=length,
            backend=backend,
        )
        torch.testing.assert_close(actual, expected, atol=0.00390625, rtol=0.01)
