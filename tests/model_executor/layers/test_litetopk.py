# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

from vllm.model_executor.layers import litetopk_indexer, sparse_attn_indexer
from vllm.utils import deep_gemm as deep_gemm_utils
from vllm.v1.attention.backends.mla import indexer as indexer_metadata


@pytest.fixture(autouse=True)
def clear_litetopk_caches():
    caches = (
        litetopk_indexer.production_extension_available,
        deep_gemm_utils._import_deep_gemm,
        deep_gemm_utils._get_fp8_fp4_mqa_logits_out_impl,
    )
    for cache in caches:
        cache.cache_clear()
    yield
    for cache in caches:
        cache.cache_clear()


@pytest.mark.parametrize(
    "length,common_end", ((196608, 188417), (1048576, 1040385), (20003, 11812))
)
def test_fixed_random_pages_preserve_causal_tail(monkeypatch, length, common_end):
    lt = litetopk_indexer
    monkeypatch.setattr(lt, "_RANDOM_PAGE_ORDER", {})
    rng = torch.get_rng_state().clone()
    order, sample_size = lt._random_page_order(length, common_end, torch.device("cpu"))
    again, again_size = lt._random_page_order(length, common_end, torch.device("cpu"))
    assert order.data_ptr() == again.data_ptr()
    assert again_size == sample_size
    assert torch.equal(torch.get_rng_state(), rng)
    pages = (length + 63) // 64
    historical_pages = common_end // 64
    assert torch.equal(order.sort().values, torch.arange(pages, dtype=torch.int32))
    assert (order[: sample_size // 64] < historical_pages).all()
    assert sample_size % 256 == 0
    assert torch.equal(
        order[historical_pages:],
        torch.arange(historical_pages, pages, dtype=torch.int32),
    )
    # The mask is deterministic after rebuilding, independently of global RNG.
    lt._RANDOM_PAGE_ORDER.clear()
    rebuilt, _ = lt._random_page_order(length, common_end, torch.device("cpu"))
    assert torch.equal(rebuilt, order)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("rows", (1024, 1016))
@pytest.mark.parametrize("rank", (0, 7))
def test_paged_random_runtime_skips_full_gather(monkeypatch, rows, rank):
    if torch.cuda.get_device_capability() != (10, 0):
        pytest.skip("requires SM100")
    if not deep_gemm_utils.is_fp8_fp4_mqa_logits_out_supported():
        pytest.skip("requires patched DeepGEMM out= API")

    lt = litetopk_indexer
    ext = lt._ext()
    assert ext is not None
    monkeypatch.setattr(lt, "ENABLED", True)
    monkeypatch.setattr(lt, "_RANDOM_PAGE_ORDER", {})
    monkeypatch.setattr(lt, "_HINTS_VALIDATED", False)
    monkeypatch.setattr(lt, "_PROBE_RES", None)
    monkeypatch.setattr(lt, "OVF_LOG", True)
    monkeypatch.setattr(lt, "_TELEMETRY", {"calls": 0, "candidate_max": 0})

    length = 196613
    pages = (length + 63) // 64
    torch.manual_seed(239)
    keys = torch.randn(pages * 64, 128, device="cuda").to(torch.float8_e4m3fn)
    scales = torch.rand(pages * 64, device="cuda") * 0.1 + 0.01
    physical = torch.randperm(pages, device="cuda")
    cache = torch.empty(pages, 64, 132, device="cuda", dtype=torch.uint8)
    flat = cache.view(pages, -1)
    flat[physical, :8192] = keys.view(torch.uint8).view(pages, 8192)
    flat[physical, 8192:] = scales.view(torch.uint8).view(pages, 256)
    table = physical.to(torch.int32)[None].contiguous()
    q = torch.randn(rows, 32, 128, device="cuda").to(torch.float8_e4m3fn)
    weights = torch.rand(rows, 32, device="cuda") * 0.05
    starts = torch.zeros(rows, device="cuda", dtype=torch.int32)
    common_end = length - rows * 8 + rows * rank + 1
    ends = torch.arange(common_end, common_end + rows, device="cuda", dtype=torch.int32)
    dst = torch.empty(length, 128, device="cuda", dtype=torch.float8_e4m3fn)
    dst.view(torch.uint8).fill_(37)
    dst_scales = torch.full((length, 4), 99, device="cuda", dtype=torch.uint8)
    out = torch.empty(rows, 2048, device="cuda", dtype=torch.int32)
    plan = lt.prepare_paged_sample(
        cache,
        dst,
        dst_scales,
        table,
        sequence_length=length,
        query_length=rows,
        num_reqs=1,
        common_end=common_end,
    )
    assert plan is not None
    assert plan["sample_size"] == 32768
    assert plan["page_order"].numel() == pages
    assert (dst.view(torch.uint8)[32768:] == 37).all()
    assert (dst_scales[32768:] == 99).all()
    assert lt.try_large_exact_once_chunk(
        q,
        dst,
        dst_scales.view(torch.float32).view(-1),
        weights,
        starts,
        ends,
        out,
        2048,
        sample_plan=plan,
        num_reqs=1,
        ke_min_hint=common_end,
    )
    torch.cuda.synchronize()
    lt._probe_candidate_telemetry(*lt._CAND_ACC)
    assert lt._TELEMETRY["candidate_max"] >= 2048
    assert ((out >= 0) & (out < ends[:, None])).all()
    sorted_ids = out.sort(1).values
    assert (sorted_ids[:, 1:] != sorted_ids[:, :-1]).all()
    check = torch.cat(
        (torch.arange(16, device="cuda"), torch.arange(rows - 16, rows, device="cuda"))
    )
    reference = deep_gemm_utils.fp8_fp4_mqa_logits(
        (q[check], None),
        (keys[:length], scales[:length]),
        weights[check],
        starts[check],
        ends[check],
        clean_logits=True,
    )
    cutoff = reference.topk(2048, dim=1).values[:, -1]
    selected = reference.gather(1, out[check].long())
    assert (selected.min(1).values >= cutoff - 1e-3 * cutoff.abs().clamp(min=1)).all()


def test_deepgemm_mqa_out_wrapper_requires_alias(monkeypatch):
    placeholder = torch.empty(1)
    out = torch.empty(16)

    def output_backend(*args, clean_logits, out=None):
        return out[:4].view(2, 2)

    monkeypatch.setattr(deep_gemm_utils, "_lazy_init", lambda: None)
    monkeypatch.setattr(
        deep_gemm_utils,
        "_get_fp8_fp4_mqa_logits_out_impl",
        lambda: output_backend,
    )
    result = deep_gemm_utils.fp8_fp4_mqa_logits(
        (placeholder, None),
        (placeholder, placeholder),
        placeholder,
        placeholder,
        placeholder,
        clean_logits=False,
        out=out,
    )
    assert result.data_ptr() == out.data_ptr()

    def non_aliasing_backend(*args, clean_logits, out=None):
        return out.clone()

    monkeypatch.setattr(
        deep_gemm_utils,
        "_get_fp8_fp4_mqa_logits_out_impl",
        lambda: non_aliasing_backend,
    )
    with pytest.raises(RuntimeError, match="did not return an alias"):
        deep_gemm_utils.fp8_fp4_mqa_logits(
            (placeholder, None),
            (placeholder, placeholder),
            placeholder,
            placeholder,
            placeholder,
            clean_logits=False,
            out=out,
        )


def _check_h2048_selection(scores, cap=49152):
    """Compare exact score/slot membership, using non-identity physical IDs."""
    ext = litetopk_indexer._ext()
    assert ext is not None
    rows, count = scores.shape
    topk = 2048
    bits = scores.contiguous().view(torch.int32).to(torch.int64)
    codes = torch.where(bits >= 0, bits ^ 0x80000000, ~bits) >> 8
    slots = torch.arange(count, device="cuda", dtype=torch.int64)[None]
    physical = (count - 1 - slots).expand(rows, -1)
    values = torch.empty(rows, cap, device="cuda", dtype=torch.float16)
    indices = torch.empty(rows, cap, device="cuda", dtype=torch.int32)
    values[:, :count] = (codes & 0xFFFF).to(torch.int16).view(torch.float16)
    indices[:, :count] = (physical | ((codes >> 16) << 20)).to(torch.int32)
    counts = torch.full((rows,), count, device="cuda", dtype=torch.int32)
    out = torch.empty(rows, topk, device="cuda", dtype=torch.int32)
    status = torch.empty(rows, device="cuda", dtype=torch.int32)
    diagnostics = torch.empty(rows, 5, device="cuda", dtype=torch.int32)
    ext.h2048_safe_topk_out_litetopk_(
        values, indices, counts, out, status, diagnostics, count
    )
    assert (status == 0).all().item()
    assert ((out >= 0) & (out < count)).all().item()
    ordered = out.sort(1).values
    assert (ordered[:, 1:] != ordered[:, :-1]).all().item()
    selected_slots = ((codes << 20) | slots).topk(topk, dim=1, largest=False).indices
    torch.testing.assert_close(
        ordered.long(),
        physical.gather(1, selected_slots).sort(1).values,
        atol=0,
        rtol=0,
    )
    return diagnostics


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 0),
    reason="LiteTopK requires SM100",
)
@pytest.mark.parametrize(
    "boundary", (31, 512, 2048, 2049, 4095, 4096, 4097, 8192, 8193, 16384)
)
@pytest.mark.parametrize("tied", (False, True))
def test_h2048_cached_and_streamed_boundaries(boundary, tied):
    """Cached and oversized boundaries must retain the same stable tie rule."""
    count, topk = max(16384, boundary + 2048), 2048
    strict = topk - min(256, max(1, boundary // 2))
    edge = (
        torch.full((boundary,), 8.03125, device="cuda")
        if tied
        else torch.linspace(8.0, 8.12, boundary, device="cuda")
    )
    row = torch.cat(
        (
            torch.linspace(1.0, 7.9, strict, device="cuda"),
            edge,
            torch.linspace(9.0, 200.0, count - strict - boundary, device="cuda"),
        )
    )
    generator = torch.Generator(device="cuda").manual_seed(17)
    row = row[torch.randperm(count, device="cuda", generator=generator)]
    diagnostics = _check_h2048_selection(torch.stack((row, row.flip(0))))
    assert (diagnostics[:, 3] == boundary).all().item()
    if tied:
        assert (diagnostics[:, 4] == boundary).all().item()


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 0),
    reason="LiteTopK requires SM100",
)
@pytest.mark.parametrize("lo", (-1.0, 0.125, 7.875, 8.0, 127.875, 255.75, 255.875))
@pytest.mark.parametrize("boundary", (2048, 8193))
def test_h2048_radix_prefix_and_clamped_buckets(lo, boundary):
    if lo < 0:
        # The first bucket includes all negative keys and must search 24 bits.
        row = torch.cat(
            (
                torch.linspace(-100.0, 0.12, boundary, device="cuda"),
                torch.linspace(1.0, 2.0, 1024, device="cuda"),
            )
        )
    else:
        # The last bucket also includes values beyond the calibrated range.
        hi = 512.0 if lo == 255.875 else lo + 0.12
        row = torch.cat(
            (
                torch.linspace(-1.0, -0.125, 1024, device="cuda"),
                torch.linspace(lo, hi, boundary, device="cuda"),
            )
        )
    _check_h2048_selection(torch.stack((row, row.flip(0))))


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 0),
    reason="LiteTopK requires SM100",
)
@pytest.mark.parametrize("boundary", (2048, 8193))
def test_h2048_tie_slots_above_uint16(boundary):
    row = torch.linspace(9.0, 200.0, 131072, device="cuda")
    row[:1792] = 1.0
    row[65536 : 65536 + boundary] = 8.03125
    _check_h2048_selection(torch.stack((row, row.flip(0))), cap=row.numel())


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 0),
    reason="LiteTopK requires SM100",
)
@pytest.mark.parametrize("count", (49152, 131072, 1 << 20))
@pytest.mark.parametrize("tied", (False, True))
def test_h2048_entire_candidate_row_in_one_bucket(count, tied):
    row = (
        torch.full((count,), 8.03125, device="cuda")
        if tied
        else torch.linspace(8.0, 8.12, count, device="cuda")
    )
    diagnostics = _check_h2048_selection(torch.stack((row, row.flip(0))), cap=count)
    assert (diagnostics[:, 3] == count).all().item()


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 0),
    reason="LiteTopK requires SM100",
)
@pytest.mark.parametrize("bad", ("count_short", "count_overflow", "index", "nonfinite"))
def test_h2048_rejects_invalid_candidates(bad):
    ext = litetopk_indexer._ext()
    assert ext is not None
    count, cap = 2048, 49152
    code = (torch.tensor(8.0).view(torch.int32).item() ^ 0x80000000) >> 8
    values = torch.zeros(1, cap, device="cuda", dtype=torch.float16)
    indices = torch.arange(cap, device="cuda", dtype=torch.int32)[None]
    indices |= (code >> 16) << 20
    counts = torch.full((1,), count, device="cuda", dtype=torch.int32)
    expected_status = 1
    if bad == "count_short":
        counts.fill_(count - 1)
    elif bad == "count_overflow":
        counts.fill_(cap + 1)
    elif bad == "index":
        indices[0, 0] = ((code >> 16) << 20) | count
        expected_status = 4
    else:
        values.view(torch.int16)[0, 0] = -32768
        indices[0, 0] = 0xFF << 20
        expected_status = 2
    out = torch.empty(1, count, device="cuda", dtype=torch.int32)
    status = torch.empty(1, device="cuda", dtype=torch.int32)
    diag = torch.empty(1, 5, device="cuda", dtype=torch.int32)
    ext.h2048_safe_topk_out_litetopk_(values, indices, counts, out, status, diag, count)
    assert status.item() & expected_status
    assert (out == -1).all().item()


@pytest.mark.parametrize("query_len", (32768, 32704))
def test_32k_queries_keep_dense_logits_budget(query_len):
    builder = indexer_metadata.DeepseekV32IndexerMetadataBuilder
    chunks = builder._split_indexer_prefill_chunks(
        torch.tensor([262144]),
        torch.tensor([query_len]),
        workspace_size=1 << 22,
        max_logits_bytes=2 * 1024**3,
        fused_min_seq_len=196608,
    )
    assert len(chunks) > 1
    assert all(
        (query.stop - query.start) * 262144 * 4 <= 2 * 1024**3 for _, query in chunks
    )


@pytest.mark.parametrize(
    "model_type,tp,heads,expected",
    (
        ("glm_moe_dsa", 8, 32, True),
        ("glm_moe_dsa", 4, 32, False),
        ("glm_moe_dsa", 1, 32, False),
        ("glm_moe_dsa", 8, 64, False),
        ("deepseek_v32", 8, 32, False),
        ("deepseek_v4", 8, 64, False),
    ),
)
def test_model_scope(model_type, tp, heads, expected):
    config = SimpleNamespace(
        model_type=model_type, index_n_heads=heads, index_head_dim=128, index_topk=2048
    )
    assert litetopk_indexer.supports_model(config, tp) is expected


@pytest.mark.parametrize("tp,pcp,dcp", ((8, 1, 1), (4, 1, 1), (8, 2, 1), (8, 1, 2)))
def test_tp8_shards_reject_other_parallel_layouts(monkeypatch, tp, pcp, dcp):
    monkeypatch.setattr(sparse_attn_indexer.envs, "VLLM_LITETOPK", True)
    monkeypatch.setattr(
        sparse_attn_indexer,
        "get_tp_group",
        lambda: SimpleNamespace(world_size=tp, rank_in_group=tp - 1),
    )
    shard = sparse_attn_indexer._litetopk_tp_query_shard(
        8128,
        2048,
        torch.device("cpu"),
        use_fp4_cache=False,
        use_pcp=pcp > 1,
        pcp_world_size=pcp,
        compress_ratio=1,
        num_heads=32,
        num_reqs=1,
        dcp_world_size=dcp,
    )
    if tp == 8 and pcp == dcp == 1:
        assert shard[:2] == (7112, 8128)
        assert shard[2].shape == (1016, 2048)
    else:
        assert shard is None
