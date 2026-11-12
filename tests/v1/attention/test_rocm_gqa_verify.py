# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Decode and speculative-decode verify attention kernels (GQA 8, head size 64, FP8
KV) for the AITER FA backend (gfx950).

The layout, gate and fallback tests run anywhere. The kernel tests need a gfx950
GPU with FlyDSL and compare `fast_verify` against a float32 reference on
bf16 queries and an FP8 e4m3 KV cache, for 1 to 8 query tokens per request.
"""

import importlib.util
from dataclasses import dataclass
from itertools import accumulate

import pytest
import torch

from vllm.platforms import current_platform
from vllm.utils.platform_utils import num_compute_units
from vllm.v1.attention.ops import rocm_gqa_verify

QL = rocm_gqa_verify.QUERY_LEN
QUERY_LENS = list(range(rocm_gqa_verify.MIN_QUERY_LEN, QL + 1))
HD = rocm_gqa_verify.HEAD_SIZE
PAGES = list(rocm_gqa_verify.BLOCK_SIZES)
K_SCALE = 0.75
V_SCALE = 1.25


def _on_gfx950_with_flydsl() -> bool:
    if not current_platform.is_rocm() or importlib.util.find_spec("flydsl") is None:
        return False
    from vllm.platforms.rocm import on_gfx950

    return on_gfx950()


requires_gfx950 = pytest.mark.skipif(
    not _on_gfx950_with_flydsl(), reason="needs gfx950 and FlyDSL"
)


def _cache(num_pages: int, kv_heads: int, device: str, page: int = PAGES[0]):
    """K and V views of a `[pages, page, kv_heads, 64]` FP8 pool, K | V interleaved
    per head like the backend's cache."""
    pool = torch.zeros(
        (num_pages, kv_heads, page, 2 * HD), dtype=torch.float8_e4m3fn, device=device
    )
    k = pool[..., :HD].permute(0, 2, 1, 3)
    v = pool[..., HD:].permute(0, 2, 1, 3)
    return pool, k, v


def _fill(pool: torch.Tensor, pages: torch.Tensor, gen: torch.Generator) -> None:
    x = torch.randn((pages.numel(), *pool.shape[1:]), device=pool.device, generator=gen)
    pool[pages.long()] = (x * 0.5).to(torch.float8_e4m3fn)


def test_layout_supported_requires_16_or_32_token_fp8_pages():
    q = torch.zeros((QL, 32, HD), dtype=torch.bfloat16)
    sl = torch.tensor([128], dtype=torch.int32)
    bt = torch.zeros((1, 8), dtype=torch.int32)
    for page in PAGES:
        _, k, v = _cache(8, 4, "cpu", page)
        assert rocm_gqa_verify.layout_supported(q, k, v, q.clone(), sl, bt)

    k64 = v64 = torch.zeros((2, 64, 4, HD), dtype=torch.float8_e4m3fn)
    assert not rocm_gqa_verify.layout_supported(q, k64, v64, q.clone(), sl, bt)
    assert not rocm_gqa_verify.layout_supported(q.float(), k, v, q.clone(), sl, bt)
    assert not rocm_gqa_verify.layout_supported(q, k, v, q.clone(), sl.long(), bt)


@pytest.mark.skipif(
    importlib.util.find_spec("flydsl") is not None, reason="FlyDSL is installed"
)
def test_supports_layer_falls_back_without_flydsl():
    assert not rocm_gqa_verify.supports_layer(32, 4, HD, None, None, 0.0)


@pytest.mark.parametrize(
    "query_lens",
    [
        pytest.param([9, 9], id="too-long"),
        pytest.param([8, 8, 5, 8], id="fewer-drafts-k7"),
        pytest.param([6, 3, 6], id="fewer-drafts-k5"),
        pytest.param([4, 0, 4], id="empty-query-request"),
    ],
)
def test_fast_verify_declines_unsupported_query_lengths(query_lens):
    """Batches whose decode requests do not all have the same 1 to 8 query tokens
    are left to the AITER kernels: `fast_verify` returns False without touching the
    output (host metadata only, so this runs on CPU)."""
    n, tokens = len(query_lens), sum(query_lens)
    _, k, v = _cache(8, 4, "cpu")
    q = torch.zeros((tokens, 32, HD), dtype=torch.bfloat16)
    out = torch.full_like(q, 7.0)
    meta = _Meta(
        query_start_loc=torch.tensor([0, *accumulate(query_lens)], dtype=torch.int32),
        seq_lens=torch.full((n,), 64, dtype=torch.int32),
        block_table=torch.zeros((n, 4), dtype=torch.int32),
    )
    layer = _Layer(torch.tensor(K_SCALE), torch.tensor(V_SCALE))
    ran = rocm_gqa_verify.fast_verify(
        q, k, v, out, meta, n, tokens, max(query_lens), HD**-0.5, 0, layer
    )
    assert not ran
    assert (out == 7.0).all()


@dataclass
class _Meta:
    query_start_loc: torch.Tensor
    seq_lens: torch.Tensor
    block_table: torch.Tensor


@dataclass
class _Layer:
    _k_scale: torch.Tensor
    _v_scale: torch.Tensor


def _reference(q, k, v, lens, table, heads, kv_heads, window, scale, ql):
    out = torch.zeros(q.shape, dtype=torch.float32, device=q.device)
    group, page = heads // kv_heads, k.shape[1]
    for i, n in enumerate(lens):
        if n == 0:
            continue
        pages = table[i, : (n + page - 1) // page].long()
        kk = k[pages].float().reshape(-1, kv_heads, HD)[:n] * K_SCALE
        vv = v[pages].float().reshape(-1, kv_heads, HD)[:n].nan_to_num()
        qq = q[i * ql : (i + 1) * ql].float()
        s = torch.einsum("thd,lhd->htl", qq, kk.repeat_interleave(group, dim=1)) * scale
        pos = torch.arange(n, device=q.device)
        qpos = n - ql + torch.arange(ql, device=q.device)
        hidden = pos[None, None, :] > qpos[None, :, None]
        if window > 0:
            hidden |= qpos[None, :, None] - pos[None, None, :] >= window
        p = torch.softmax(s.masked_fill(hidden, float("-inf")), dim=-1)
        out[i * ql : (i + 1) * ql] = torch.einsum(
            "htl,lhd->thd", p, vv.repeat_interleave(group, dim=1)
        )
    return out * V_SCALE


@dataclass
class _Case:
    q: torch.Tensor
    k: torch.Tensor
    v: torch.Tensor
    out: torch.Tensor
    meta: _Meta
    layer: _Layer
    heads: int
    kv_heads: int
    window: int
    ql: int

    def run(self) -> bool:
        n = self.meta.seq_lens.numel()
        return rocm_gqa_verify.fast_verify(
            self.q,
            self.k,
            self.v,
            self.out,
            self.meta,
            n,
            n * self.ql,
            self.ql,
            HD**-0.5,
            self.window,
            self.layer,
        )

    def reference(self, lens) -> torch.Tensor:
        return _reference(
            self.q,
            self.k,
            self.v,
            lens,
            self.meta.block_table,
            self.heads,
            self.kv_heads,
            self.window,
            HD**-0.5,
            self.ql,
        )


def _make_case(
    lens,
    heads,
    kv_heads,
    window,
    ql=QL,
    num_pages=None,
    seed=0,
    page=PAGES[0],
    pin_last_page=False,
) -> _Case:
    device = "cuda"
    gen = torch.Generator(device=device).manual_seed(seed)
    total = sum((n + page - 1) // page for n in lens)
    num_pages = num_pages or max(2 * total, 64)
    pool, k, v = _cache(num_pages, kv_heads, device, page)
    perm = torch.randperm(num_pages - 1, device=device, generator=gen)[:total]
    if pin_last_page:
        perm[0] = num_pages - 1
    width = max(1, max((n + page - 1) // page for n in lens))
    table = torch.zeros((len(lens), width), dtype=torch.int32, device=device)
    offset = 0
    for i, n in enumerate(lens):
        pages = (n + page - 1) // page
        table[i, :pages] = perm[offset : offset + pages].to(torch.int32)
        offset += pages
    _fill(pool, perm, gen)
    # Stale bytes past each request's last token (NaN encodings) must not leak in.
    for i, n in enumerate(lens):
        if n % page:
            last = table[i, n // page].long()
            k[last, n % page :] = float("nan")
            v[last, n % page :] = float("nan")
    q = (torch.randn((len(lens) * ql, heads, HD), device=device, generator=gen) * 2).to(
        torch.bfloat16
    )
    meta = _Meta(
        query_start_loc=torch.arange(len(lens) + 1, device=device, dtype=torch.int32)
        * ql,
        seq_lens=torch.tensor(lens, dtype=torch.int32, device=device),
        block_table=table,
    )
    layer = _Layer(
        torch.tensor(K_SCALE, device=device), torch.tensor(V_SCALE, device=device)
    )
    out = torch.full(q.shape, float("nan"), dtype=torch.bfloat16, device=device)
    return _Case(q, k, v, out, meta, layer, heads, kv_heads, window, ql)


def _rel_l2(got: torch.Tensor, ref: torch.Tensor) -> float:
    return ((got.float() - ref.float()).norm() / ref.float().norm()).item()


def _check(out: torch.Tensor, ref: torch.Tensor, lens, ql: int) -> None:
    live = torch.tensor([n > 0 for n in lens for _ in range(ql)], device=out.device)
    got = out.float()
    assert torch.isfinite(got).all()
    # Zero-length requests (CUDA-graph padding) are written as zeros.
    assert (got[~live] == 0).all()
    rel_l2 = _rel_l2(got[live], ref[live])
    assert rel_l2 < 5e-3, f"relative L2 error {rel_l2:.2e}"


def _fits(num_requests: int, window: int) -> bool:
    """Full attention takes fewer than two requests per CU."""
    return window > 0 or num_requests < 2 * num_compute_units()


LENGTH_CASES = [
    pytest.param([4100], id="single"),
    pytest.param([70000, 2049], id="pair"),
    pytest.param([8, 9, 31, 33, 40, 2, 3], id="short"),
    pytest.param([1000, 5000, 37, 70000], id="mixed"),
    pytest.param([100000] + [3000] * 11, id="one-long"),
    pytest.param([2047, 2048, 2049, 2056, 4096, 123457, 64, 65], id="window-edges"),
    pytest.param([0, 5000, 12345, 0], id="zero-length-padding"),
    pytest.param([50000] * 24, id="uniform-24"),
    pytest.param([300] * 130, id="many-requests"),
]


@requires_gfx950
@pytest.mark.parametrize("window", [0, 2048], ids=["full", "swa"])
@pytest.mark.parametrize("heads,kv_heads", [(32, 4), (16, 2)])
@pytest.mark.parametrize("lens", LENGTH_CASES)
@pytest.mark.parametrize("ql", QUERY_LENS)
@pytest.mark.parametrize("page", PAGES)
def test_fast_verify_matches_float32_reference(page, ql, lens, heads, kv_heads, window):
    lens = [max(n, ql) if n else 0 for n in lens]
    case = _make_case(lens, heads, kv_heads, window, ql, page=page)
    if window > 0:
        # Pages that slid out of the window may hold anything.
        for i, n in enumerate(lens):
            stale = case.meta.block_table[i, : max(0, n - ql - window + 1) // page]
            case.k[stale.long()] = float("nan")
            case.v[stale.long()] = float("nan")
    if not _fits(len(lens), window):
        assert not case.run()
        return
    assert case.run()
    torch.accelerator.synchronize()
    _check(case.out, case.reference(lens), lens, ql)


@requires_gfx950
@pytest.mark.parametrize("window", [0, 2048], ids=["full", "swa"])
@pytest.mark.parametrize("ql", [3, QL])
@pytest.mark.parametrize("page", PAGES)
def test_fast_verify_with_a_pool_beyond_32_bit_offsets(page, ql, window):
    """A pool spanning more than 2^31 bytes takes the 64-bit addressing path."""
    kv_heads = 4
    pages = 2**31 // (page * kv_heads * 2 * HD) + 4096
    lens = [40000, 9000, 70]
    case = _make_case(
        lens, 32, kv_heads, window, ql, num_pages=pages, page=page, pin_last_page=True
    )
    assert case.run()
    torch.accelerator.synchronize()
    _check(case.out, case.reference(lens), lens, ql)


@requires_gfx950
@pytest.mark.parametrize("window", [0, 2048], ids=["full", "swa"])
@pytest.mark.parametrize("n", [16, 24, 130])
@pytest.mark.parametrize("ql", QUERY_LENS)
@pytest.mark.parametrize("page", PAGES)
def test_fast_verify_cudagraph_replay_follows_new_lengths(page, ql, n, window):
    """The grid and the balance map are fixed per batch size (16 and 24 requests
    take one and two workgroups per CU): a captured graph must follow `seq_lens`
    written into the same buffer afterwards."""
    if not _fits(n, window):
        pytest.skip("too many requests for the full-attention grid on this device")
    longest = 40000 if n <= 24 else 3000
    case = _make_case([longest] * n, 32, 4, window, ql, page=page)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        assert case.run()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        assert case.run()
    # The block table covers `longest` tokens per request; shorter lengths stay inside.
    for lens in (
        [longest] + [longest // 40] * (n - 1),
        [s * 1013 % (longest - 17) + 17 for s in range(1, n + 1)],
    ):
        case.meta.seq_lens.copy_(torch.tensor(lens, dtype=torch.int32))
        graph.replay()
        torch.accelerator.synchronize()
        _check(case.out, case.reference(lens), lens, ql)


@requires_gfx950
@pytest.mark.parametrize("window", [0, 2048], ids=["full", "swa"])
@pytest.mark.parametrize(
    "lens", [[70000, 2049, 5000, 9], [3000] * 24], ids=["four", "uniform-24"]
)
@pytest.mark.parametrize("kv_heads", [1, 8, rocm_gqa_verify.MAX_KV_HEADS])
@pytest.mark.parametrize("ql", [1, 3, QL])
def test_fast_verify_other_kv_heads_per_rank(ql, kv_heads, lens, window):
    """KV heads per rank set the workgroup size (one wave each)."""
    case = _make_case(lens, rocm_gqa_verify.GQA * kv_heads, kv_heads, window, ql)
    assert case.run()
    torch.accelerator.synchronize()
    _check(case.out, case.reference(lens), lens, ql)


@requires_gfx950
@pytest.mark.parametrize("window", [None, 100], ids=["full", "swa"])
@pytest.mark.parametrize("ql", [1, 4], ids=["decode", "verify"])
def test_backend_runs_uniform_decode_batches_on_the_kernels(monkeypatch, ql, window):
    """Through the AITER FA backend: a head-64, GQA-8 layer with an FP8 cache takes
    the kernels for a uniform decode batch, plain (no speculative config) or verify
    (builder map, forward hook), and matches the float32 reference."""
    from tests.v1.attention.utils import (
        BatchSpec,
        create_common_attn_metadata,
        create_standard_kv_cache_spec,
        create_vllm_config,
    )
    from vllm.config import SpeculativeConfig, set_current_vllm_config
    from vllm.model_executor.layers.attention import Attention
    from vllm.v1.attention.backends.rocm_aiter_fa import (
        AiterFlashAttentionBackend,
        AiterFlashAttentionImpl,
        AiterFlashAttentionMetadataBuilder,
    )

    page, num_blocks = PAGES[-1], 512
    config = create_vllm_config(
        model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        dtype="bfloat16",
        block_size=page,
        max_model_len=2048,
    )
    config.cache_config.cache_dtype = "fp8"
    if ql > 1:
        config.speculative_config = SpeculativeConfig(
            method="ngram", num_speculative_tokens=ql - 1
        )
    heads = config.model_config.get_num_attention_heads(config.parallel_config)
    kv_heads = config.model_config.get_num_kv_heads(config.parallel_config)
    head_size = config.model_config.get_head_size()
    with set_current_vllm_config(config), torch.device("cuda"):
        layer = Attention(
            heads,
            head_size,
            head_size**-0.5,
            kv_heads,
            cache_config=config.cache_config,
            per_layer_sliding_window=window,
            prefix="layer",
            attn_backend=AiterFlashAttentionBackend,
        )
        builder = AiterFlashAttentionMetadataBuilder(
            create_standard_kv_cache_spec(config),
            ["layer"],
            config,
            torch.device("cuda"),
        )
    impl = layer.impl
    assert isinstance(impl, AiterFlashAttentionImpl) and impl._fast_verify
    layer._k_scale.fill_(K_SCALE)
    layer._v_scale.fill_(V_SCALE)
    lens = [2000, 70, 1100, 9]
    batch = BatchSpec(seq_lens=lens, query_lens=[ql] * len(lens))
    common = create_common_attn_metadata(
        batch, page, torch.device("cuda"), max_block_idx=num_blocks
    )
    metadata = builder.build(0, common)
    decode = metadata.decode_metadata
    assert metadata.num_decodes == len(lens) and decode is not None
    assert (decode.verify_map is None) == (window is not None)
    # The kernel applies the window itself: no trimmed block tables.
    assert decode.swa_block_table is None
    kv_cache, k, v = _cache(num_blocks, kv_heads, "cuda", page)
    gen = torch.Generator("cuda").manual_seed(0)
    _fill(kv_cache, torch.arange(num_blocks, device="cuda"), gen)
    query = torch.randn(
        (len(lens) * ql, heads, head_size), device="cuda", generator=gen
    ).to(torch.bfloat16)

    ran = []
    fast_verify = rocm_gqa_verify.fast_verify

    def spy(*args):
        ran.append(fast_verify(*args))
        return ran[-1]

    monkeypatch.setattr(rocm_gqa_verify, "fast_verify", spy)
    out = torch.full_like(query, float("nan"))
    impl.forward(layer, query, None, None, kv_cache, metadata, out)
    assert ran == [True]
    expected = _reference(
        query,
        k,
        v,
        lens,
        common.block_table_tensor,
        heads,
        kv_heads,
        window or 0,
        head_size**-0.5,
        ql,
    )
    # A window off by one key or a missed KV chunk does not fit this bound.
    rel_l2 = _rel_l2(out, expected)
    assert rel_l2 < 5e-3, f"relative L2 error {rel_l2:.2e}"
