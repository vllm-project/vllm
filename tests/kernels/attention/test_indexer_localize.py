# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Die-local sparse-attention indexer decode (`VLLM_INDEXER_LOCALIZE`).

The merge, join and build kernel tests run on CUDA GPUs (the build kernel's
reference also needs DeepGEMM). The placement and scan tests need a GPU with
two memory locality domains, DeepGEMM and `torch.cuda.execute_on_streams`.
"""

import math

import pytest
import torch

from vllm.platforms import current_platform

if not current_platform.is_cuda():
    pytest.skip("CUDA only", allow_module_level=True)

from vllm.config import CacheConfig, VllmConfig, set_current_vllm_config  # noqa: E402
from vllm.model_executor.layers import indexer_overlap  # noqa: E402
from vllm.model_executor.layers.indexer_topk import get_indexer_topk  # noqa: E402
from vllm.model_executor.layers.sparse_attn_indexer import (  # noqa: E402
    as_fp32_logits,
    kv_cache_as_quant_view,
)
from vllm.utils.import_utils import has_deep_gemm  # noqa: E402
from vllm.v1.attention.ops.indexer_localize import build, placement, scan  # noqa: E402
from vllm.v1.kv_cache_interface import KVCacheConfig, KVCacheTensor  # noqa: E402

UNIT_UE8M0_X4 = 0x7F7F7F7F


def _device() -> torch.device:
    return torch.device("cuda", torch.accelerator.current_device_index())


@pytest.fixture(scope="module")
def domains() -> placement.DomainResources | None:
    if not has_deep_gemm():
        pytest.skip("needs DeepGEMM")
    torch.empty(1, device=_device())  # the green contexts need a CUDA context
    res = placement.get_domain_resources(_device())
    if res is None:
        pytest.skip("needs a GPU with two memory locality domains")
    from vllm.v1.worker.workspace import init_workspace_manager

    init_workspace_manager(_device())
    return res


def _row_pages(rows: int, pages: int, page_rows: int, gen: torch.Generator):
    """Metadata for the merge and join kernels: each row's pages split between
    the domains, as local-to-row page maps and the join map ((local page << 1)
    | domain per row page)."""
    device = _device()
    order = torch.argsort(torch.rand(rows, 2 * pages, generator=gen, device=device))
    inv = [order[:, :pages].int().contiguous(), order[:, pages:].int().contiguous()]
    jmap = torch.empty(rows, 2 * pages, dtype=torch.int32, device=device)
    local = torch.arange(pages, dtype=torch.int32, device=device).expand(rows, -1)
    for d in range(2):
        jmap.scatter_(1, inv[d].long(), local * 2 + d)
    # The kernels read only the device and its SM count (the join's grid).
    sms = torch.cuda.get_device_properties(device).multi_processor_count
    meta = build.LocalizedDecodeMetadata(
        page_rows=page_rows,
        max_rows=rows,
        block_tables=[],
        seq_lens=[],
        schedules=[],
        join_map=jmap,
        inv_maps=inv,
        resources=placement.DomainResources(device.index, [], [sms // 2] * 2, sms),
    )
    return inv, meta


@pytest.fixture(scope="module")
def nvrtc() -> None:
    """The merge and build kernels are compiled at runtime with NVRTC."""
    bindings = pytest.importorskip("cuda.bindings.nvrtc", exc_type=ImportError)
    try:
        bindings.nvrtcVersion()
    except RuntimeError as e:  # no NVRTC library
        pytest.skip(f"needs NVRTC ({e})")


# ---------------------------------------------------------------------------
# Kernels
# ---------------------------------------------------------------------------
def _reference_merge(
    scores: list[torch.Tensor],
    idx: list[torch.Tensor],
    inv: list[torch.Tensor],
    page_rows: int,
    topk: int,
) -> torch.Tensor:
    """The K largest of both domains' candidates (ties at the K-th largest
    taken in candidate order: domain 0's, then domain 1's) as request-local
    token positions in candidate order, then -1."""
    keys, positions = [], []
    for d in range(2):
        cols = idx[d].long().clamp(min=0)
        page = inv[d].long().gather(1, cols // page_rows)
        positions.append((page * page_rows + cols % page_rows).int())
        # Order-preserving integer keys of the scores' bits.
        s = scores[d].gather(1, cols)
        if s.dtype == torch.float32:
            bits, magnitude = s.view(torch.int32), 0x7FFFFFFF
        else:
            bits, magnitude = s.view(torch.int16).int(), 0x7FFF
        keys.append(torch.where(bits >= 0, bits, bits ^ magnitude).long())
    valid = torch.cat(idx, 1) >= 0
    key = torch.where(valid, torch.cat(keys, 1), -(2**40))
    order = key.argsort(dim=1, descending=True, stable=True)
    kept = torch.zeros_like(valid).scatter_(1, order[:, :topk], True) & valid
    pos = torch.cat(positions, 1)
    ref = torch.full_like(idx[0], -1)
    for r, row_kept in enumerate(kept):
        ref[r, : int(row_kept.sum())] = pos[r][row_kept]
    return ref


@pytest.mark.parametrize(
    "topk,score_dtype,page_rows",
    [(2048, torch.float32, 64), (512, torch.bfloat16, 64), (512, torch.bfloat16, 128)],
    ids=["kimi_k3", "deepseek_v4_c2", "deepseek_v4_c1"],
)
def test_merge_kernel_matches_reference(nvrtc, topk, score_dtype, page_rows):
    """The CUDA merge keeps the K largest of both domains' candidates, ties in
    candidate order, at their request-local token positions."""
    device = _device()
    gen = torch.Generator(device=device).manual_seed(topk + page_rows)
    rows, pages = 8, 4 * topk // page_rows
    width = pages * page_rows
    inv, loc = _row_pages(rows, pages, page_rows, gen)
    # Few distinct scores, so the K-th largest is tied.
    scores = [
        (torch.randint(-64, 64, (rows, width), generator=gen, device=device) / 8).to(
            score_dtype
        )
        for _ in range(2)
    ]
    idx = []
    for d in range(2):
        cols = torch.rand(rows, width, generator=gen, device=device).argsort(dim=1)
        cols = cols[:, :topk].int()
        # Short rows have fewer than K candidates.
        cols[d, topk // 3 :] = -1
        idx.append(cols.contiguous())
    out = torch.full((rows, topk), -7, dtype=torch.int32, device=device)
    scan.merge_candidates(loc, scores[0], scores[1], idx, topk, out)
    assert torch.equal(out, _reference_merge(scores, idx, inv, page_rows, topk))

    # The kept scores are the K largest candidate scores.
    full = torch.full((rows, 2 * width + 1), -math.inf, device=device)
    for d in range(2):
        valid = idx[d] >= 0
        cols = idx[d].clamp(min=0).long()
        page = inv[d].long().gather(1, cols // page_rows)
        # Invalid candidates go to the spare last column.
        pos = torch.where(valid, page * page_rows + cols % page_rows, 2 * width)
        vals = scores[d].float().gather(1, cols)
        full.scatter_(1, pos, torch.where(valid, vals, -math.inf))
    expected = full.sort(dim=1, descending=True).values[:, :topk]
    kept = torch.where(out >= 0, full.gather(1, out.clamp(min=0).long()), -math.inf)
    torch.testing.assert_close(kept.sort(dim=1, descending=True).values, expected)


@pytest.mark.parametrize("far", [False, True], ids=["near", "2_31_apart"])
@pytest.mark.parametrize("out_dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("page_rows", [64, 128])
def test_join_matches_reference(page_rows, out_dtype, far):
    """Both domains' compact scores land at their row positions bit for bit,
    for output rows with and without 16-byte alignment; columns at or past a
    row's length stay unwritten. `far` starts domain 1's scores just under
    2**31 elements after domain 0's, so their distance plus a column passes
    2**31."""
    device = _device()
    gen = torch.Generator(device=device).manual_seed(page_rows)
    rows, pages = 6, 40
    width = 2 * pages * page_rows
    stride = math.ceil(width / 1536) * 1536  # DeepGEMM's row alignment
    inv, loc = _row_pages(rows, pages, page_rows, gen)
    compact = [
        torch.randn(rows, stride, generator=gen, device=device).to(torch.bfloat16)
        for _ in range(2)
    ]
    if far:
        delta = 2**31 - 1024  # a multiple of 16, as allocations are
        assert delta + pages * page_rows > 2**31
        big = torch.empty(delta + rows * stride, dtype=torch.bfloat16, device=device)
        for d, offset in enumerate((0, delta)):
            dst = big[offset : offset + rows * stride].view(rows, stride)
            compact[d] = dst.copy_(compact[d])
    compact = [c[:, :width] for c in compact]
    lens = torch.tensor(
        [0, 1, page_rows, 7 * page_rows + 5, width - 3, width],
        dtype=torch.int32,
        device=device,
    )
    ref = torch.full((rows, width), -9.0, dtype=out_dtype, device=device)
    for r in range(rows):
        for d in range(2):
            for lp, p in enumerate(inv[d][r].tolist()):
                lo, hi = p * page_rows, min((p + 1) * page_rows, int(lens[r]))
                if lo < hi:
                    src = compact[d][r, lp * page_rows : lp * page_rows + hi - lo]
                    ref[r, lo:hi] = src.to(out_dtype)
    for out_stride in (stride, width + 1):  # 16-byte aligned rows or not
        out = torch.full((rows, out_stride), -9.0, dtype=out_dtype, device=device)
        out = out[:, :width]
        scan.localized_join_scores(loc, compact[0], compact[1], out, lens)
        assert torch.equal(out, ref)


def _decode_rows(
    rows: int,
    width: int,
    page_rows: int,
    num_pages: int,
    rows_per_request: int,
    gen: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Block table, lengths and request indices of `rows` decode rows in the
    varlen layout: requests of up to `rows_per_request` consecutive rows (at
    least two when allowed) share a block-table row and an index, each row
    seeing one key more than the one before. The first requests are empty,
    end one key in and fill the table; the last row of every third one reads
    only the first key of a page. The last rows, a varying number, are
    padding: length 0 and an index each."""
    device = gen.device

    def randint(lo: int, hi: int) -> int:
        return int(torch.randint(lo, hi, (), generator=gen, device=device))

    real = rows - randint(0, rows // 4 + 1)
    bt = torch.zeros(rows, width, dtype=torch.int32, device=device)
    lens = torch.zeros(rows, dtype=torch.int32, device=device)
    ids = torch.randperm(4 * rows, generator=gen, device=device).int()
    indices = ids[:rows].clone()  # padding rows keep one each
    start = req = 0
    while start < real:
        end = min(start + randint(min(2, rows_per_request), rows_per_request + 1), real)
        bt[start:end] = torch.randint(
            1, num_pages, (width,), generator=gen, device=device
        )
        if req < 3:
            last = (0, 1, width * page_rows)[req]
        elif req % 3 == 0:
            last = randint(1, width) * page_rows + 1
        else:
            last = randint(0, width * page_rows + 1)
        before = torch.arange(end - start - 1, -1, -1, device=device)
        lens[start:end] = (last - before).clamp(min=0)
        indices[start:end] = ids[rows + req]
        start, req = end, req + 1
    return bt, lens, indices


def _reference_build(
    bt: torch.Tensor,
    lens: torch.Tensor,
    dom: torch.Tensor,
    indices: torch.Tensor,
    page_rows: int,
    sms: list[int],
    out: dict[str, list[torch.Tensor]],
) -> None:
    """What `build.cu` writes, into the CPU tensors `out`. Consecutive rows
    with the same index are one request: each row gets its request's pages up
    to the request's last row, split by domain in page order (compact block
    tables followed by 64 zeros, local-to-request page maps, the join map),
    and its own per-domain lengths (its pages are a prefix of the request's).
    Rows past `lens` are reset; DeepGEMM schedules each domain's lengths."""
    from vllm.utils.deep_gemm import get_paged_mqa_logits_metadata

    cap, rows = out["bt"][0].shape[0], lens.numel()
    # Rows past `lens` are empty.
    bt = torch.nn.functional.pad(bt.cpu(), (0, 0, 0, cap - rows))
    row_lens = torch.nn.functional.pad(lens.cpu().view(-1), (0, cap - rows))
    ids = indices.cpu().tolist()
    dom = dom.cpu()
    for r in range(cap):
        e = r
        while r < rows and e + 1 < rows and ids[e + 1] == ids[r]:
            e += 1
        n = int(row_lens[e])  # the request's last row: its split length
        own = min(int(row_lens[r]), n)
        npages, own_pages = -(-n // page_rows), -(-own // page_rows)
        on1 = dom[bt[r, :npages].long()] != 0
        short = own_pages * page_rows - own  # keys missing from its last page
        for d in range(2):
            pages = (on1 == d).nonzero().view(-1)
            k = pages.numel()
            out["bt"][d][r, :k] = bt[r, pages]
            out["bt"][d][r, k : k + 64] = 0
            out["inv"][d][r, :k] = pages
            local = torch.arange(k, dtype=torch.int32)
            out["jmap"][0][r, pages] = local * 2 + d
            mine = int((on1[:own_pages] == d).sum())
            last = own_pages > 0 and bool(on1[own_pages - 1]) == bool(d)
            out["lens"][d][r] = mine * page_rows - (short if last else 0)
    for d, num_sms in enumerate(sms):
        meta = get_paged_mqa_logits_metadata(
            out["lens"][d][:rows].to(indices.device),
            page_rows,
            num_sms,
            indices=indices,
        )
        out["sched"][d][: meta.shape[0]] = meta.cpu()


@pytest.mark.parametrize("rows_per_request", [1, 6], ids=["one_row", "2_to_6_rows"])
@pytest.mark.parametrize("rows", [19, 40])
@pytest.mark.parametrize("domain_plan", [True, False])
def test_build_kernel_matches_reference(nvrtc, domain_plan, rows, rows_per_request):
    """The one-launch build writes what the torch split and DeepGEMM's
    schedules write, including in a CUDA graph replayed over new inputs. With
    several rows per request (speculative decoding), every row gets its
    request's pages up to the request's last row and its own lengths, also
    where a later row starts a new page."""
    if not (has_deep_gemm() and current_platform.is_device_capability_family(100)):
        pytest.skip("needs DeepGEMM's SM100 paged MQA logits schedule")
    device = _device()
    gen = torch.Generator(device=device).manual_seed(int(domain_plan) + rows)
    cap, width, page_rows, num_pages = rows + 5, 2300, 64, 40000
    pages = torch.arange(num_pages, device=device)
    if domain_plan:
        q0, k = 11, 37
        dom = (((q0 + pages) // k) % 2).to(torch.int8)
    else:
        q0, k = 0, 0
        dom = torch.randint(0, 2, (num_pages,), generator=gen, device=device)
        dom = dom.to(torch.int8)
    sms = [100, 100]

    def int32(*shape: int) -> torch.Tensor:
        return torch.randint(
            0, 1 << 20, shape, generator=gen, device=device, dtype=torch.int32
        )

    def inputs():
        return _decode_rows(rows, width, page_rows, num_pages, rows_per_request, gen)

    bt, lens, indices = inputs()
    # Both builds start from the same junk, so unwritten entries compare too.
    junk = dict(
        bt=[int32(cap, width) for _ in range(2)],
        lens=[int32(cap, 1) for _ in range(2)],
        inv=[int32(cap, width) for _ in range(2)],
        sched=[int32(s + 1, 2) for s in sms],
        jmap=[int32(cap, width)],
    )
    out = {name: [t.clone() for t in ts] for name, ts in junk.items()}
    ref = {name: [t.cpu() for t in ts] for name, ts in junk.items()}
    args = build._GroupArgs(
        bt=bt,
        lens=lens,
        dom=dom,
        indices=indices,
        lbt=out["bt"],
        inv=out["inv"],
        jmap=out["jmap"][0],
        llen=out["lens"],
        sched=out["sched"],
        flags=torch.zeros(cap * 3, dtype=torch.int64, device=device),
        state=torch.zeros(4, dtype=torch.int32, device=device),
        R=page_rows,
        width=width,
        cap=cap,
        num_rows=rows,
        dom_q0=q0,
        dom_k=k,
    )

    def check() -> None:
        _reference_build(bt, lens, dom, indices, page_rows, sms, ref)
        for name, tensors in out.items():
            for got, want in zip(tensors, ref[name]):
                assert torch.equal(got.cpu(), want), name

    build._launch_build([args], sms, device.index)
    check()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        build._launch_build([args], sms, device.index)
    for _ in range(20):
        for t, new in zip((bt, lens, indices), inputs()):
            t.copy_(new)
        graph.replay()
        check()


# Clusters of 16, 8, 4, 2 and 1 top-k CTAs a VR200 locality domain (100 SMs)
# holds at once, as CUDA's occupancy query reports.
_VR200_DOMAIN = {16: 4, 8: 11, 4: 23, 2: 50, 1: 100}


def test_policy_takes_die_split_in_one_round_of_topk_ctas():
    """A batch takes the die split when each domain holds all of its top-k's
    clusters (by the stock selector's launch rule) at once."""
    assert [build.topk_ctas_per_row(n) for n in (1, 4, 5, 8, 9, 33, 34, 64, 65)] == [
        16, 16, 8, 8, 4, 4, 2, 2, 1,
    ]  # fmt: skip
    taken = [n for n in range(1, 200) if build.use_die_split(n, _VR200_DOMAIN)]
    assert taken == [*range(1, 24), *range(34, 51), *range(65, 101)]
    # DeepSeek-V4.1 with 16 requests and 1 to 6 tokens per request.
    assert [build.use_die_split(16 * t, _VR200_DOMAIN) for t in range(1, 7)] == [
        True, False, True, False, True, True,
    ]  # fmt: skip
    assert build.use_die_split(4, {**_VR200_DOMAIN, 16: 0}) is False


def test_topk_cluster_capacity_of_the_domains(domains):
    """The occupancy query per domain: clusters never hold more CTAs than the
    domain's SMs, and single-CTA clusters fill every SM (one CTA per SM)."""
    capacity = build.topk_cluster_capacity(domains)
    assert set(capacity) == {16, 8, 4, 2, 1}
    for cluster, n in capacity.items():
        assert n * cluster <= min(domains.sms), (cluster, n)
    assert capacity[1] == min(domains.sms)


# ---------------------------------------------------------------------------
# Placement and the die-split decode (two locality domains)
# ---------------------------------------------------------------------------
def _cache_config(num_blocks: int, page_bytes: int, layers: int) -> KVCacheConfig:
    """A block-outermost KV cache: block b holds every layer's page b."""
    block_bytes = math.ceil(layers * page_bytes / 512) * 512
    tensor = KVCacheTensor(
        size=num_blocks * block_bytes,
        layers=[f"layers.{i}.indexer.k_cache" for i in range(layers)],
        layer_stride=page_bytes,
        block_stride=block_bytes,
    )
    return KVCacheConfig(num_blocks, [tensor], [])


def _layer1_pages(
    buf: torch.Tensor, config: KVCacheConfig, page_rows: int, row_bytes: int
) -> torch.Tensor:
    """Layer 1's pages of a `_cache_config` cache: `[blocks, rows, bytes]`."""
    return torch.as_strided(
        buf.view(torch.uint8),
        (config.num_blocks, page_rows, row_bytes),
        (config.kv_cache_tensors[0].block_stride, row_bytes, 1),
        storage_offset=page_rows * row_bytes,
    )


@pytest.mark.parametrize(
    "arch,opted_in",
    [
        ("KimiK31ForConditionalGeneration", True),
        ("DeepseekV41ForCausalLM", True),
        ("DeepseekV32ForCausalLM", False),
    ],
)
def test_models_opt_in_to_die_local_indexer(arch, opted_in):
    """The die-local indexer applies to models whose class (as the registry
    resolves the architecture) opts in; no GPU needed."""
    from vllm.model_executor.models import ModelRegistry

    model_cls = ModelRegistry._try_load_model_cls(arch)
    assert model_cls is not None
    assert getattr(model_cls, "supports_localized_indexer", False) is opted_in


@pytest.mark.parametrize("extensible", [False, True], ids=["plain", "extensible"])
def test_kv_cache_granules_live_in_their_planned_domain(domains, extensible):
    """Every committed granule of the KV cache is in the locality domain of
    its midpoint's run, and the page-to-domain table follows the plan."""
    from cuda.bindings import driver as cu

    from vllm.utils.extensible_tensor import granule_size
    from vllm.v1.kv_cache_layout import KVCacheLayout
    from vllm.v1.worker.extensible_kv_cache import ExtensibleKVCache

    domain_type = cu.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE_LOCALITY_DOMAIN
    device = _device()
    page_rows, row_bytes = 64, 132
    config = _cache_config(6000, page_rows * row_bytes, 4)
    size = config.kv_cache_tensors[0].size
    kv_cache = None
    if extensible:
        kv_cache = ExtensibleKVCache(config, device, KVCacheLayout.BLHNC)
    allocate = placement.localized_kv_allocate(
        config, device, None if kv_cache is None else kv_cache.allocate
    )
    assert allocate is not None
    buf = allocate(size)
    if kv_cache is not None:
        kv_cache.commit(config.num_blocks)
    found = placement.find_buffer(buf.data_ptr())
    assert found is not None
    base, _, run_bytes, _ = found
    block_bytes = config.kv_cache_tensors[0].block_stride
    assert run_bytes % block_bytes == 0

    granule = granule_size(device.index)
    for g in range(size // granule):
        handle = cu.cuMemRetainAllocationHandle(base + g * granule + granule // 2)[1]
        location = cu.cuMemGetAllocationPropertiesFromHandle(handle)[1].location
        cu.cuMemRelease(handle)
        planned = ((g * granule + granule // 2) // run_bytes) % 2
        assert location.type == domain_type, g
        assert location.localized.localityDomainId == planned, g
    buf.fill_(1)
    assert int(buf.view(torch.uint8).sum()) == size

    view = _layer1_pages(buf, config, page_rows, row_bytes)
    table = placement.page_domain_table(view)
    assert table is not None
    mid = view.data_ptr() - base + torch.arange(config.num_blocks) * block_bytes
    mid += page_rows * row_bytes // 2
    assert torch.equal(table.cpu(), ((mid // run_bytes) % 2).to(torch.int8))
    q0, k = placement.page_domain_plan(view)
    pages = torch.arange(config.num_blocks, device=device)
    assert k > 0 and torch.equal((((q0 + pages) // k) % 2).to(torch.int8), table)


def _fill_index_cache(kv: torch.Tensor, key_bytes: int, gen: torch.Generator):
    """Random valid index keys (FP8 without NaN codes, or any MXFP4 byte)
    with unit UE8M0 scales."""
    pages, page_rows, row_bytes = kv.shape
    flat = torch.as_strided(kv, (pages, page_rows * row_bytes), (kv.stride(0), 1))
    data = torch.randint(
        0, 256, flat.shape, dtype=torch.uint8, generator=gen, device=kv.device
    )
    if key_bytes == 128:  # FP8: no NaN
        data.masked_fill_((data & 0x7F) == 0x7F, 0)
    data[:, page_rows * key_bytes :] = 0x7F
    flat.copy_(data)


class _DecodeCase:
    """A localized index-K cache and one decode batch over it, with the
    stock (single-launch) scan for comparison. Each request has
    `rows_per_request` consecutive rows (query tokens) sharing its pages, each
    seeing one key more than the one before; in every other request the last
    row reads only the first key of a page."""

    def __init__(
        self,
        domains: placement.DomainResources,
        fp4: bool,
        page_rows: int,
        requests: int,
        rows_per_request: int,
    ):
        from vllm.model_executor.models.deepseek_v2 import DeepseekV32IndexerCache
        from vllm.utils.deep_gemm import (
            fp8_fp4_paged_mqa_logits,
            get_num_sms,
            get_paged_mqa_logits_metadata,
            mqa_logits_mx_inputs,
        )

        device = _device()
        rows = requests * rows_per_request
        gen = torch.Generator(device=device).manual_seed(rows + page_rows)
        heads, dim, context = 32, 128, 96 * 1024
        key_bytes = dim // 2 if fp4 else dim
        row_bytes = key_bytes + 4
        pages = math.ceil(context / page_rows)
        self.max_model_len = 2 * context
        self.width = math.ceil(self.max_model_len / page_rows)
        num_blocks = requests * pages + 2
        config = _cache_config(num_blocks, page_rows * row_bytes, 4)
        allocate = placement.localized_kv_allocate(config, device, None)
        assert allocate is not None
        self.buffer = allocate(config.kv_cache_tensors[0].size)
        kv = _layer1_pages(self.buffer, config, page_rows, row_bytes)
        _fill_index_cache(kv, key_bytes, gen)
        self.kv = kv_cache_as_quant_view(kv, dim, fp4)
        vllm_config = VllmConfig()
        with set_current_vllm_config(vllm_config):
            cache = DeepseekV32IndexerCache(
                row_bytes, torch.uint8, "layers.1.indexer.k_cache", CacheConfig()
            )
        cache.bind_kv_cache(kv.unsqueeze(1))
        blocks = torch.randperm(num_blocks - 1, generator=gen, device=device) + 1
        self.block_table = torch.zeros(
            requests, self.width, dtype=torch.int32, device=device
        )
        self.block_table[:, :pages] = blocks[: requests * pages].view(-1, pages).int()
        last = torch.randint(
            context // 2, context + 1, (requests, 1), generator=gen, device=device
        )
        last[::2] = (last[::2] - 1) // page_rows * page_rows + 1
        before = torch.arange(rows_per_request - 1, -1, -1, device=device)
        self.seq_lens = (last - before).view(-1, 1).int()
        self.block_table = self.block_table.repeat_interleave(rows_per_request, 0)
        self.indices = torch.arange(
            1, 2 * requests, 2, dtype=torch.int32, device=device
        ).repeat_interleave(rows_per_request)
        self.localizer = build.IndexerDecodeLocalizer(
            [cache.prefix],
            vllm_config.compilation_config.static_forward_context,
            domains,
            rows,
            self.width,
            page_rows,
        )
        if fp4:
            self.q_values = torch.randint(
                -128, 128, (rows, 1, heads, dim // 2), dtype=torch.int8, device=device
            )
            q_scale = torch.full(
                (rows, 1, heads), UNIT_UE8M0_X4, dtype=torch.int32, device=device
            )
        else:
            self.q_values = torch.randn(rows, 1, heads, dim, device=device).to(
                torch.float8_e4m3fn
            )
            q_scale = None
        self.logits_dtype = torch.bfloat16 if fp4 else torch.float32
        self.q_scale, self.weights = mqa_logits_mx_inputs(
            self.q_values,
            q_scale,
            torch.randn(rows, heads, device=device) * 0.05,
            self.logits_dtype,
        )
        schedule = get_paged_mqa_logits_metadata(
            self.seq_lens, page_rows, get_num_sms(), indices=self.indices
        )
        self.stock = lambda: fp8_fp4_paged_mqa_logits(
            (self.q_values, self.q_scale),
            self.kv,
            self.weights,
            self.seq_lens,
            self.block_table,
            schedule,
            self.max_model_len,
            logits_dtype=self.logits_dtype,
            indices=self.indices,
        )

    @property
    def q(self) -> tuple[torch.Tensor, torch.Tensor | None]:
        return self.q_values, self.q_scale

    def localize(self, rows: int) -> build.LocalizedDecodeMetadata | None:
        return self.localizer.build(self.block_table, self.seq_lens, self.indices, rows)

    def join(self, loc: build.LocalizedDecodeMetadata, fp32: bool) -> torch.Tensor:
        """The die-split scan and join, FP32 or in the scan's dtype."""
        return scan.localized_paged_mqa_logits(
            loc,
            self.q,
            self.kv,
            self.weights,
            self.seq_lens,
            self.max_model_len,
            self.logits_dtype,
            self.indices,
            lambda t: fp32,
        )


def _assert_same_scores(out: torch.Tensor, ref: torch.Tensor, lens: torch.Tensor):
    assert out.shape == ref.shape and out.stride() == ref.stride()
    assert out.dtype == ref.dtype
    bits = torch.int32 if ref.dtype == torch.float32 else torch.int16
    for r, n in enumerate(lens.view(-1).tolist()):
        assert torch.equal(out[r, :n].view(bits), ref[r, :n].view(bits)), r


def _assert_same_kept_scores(
    kept: torch.Tensor, scores: torch.Tensor, seq_lens: torch.Tensor
) -> None:
    """`kept` selects the score multiset per row of the stock top-k of
    `scores` (ties may pick other tokens)."""
    ref = torch.full_like(kept, -1)
    topk = kept.shape[1]
    get_indexer_topk("auto")(scores, seq_lens, 1, ref, topk, int(seq_lens.max()))

    def values(idx: torch.Tensor) -> torch.Tensor:
        v = scores.float().gather(1, idx.clamp(min=0).long())
        return torch.where(idx >= 0, v, -math.inf).sort(dim=1).values

    assert torch.equal((kept >= 0).sum(1), (ref >= 0).sum(1))
    assert torch.equal(values(kept), values(ref))


@pytest.mark.parametrize(
    "fp4,page_rows,topk,requests,rows_per_request",
    [
        pytest.param(False, 64, 2048, 4, 1, id="kimi_k3"),
        pytest.param(False, 64, 2048, 2, 4, id="kimi_k3_4_tokens"),
        pytest.param(True, 64, 512, 4, 1, id="deepseek_v4_c2"),
        pytest.param(True, 64, 512, 12, 4, id="deepseek_v4_c2_4_tokens"),
        pytest.param(True, 128, 512, 4, 1, id="deepseek_v4_c1"),
        pytest.param(True, 128, 512, 16, 5, id="deepseek_v4_c1_80_rows"),
    ],
)
def test_localized_decode_matches_single_launch(
    domains, fp4, page_rows, topk, requests, rows_per_request
):
    """The die-split scan and join return the single-launch scores bit for
    bit, row by row, also with several rows (query tokens) per request; the
    per-domain top-k (`cooperative_topk`, `persistent_topk` above 64 rows)
    and merge keep the stock top-k's scores, also deferred inside a CUDA
    graph replayed over new queries."""
    case = _DecodeCase(domains, fp4, page_rows, requests, rows_per_request)
    rows = case.seq_lens.shape[0]
    meta = case.localize(rows)
    assert meta is not None, "the localizer refused the batch"
    # A row count whose per-domain top-k needs a second round takes the stock
    # indexer.
    capacity = build.topk_cluster_capacity(domains)
    second_round = next(
        n for n in range(1, 256) if not build.use_die_split(n, capacity)
    )
    assert case.localize(second_round) is None
    ref = case.stock()
    for want_fp32 in (False, True):
        expected = as_fp32_logits(ref, case.seq_lens.view(-1)) if want_fp32 else ref
        _assert_same_scores(case.join(meta, want_fp32), expected, case.seq_lens)

    selector = get_indexer_topk("auto")
    max_seq_len = int(case.seq_lens.max())
    backend = scan.localized_topk_backend(selector, rows, case.max_model_len, topk)
    assert backend == ("cooperative" if rows <= 64 else "persistent")
    kept = torch.full((rows, topk), -7, dtype=torch.int32, device=ref.device)

    def localized_topk() -> None:
        scan.localized_paged_mqa_topk(
            meta,
            case.q,
            case.kv,
            case.weights,
            case.max_model_len,
            case.logits_dtype,
            case.indices,
            topk,
            max_seq_len,
            kept,
            topk_backend=backend,
        )

    localized_topk()
    _assert_same_kept_scores(kept, case.stock(), case.seq_lens)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        indexer_overlap.request_deferred_topk()
        localized_topk()
        stream, _ = indexer_overlap.take_deferred_topk()
        assert stream is not None, "the top-k was not deferred"
        torch.cuda.current_stream().wait_stream(stream)
    for _ in range(2):
        case.q_values.copy_(
            torch.randn_like(case.q_values, dtype=torch.float32).to(case.q_values.dtype)
            if not fp4
            else torch.randint_like(case.q_values, -128, 128)
        )
        kept.fill_(-7)
        graph.replay()
        _assert_same_kept_scores(kept, case.stock(), case.seq_lens)


@pytest.fixture
def candidate_stream(monkeypatch):
    """The overlap's candidate stream for one test; the overlap's module state
    is restored afterwards, also when the test fails."""
    monkeypatch.setenv("VLLM_INDEXER_OVERLAP", "1")
    monkeypatch.setattr(indexer_overlap, "_candidates", None)
    monkeypatch.setattr(indexer_overlap, "_expanded", None)
    indexer_overlap.init_candidate_stream(allowed=True)


@pytest.mark.parametrize("rows_per_request", [1, 3])
def test_candidate_source_matches_stock(domains, candidate_stream, rows_per_request):
    """The candidate source layer's per-domain top-k with deferred candidates
    (FULL CUDA graph decode) writes the stock path's candidate blocks and
    sparse block ids bit for bit, and keeps the stock top-k's scores, also
    with several rows per request."""
    from vllm.model_executor.kernels.attention.dsa.candidate_blocks import (
        select_candidate_blocks,
    )
    from vllm.model_executor.kernels.attention.dsa.sparse_mqa_logits import (
        candidate_blocks_to_sparse_indices,
    )
    from vllm.model_executor.layers.sparse_attn_indexer import (
        _localized_candidate_source_topk,
    )
    from vllm.v1.attention.backends.mla.indexer import (
        DeepSeekV32IndexerDecodeMetadata,
    )
    from vllm.v1.attention.backends.mla.sparse_indexer import (
        DeepseekV41SparseIndexerMetadata,
        SparseMQARowsMetadata,
    )

    topk, block_size, topk_blocks, sparse_block_kv = 512, 8, 2048, 8
    case = _DecodeCase(domains, True, 128, 4, rows_per_request)
    rows = case.seq_lens.shape[0]
    device = case.seq_lens.device
    loc = case.localize(rows)
    assert loc is not None

    def int32(*shape: int) -> torch.Tensor:
        return torch.full(shape, -3, dtype=torch.int32, device=device)

    row_ke = case.seq_lens.view(-1)
    zeros = torch.zeros(rows, dtype=torch.int32, device=device)
    candidates = int32(rows, topk_blocks)
    sparse = SparseMQARowsMetadata(
        row_ks=zeros,
        row_ke=row_ke,
        sparse_indices=int32(rows, topk_blocks),
        end=int32(rows),
        col_indices=int32(rows, topk),
        block_table=case.block_table,
        row_indices=case.indices,
    )
    metadata = DeepseekV41SparseIndexerMetadata(
        seq_lens=row_ke,
        max_seq_len=int(row_ke.max()),
        slot_mapping=zeros,
        num_decodes=rows // rows_per_request,
        num_decode_tokens=rows,
        num_prefills=0,
        num_prefill_tokens=0,
        prefill=None,
        decode=DeepSeekV32IndexerDecodeMetadata(
            block_table=case.block_table,
            seq_lens=case.seq_lens,
            decode_lens=torch.ones(rows, dtype=torch.int32, device=device),
            requires_padding=False,
            schedule_metadata=int32(1, 2),
            indices=case.indices,
            localized=loc,
        ),
        block_table=case.block_table,
        sparse_block_kv=sparse_block_kv,
        sparse_decode=sparse,
    )
    kept = int32(rows, topk)

    def localized() -> bool:
        return _localized_candidate_source_topk(
            metadata,
            case.q,
            case.kv,
            case.weights,
            case.seq_lens,
            case.max_model_len,
            case.logits_dtype,
            "auto",
            topk,
            kept,
            candidates,
            block_size,
        )

    # Outside a FULL CUDA graph capture the stock path runs.
    assert not localized()
    assert bool((kept == -3).all())
    # The stock path, eagerly, as in the warmup step before each capture.
    scores = case.stock()
    joined = case.join(loc, fp32=False)
    ref_candidates = int32(rows, topk_blocks)
    select_candidate_blocks(
        joined,
        None,
        row_ke,
        topk_blocks,
        block_size,
        ref_candidates,
        1,
        score_dtype=torch.float32,
    )
    ref_sparse, ref_end = candidate_blocks_to_sparse_indices(
        ref_candidates, zeros, row_ke, block_size, sparse_block_kv
    )

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        assert localized()
        indexer_overlap.join_candidates()
    graph.replay()
    assert torch.equal(candidates, ref_candidates)
    assert torch.equal(sparse.sparse_indices, ref_sparse)
    assert torch.equal(sparse.end, ref_end)
    _assert_same_kept_scores(kept, scores, case.seq_lens)
