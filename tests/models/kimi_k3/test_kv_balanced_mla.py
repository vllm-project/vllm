# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The Kimi K3 KV-balanced MLA decode against an FP32 reference."""

import math
from types import SimpleNamespace

import pytest
import torch

from vllm.platforms import current_platform
from vllm.v1.attention.backend import AttentionType
from vllm.v1.attention.backends.registry import AttentionBackendEnum

LATENT, ROPE, PAGE = 512, 64, 64
SCALE = 1.0 / math.sqrt(192)

requires_sm100 = pytest.mark.skipif(
    not current_platform.is_cuda()
    or not current_platform.is_device_capability_family(100),
    reason="the KV-balanced MLA decode runs on SM100-family GPUs",
)


def test_kv_balanced_backend_is_registered_and_opt_in():
    """KIMI_K3_KV_BALANCED_MLA resolves to the standalone KV-balanced backend, and
    the platforms never pick it on their own."""
    from vllm.models.kimi_k3.nvidia.kv_balanced_mla.backend import (
        KimiK3KvBalancedMLABackend,
        KimiK3KvBalancedMLAImpl,
        KimiK3KvBalancedMLAMetadataBuilder,
    )
    from vllm.platforms.cuda import _get_backend_priorities
    from vllm.platforms.interface import DeviceCapability

    backend = AttentionBackendEnum.KIMI_K3_KV_BALANCED_MLA
    assert backend.get_class() is KimiK3KvBalancedMLABackend
    assert KimiK3KvBalancedMLABackend.get_name() == backend.name
    assert KimiK3KvBalancedMLABackend.get_impl_cls() is KimiK3KvBalancedMLAImpl
    assert (
        KimiK3KvBalancedMLABackend.get_builder_cls()
        is KimiK3KvBalancedMLAMetadataBuilder
    )
    assert not KimiK3KvBalancedMLAImpl.lse_base_on_e
    for major in (9, 10, 12):
        assert backend not in _get_backend_priorities(
            use_mla=True, device_capability=DeviceCapability(major, 0)
        )


def test_kv_balanced_backend_rejects_unsupported(monkeypatch):
    """Selection rejects what the kernel can't run: a non-BF16 model, a non-FP8
    KV cache, a GPU outside the SM100 family or other MLA dimensions."""
    import vllm.config
    from vllm.models.kimi_k3.nvidia.kv_balanced_mla.backend import (
        KimiK3KvBalancedMLABackend,
    )
    from vllm.platforms.interface import DeviceCapability

    text_config = SimpleNamespace(
        qk_nope_head_dim=128, qk_rope_head_dim=64, v_head_dim=128, kv_lora_rank=512
    )
    config = SimpleNamespace(model_config=SimpleNamespace(hf_text_config=text_config))
    monkeypatch.setattr(vllm.config, "get_current_vllm_config", lambda: config)

    def reasons(dtype=torch.bfloat16, kv_cache_dtype="fp8", major=10):
        return KimiK3KvBalancedMLABackend.validate_configuration(
            head_size=576,
            dtype=dtype,
            kv_cache_dtype=kv_cache_dtype,
            block_size=64,
            use_mla=True,
            has_sink=False,
            use_sparse=False,
            use_mm_prefix=False,
            use_per_head_quant_scales=False,
            device_capability=DeviceCapability(major, 0),
            attn_type=AttentionType.DECODER,
        )

    assert reasons() == []
    assert "dtype not supported" in reasons(torch.float16)
    assert "kv_cache_dtype not supported" in reasons(kv_cache_dtype="auto")
    assert any("compute capability" in reason for reason in reasons(major=9))
    text_config.kv_lora_rank = 256
    assert any("MLA dimensions" in reason for reason in reasons())


def test_kv_balanced_impl_runs_the_kernel():
    """forward_mqa is the KV-balanced decode."""
    from vllm.models.kimi_k3.nvidia.kv_balanced_mla.backend import (
        KimiK3KvBalancedMLAImpl,
    )

    impl = object.__new__(KimiK3KvBalancedMLAImpl)
    calls = []

    def kv_balanced(*args):
        calls.append(args)
        return "kv_balanced", None

    impl.kv_balanced = kv_balanced
    assert impl.forward_mqa("q", "cache", "metadata", "layer") == ("kv_balanced", None)
    assert calls == [(impl, "q", "cache", "metadata", "layer")]


def _paged(rows_per_request, device):
    """A paged FP8 cache holding ``rows_per_request`` [n, 576] FP32 rows per request
    at shuffled pages, and its block table."""
    pages = [-(-len(rows) // PAGE) for rows in rows_per_request]
    cache = torch.zeros(sum(pages) + 2, PAGE, LATENT + ROPE, device=device)
    order = torch.randperm(cache.shape[0], device=device).to(torch.int32)
    block_table = torch.zeros(
        len(pages), max(max(pages), 1), dtype=torch.int32, device=device
    )
    start = 0
    for b, (rows, count) in enumerate(zip(rows_per_request, pages)):
        block_table[b, :count] = order[start : start + count]
        flat = cache[order[start : start + count].long()].reshape(-1, LATENT + ROPE)
        flat[: len(rows)] = rows
        cache[order[start : start + count].long()] = flat.view(-1, PAGE, LATENT + ROPE)
        start += count
    return cache.to(torch.float8_e4m3fn), block_table


def _shard_len(n, cp):
    """Tokens of a length-``n`` request on rank ``rank`` of ``cp`` = (world, rank,
    interleave): rank r holds the interleave-sized chunks r, r + world, ..."""
    world, rank, interleave = cp
    chunks, tail = divmod(n, interleave * world)
    return chunks * interleave + min(max(tail - rank * interleave, 0), interleave)


def _reference(query, cache, block_table, local, lengths, causal, cp=(1, 0, 1)):
    """FP32 latent output and log2 LSE of each query row; an empty request gives a
    zero output and an LSE of -inf."""
    world, rank, interleave = cp
    q, heads = query.shape[1], query.shape[2]
    outs, lses = [], []
    for b, n in enumerate(local.tolist()):
        if n == 0:
            outs.append(query.new_zeros((q, heads, LATENT), dtype=torch.float32))
            lses.append(torch.full((q, heads), -math.inf, device=query.device))
            continue
        pages = block_table[b, : -(-n // PAGE)].long()
        kv = cache[pages].reshape(-1, LATENT + ROPE)[:n].float()
        scores = torch.einsum("qhd,kd->qhk", query[b].float(), kv) * SCALE
        if causal:
            # Row i of a request sees global positions up to length - q + i.
            visible = int(lengths[b]) - q + torch.arange(q, device=kv.device)
            local_pos = torch.arange(n, device=kv.device)
            position = (
                local_pos // interleave * interleave * world
                + rank * interleave
                + local_pos % interleave
            )
            scores = scores.masked_fill(
                position[None, None, :] > visible[:, None, None], float("-inf")
            )
        outs.append(torch.einsum("qhk,kd->qhd", scores.softmax(-1), kv[:, :LATENT]))
        lses.append(torch.logsumexp(scores, -1) / math.log(2))
    return torch.stack(outs), torch.stack(lses)


def _run(query, cache, block_table, local, totals, causal, cp=(1, 0, 1)):
    from vllm.models.kimi_k3.nvidia.kv_balanced_mla.core import KvBalancedMla

    requests, q, heads, _ = query.shape
    device = query.device
    core = KvBalancedMla(heads, q, max_requests=8, causal=causal, device=device)
    counts = torch.zeros(core.max_requests, dtype=torch.int32, device=device)
    work = torch.zeros(core.list_len, dtype=torch.int32, device=device)
    # A shared workspace larger than the kernel's Int32 sizes allow.
    workspace = torch.empty(2**31 + 1024, dtype=torch.int8, device=device)
    core.plan(local, counts, work)
    lse = torch.empty(requests, q, heads, dtype=torch.float32, device=device)
    out = core.decode(
        query.view(-1, heads, LATENT + ROPE),
        cache,
        block_table,
        local,
        SCALE,
        1.0,
        counts,
        work,
        workspace,
        lse=lse,
        causal_lens=totals if cp[0] > 1 else None,
        cp=cp,
    )
    # Every nonempty request has 1 .. 8 splits, and the list ends in empty entries.
    planned = counts[:requests]
    assert bool(((planned >= 1) & (planned <= 8)).all())
    entries = int((work >= 0).sum())
    assert entries == int(planned.sum()) * core.q_tiles
    assert bool((work[entries:] == -1).all())
    return out.view(requests, q, heads, LATENT).float(), lse


def _cat(tensors):
    """torch.cat of FP8 tensors (through their bytes)."""
    return torch.cat([t.view(torch.uint8) for t in tensors]).view(torch.float8_e4m3fn)


def _check(out, lse, ref_out, ref_lse, tolerance=0.03):
    assert (out - ref_out).norm() / ref_out.norm() < tolerance
    torch.testing.assert_close(lse, ref_lse, atol=0.05, rtol=0)


@requires_sm100
@pytest.mark.parametrize(
    ("lengths", "heads", "q", "causal", "cp"),
    [
        ([130, 4097, 30000], 96, 5, True, (1, 0, 1)),  # DEP verify, unequal lengths
        ([64, 20000, 0, 900, 9000, 70000, 7], 96, 5, True, (1, 0, 1)),  # empty row
        ([5000, 16000], 64, 4, False, (1, 0, 1)),  # DSpark draft block
        ([2000, 64000, 12345], 96, 5, True, (8, 0, 1)),  # DCP8 shard of rank 0
        # Rank 5 of 8, interleave 64: the first three requests end inside one of its
        # chunks, so the causal mask cuts its last keys.
        ([5443, 51580, 330, 2000], 96, 5, True, (8, 5, 64)),
    ],
)
def test_kv_balanced_decode_matches_reference(lengths, heads, q, causal, cp):
    torch.manual_seed(0)
    device = torch.device("cuda")
    local = [_shard_len(n, cp) for n in lengths]
    cache, block_table = _paged(
        [torch.randn(n, LATENT + ROPE, device=device) / 2 for n in local], device
    )
    query = (torch.randn(len(lengths), q, heads, LATENT + ROPE, device=device) / 2).to(
        torch.float8_e4m3fn
    )
    as_tensor = lambda x: torch.tensor(x, dtype=torch.int32, device=device)  # noqa: E731
    local_t, totals = as_tensor(local), as_tensor(lengths)
    out, lse = _run(query, cache, block_table, local_t, totals, causal, cp)
    ref_out, ref_lse = _reference(
        query, cache, block_table, local_t, totals, causal, cp
    )
    _check(out, lse, ref_out, ref_lse)


@requires_sm100
def test_kv_balanced_decode_keeps_small_probabilities():
    """A peaked softmax: one key per 128 scores 2^10 above a long tail, so each tail
    probability is 2^-10 of the row max (below E4M3's 2^-9 without the P prescale),
    and the tail's values point away from the peaks' so losing it moves the output
    by about a fifth."""
    torch.manual_seed(0)
    device = torch.device("cuda")
    heads, q, lengths = 96, 5, [40000, 90000]
    rows = []
    for n in lengths:
        kv = torch.zeros(n, LATENT + ROPE, device=device)
        kv[:, :LATENT] = 0.5 + torch.randn(n, LATENT, device=device) / 32
        kv[::128, :LATENT] = -0.5
        # Only the RoPE part of the key meets the query: the peak keys score
        # 1.5 * 64 * SCALE / ln(2) = 10 log2 units above the rest.
        kv[::128, LATENT:] = 1.5
        rows.append(kv)
    cache, block_table = _paged(rows, device)
    query = torch.zeros(len(lengths), q, heads, LATENT + ROPE, device=device)
    query[..., LATENT:] = 1.0
    query = query.to(torch.float8_e4m3fn)
    local = torch.tensor(lengths, dtype=torch.int32, device=device)
    out, lse = _run(query, cache, block_table, local, local, False)
    ref_out, ref_lse = _reference(query, cache, block_table, local, local, False)
    _check(out, lse, ref_out, ref_lse)


@requires_sm100
def test_kv_balanced_plan_replays_in_cuda_graph():
    """A captured plan and decode follow KV lengths changed in place."""
    from vllm.models.kimi_k3.nvidia.kv_balanced_mla.core import KvBalancedMla

    torch.manual_seed(0)
    device = torch.device("cuda")
    heads, q, capacity = 96, 5, [40000, 9000, 70000]
    cache, block_table = _paged(
        [torch.randn(n, LATENT + ROPE, device=device) / 2 for n in capacity], device
    )
    query = (torch.randn(len(capacity), q, heads, LATENT + ROPE, device=device) / 2).to(
        torch.float8_e4m3fn
    )
    core = KvBalancedMla(heads, q, max_requests=8, causal=True, device=device)
    counts = torch.zeros(core.max_requests, dtype=torch.int32, device=device)
    work = torch.zeros(core.list_len, dtype=torch.int32, device=device)
    workspace = torch.empty(core.workspace_bytes, dtype=torch.int8, device=device)
    lens = torch.tensor([100, 9000, 1500], dtype=torch.int32, device=device)

    def step():
        core.plan(lens, counts, work)
        return core.decode(
            query.view(-1, heads, LATENT + ROPE),
            cache,
            block_table,
            lens,
            SCALE,
            1.0,
            counts,
            work,
            workspace,
        )

    step()  # compile outside the capture
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = step()
    for lengths in ([100, 9000, 1500], [40000, 64, 70000], [0, 2048, 33333]):
        lens.copy_(torch.tensor(lengths, dtype=torch.int32))
        graph.replay()
        ref_out, _ = _reference(query, cache, block_table, lens, lens, True)
        got = out.view(len(capacity), q, heads, LATENT).float()
        assert (got - ref_out).norm() / ref_out.norm() < 0.03, lengths


def _layer(monkeypatch, lengths, cp, lse):
    """The KV-balanced decode of an MLA layer over random shards of ``lengths`` on
    rank ``cp[1]``."""
    from vllm.models.kimi_k3.nvidia.kv_balanced_mla import decode

    device = torch.device("cuda")
    heads, q = 96, 5
    world, rank, interleave = cp
    config = SimpleNamespace(
        parallel_config=SimpleNamespace(
            decode_context_parallel_size=world, cp_kv_cache_interleave_size=interleave
        ),
        scheduler_config=SimpleNamespace(max_num_seqs=4),
    )
    monkeypatch.setattr(decode, "get_current_vllm_config", lambda: config)
    monkeypatch.setattr(decode, "max_decode_query_len", lambda _: q)
    monkeypatch.setattr(decode, "_SHAPES", {})
    monkeypatch.setattr(decode, "_PLANNED", {})
    local = [_shard_len(n, cp) for n in lengths]
    cache, block_table = _paged(
        [torch.randn(n, LATENT + ROPE, device=device) / 2 for n in local], device
    )
    as_tensor = lambda x: torch.tensor(x, dtype=torch.int32, device=device)  # noqa: E731
    impl = SimpleNamespace(
        scale=SCALE,
        dcp_world_size=world,
        dcp_rank=rank,
        need_to_return_lse_for_decode=lse,
    )
    attn_layer = SimpleNamespace(_q_scale_float=1.0, _k_scale_float=1.0)
    metadata = SimpleNamespace(
        decode=SimpleNamespace(
            seq_lens=as_tensor(local),
            block_table=block_table,
            dcp_tot_seq_lens=as_tensor(lengths) if world > 1 else None,
        ),
        num_decodes=len(lengths),
        num_decode_tokens=len(lengths) * q,
        causal=True,
    )
    query = (torch.randn(len(lengths) * q, heads, LATENT + ROPE, device=device) / 2).to(
        torch.float8_e4m3fn
    )
    kv_balanced = decode.KvBalancedDecode(heads // world)

    def mqa(query, metadata):
        return kv_balanced(impl, query, cache, metadata, attn_layer)

    def reference():
        decode = metadata.decode
        totals = decode.seq_lens if world == 1 else decode.dcp_tot_seq_lens
        return _reference(
            query.view(len(lengths), q, heads, -1),
            cache,
            block_table,
            decode.seq_lens,
            totals,
            True,
            cp,
        )

    return mqa, metadata, query, reference


@requires_sm100
def test_kv_balanced_mqa_plans_once_per_forward_and_rejects_unsupported(
    monkeypatch, workspace_init
):
    """The per-layer dispatch: the first layer of a forward plans and later layers
    reuse it; a step the kernel can't run raises."""
    from vllm.models.kimi_k3.nvidia.kv_balanced_mla import core, decode

    torch.manual_seed(0)
    q = 5
    mqa, metadata, query, reference = _layer(
        monkeypatch, [3000, 21000], (1, 0, 1), lse=False
    )
    context = [object()]
    monkeypatch.setattr(decode, "get_forward_context", lambda: context[0])
    plans = []
    plan = core.KvBalancedMla.plan

    def counted_plan(self, *args):
        plans.append(1)
        plan(self, *args)

    monkeypatch.setattr(core.KvBalancedMla, "plan", counted_plan)

    out, lse = mqa(query, metadata)
    assert lse is None
    ref_out, _ = reference()
    assert (out.view_as(ref_out).float() - ref_out).norm() / ref_out.norm() < 0.03
    mqa(query, metadata)  # a later layer of the same forward
    assert len(plans) == 1
    context[0] = object()  # the next forward
    mqa(query, metadata)
    assert len(plans) == 2

    requests = metadata.num_decodes
    with pytest.raises(ValueError, match="FP8 query"):
        mqa(query.to(torch.bfloat16), metadata)
    many = SimpleNamespace(
        **{**vars(metadata), "num_decodes": 5, "num_decode_tokens": 5 * q}
    )
    with pytest.raises(ValueError, match="5 requests"):
        mqa(_cat([query] * 3)[: 5 * q], many)
    longer = SimpleNamespace(
        **{**vars(metadata), "num_decode_tokens": requests * (q + 1)}
    )  # more query rows per request than the reserved buffers hold
    with pytest.raises(ValueError, match="at most 5 each"):
        mqa(_cat([query, query[:requests]]), longer)
    # The heads of a wider (padded) query buffer: not packed.
    wide = torch.cat([query.view(torch.uint8)] * 2, dim=1).view(torch.float8_e4m3fn)
    with pytest.raises(ValueError, match="contiguous"):
        mqa(wide[:, : query.shape[1]], metadata)


@requires_sm100
def test_kv_balanced_mqa_dcp_shard(monkeypatch, workspace_init):
    """Under DCP the decode passes this rank's shard, the global lengths and the
    interleave to the kernel and returns the log2 LSE for the DCP combine."""
    from vllm.models.kimi_k3.nvidia.kv_balanced_mla import decode

    torch.manual_seed(0)
    monkeypatch.setattr(decode, "get_forward_context", lambda: object())
    mqa, metadata, query, reference = _layer(
        monkeypatch, [3000, 21000, 700], (4, 2, 64), lse=True
    )
    out, lse = mqa(query, metadata)
    ref_out, ref_lse = reference()
    _check(out.view_as(ref_out).float(), lse.view_as(ref_lse), ref_out, ref_lse)


def test_specialized_mla_core_policy():
    """The specialized decode's MLA core per (rows, q): balanced for spec verify
    (q > 1) and from BALANCED_Q1_BATCH rows at q = 1 where the kernel runs,
    tokenspeed otherwise (both tokenspeed's kernel, the stock backend's; there is no
    trtllm-gen core); a named balanced core falls back where it can't run."""
    pytest.importorskip("cutlass")
    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.mla.prepared import (
        BALANCED_Q1_BATCH,
        core_for,
    )

    for dcp in (False, True):
        for rows in (5, 60):
            assert core_for(rows, q=5, dcp=dcp, balanced=True) == "balanced"
        assert core_for(60, q=5, dcp=dcp, balanced=False) == "tokenspeed"
        for rows in (BALANCED_Q1_BATCH, 64):
            assert core_for(rows, q=1, dcp=dcp, balanced=True) == "balanced"
        below = BALANCED_Q1_BATCH - 1
        assert core_for(below, q=1, dcp=dcp, balanced=True) == "tokenspeed"
        assert core_for(60, "tokenspeed", 5, dcp, True) == "tokenspeed"
        assert core_for(5, "balanced", 5, dcp, True) == "balanced"
        assert core_for(5, "balanced", 5, dcp, False) == "tokenspeed"
    # SM100 TP8 (12 heads: M64 tiles, no balanced core): tokenspeed at every batch.
    for rows in (1, 16, 64):
        for dcp in (False, True):
            assert core_for(rows, q=1, dcp=dcp, balanced=False) == "tokenspeed"
    for core in ("trtllm", "flash"):
        with pytest.raises(ValueError, match="expected auto"):
            core_for(8, core)


@requires_sm100
@pytest.mark.parametrize(
    ("lengths", "heads", "q", "cp"),
    [
        # TP8 (12 heads, whole KV): one long request needs more than 8 splits.
        ([130000], 12, 5, (1, 0, 1)),
        # TP8, AgentX-like mixed lengths with an empty padded request.
        ([300, 150000, 7000, 64000, 0, 2500, 90000, 41], 12, 5, (1, 0, 1)),
        ([1, 6000, 33000, 100000], 12, 1, (1, 0, 1)),
        # q = 1 past 12 requests: the 8-split core.
        ([i * 7919 % 60000 + 1 for i in range(16)], 12, 1, (1, 0, 1)),
        # DCP8 (96 gathered heads, a shard): rank 5 of 8, interleave 64 and 1.
        ([5443, 51580, 330, 2000, 160000], 96, 5, (8, 5, 64)),
        # The 7-token request leaves rank 3 one key, past the first rows' bounds.
        ([2000, 64000, 12345, 7], 96, 5, (8, 3, 1)),
        ([700, 300000], 96, 1, (8, 0, 1)),
    ],
)
def test_specialized_balanced_core_matches_reference(lengths, heads, q, cp):
    """The specialized decode's balanced MLA core (``BalancedMla``: TP8 / DCP8
    capacity, PDL launch, caller-owned output) against the FP32 reference, captured
    in a CUDA graph with the plan, as the decode step runs it."""
    from vllm.models.kimi_k3.nvidia.kv_balanced_mla.core import supports
    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.mla.balanced import (
        CAPACITY,
        BalancedMla,
    )

    torch.manual_seed(0)
    device = torch.device("cuda")
    if not supports(heads, q, device):
        pytest.skip("this GPU runs the shape on M64 tiles, without the balanced core")
    dcp = cp[0] > 1
    local = [_shard_len(n, cp) for n in lengths]
    cache, block_table = _paged(
        [torch.randn(n, LATENT + ROPE, device=device) / 2 for n in local], device
    )
    query = (torch.randn(len(lengths), q, heads, LATENT + ROPE, device=device) / 2).to(
        torch.float8_e4m3fn
    )
    as_tensor = lambda x: torch.tensor(x, dtype=torch.int32, device=device)  # noqa: E731
    local_t, totals = as_tensor(local), as_tensor(lengths)
    balanced = BalancedMla(heads, device, dcp)
    balanced.require(q, len(lengths))
    out = torch.full((len(lengths), q, heads, LATENT), float("nan"), device=device)
    out = out.to(torch.bfloat16)
    lse = torch.empty(len(lengths), q, heads, dtype=torch.float32, device=device)
    extra = dict(cp=cp, totals=totals) if dcp else {}

    def step():
        balanced.plan(q, local_t)
        balanced.decode(
            query, cache, block_table, local_t, (SCALE, 1.0), out, lse=lse, **extra
        )

    step()  # compile outside the capture
    out.fill_(float("nan"))
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        step()
    graph.replay()
    key = balanced.key(q, len(lengths))
    counts = balanced.plans[key][0][: len(lengths)]
    capacity = key[1]
    assert capacity in [c for _, c in CAPACITY[dcp]]
    assert bool((counts <= capacity).all()) and int(counts.max()) >= 1
    if not dcp and max(lengths) >= 100000:  # beyond the kernel's default of 8 splits
        assert int(counts.max()) > 8
    ref_out, ref_lse = _reference(query, cache, block_table, local_t, totals, True, cp)
    # A row with no visible local key (DCP: a short request's shard) carries no
    # weight in DcpValue's merge: a finite output under an LSE far below any real one
    # (as tokenspeed returns it).
    masked = ref_out.isnan().any(-1)
    assert bool(out[masked].float().isfinite().all())
    assert bool((lse[masked] < -1e4).all())
    live = ~masked
    _check(out.float()[live], lse[live], ref_out[live], ref_lse[live])


def _stock_split(heads, q, requests, max_seq, device):
    """tokenspeed_mla_decode's split count (shared by every request of the batch) for
    ``requests`` x ``q`` query rows of ``heads`` heads and ``max_seq``."""
    from tokenspeed_mla import mla_decode as md

    fp8 = torch.float8_e4m3fn
    arch = torch.cuda.get_device_capability(device)
    tiler, _ = md.select_mla_decode_tilers(
        heads, q, is_fp8=True, compute_capability=arch
    )
    fold = md.get_mla_decode_fold_sq_factor(heads, q, tiler[0])
    rows = (q // fold, heads * fold)
    if tiler[0] == 128 and -(-heads * q // 128) < rows[0]:  # packed M128 tiles
        rows = (md.compute_q_tile_layout(heads, q, 128)[1], 128)
    sms = md.get_num_sm(device)
    split, _ = md._get_split_kv_and_workspace_size(
        requests, *rows, LATENT, sms, max_seq, fp8, tiler, 1
    )
    return split


@requires_sm100
@pytest.mark.parametrize(
    ("lengths", "heads", "q", "cp"),
    [
        # TP8: 12 heads, one query tile per request (R1: 2 D bands per reduced row).
        ([30000], 12, 5, (1, 0, 1)),
        ([9000, 9000], 12, 5, (1, 0, 1)),
        ([3000, 20000, 777, 12345], 12, 5, (1, 0, 1)),
        ([40000] * 3 + [5000] * 9, 12, 1, (1, 0, 1)),
        # DCP8: 96 gathered heads over rank 3's shard (4 packed tiles at q = 5).
        ([30000, 6000, 61000], 96, 5, (8, 3, 1)),
        ([16000, 70000, 40000], 96, 5, (8, 6, 64)),
        ([90000, 50000] * 5, 96, 1, (8, 0, 1)),
    ],
)
def test_specialized_balanced_core_bitwise_matches_tokenspeed(lengths, heads, q, cp):
    """At tokenspeed_mla_decode's split points, the specialized decode's balanced core
    (BalancedMla, as served) returns stock's BF16 latent and log2 LSE bit for bit:
    the same FP32 split-KV partials, P scaling and reduction. Each request is given
    stock's split count in the plan (the balanced core only moves the split points,
    a summation order); stock's own call is the TOKENSPEED_MLA backend's."""
    from tokenspeed_mla import tokenspeed_mla_decode

    from vllm.models.kimi_k3.nvidia.kv_balanced_mla.core import KV_TILE, supports
    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.mla.balanced import (
        BalancedMla,
    )

    torch.manual_seed(0)
    device = torch.device("cuda")
    if not supports(heads, q, device):
        pytest.skip("this GPU runs the shape on M64 tiles, without the balanced core")
    world, rank, interleave = cp
    dcp = world > 1
    local = [_shard_len(n, cp) for n in lengths]
    cache, block_table = _paged(
        [torch.randn(n, LATENT + ROPE, device=device) / 2 for n in local], device
    )
    requests = len(lengths)
    query = (torch.randn(requests, q, heads, LATENT + ROPE, device=device) / 2).to(
        torch.float8_e4m3fn
    )
    as_tensor = lambda x: torch.tensor(x, dtype=torch.int32, device=device)  # noqa: E731
    local_t, totals = as_tensor(local), as_tensor(lengths)
    softmax, output_scale = SCALE * 0.75, 1.25

    # Stock: the TOKENSPEED_MLA backend's call (one split count for the batch).
    max_seq = max(local)
    split = _stock_split(heads, q, requests, max_seq, device)
    workspace = torch.zeros(1 << 30, dtype=torch.int8, device=device)
    stock_out, stock_lse = tokenspeed_mla_decode(
        query=query,
        kv_cache=cache,
        workspace_buffer=workspace,
        kv_lora_rank=LATENT,
        qk_rope_head_dim=ROPE,
        block_tables=block_table,
        seq_lens=local_t,
        max_seq_len=max_seq,
        softmax_scale=softmax,
        output_scale=output_scale,
        return_lse=True,
        causal_mask=True,
        causal_seqs=totals if dcp else None,
        cp_world=world,
        cp_rank=rank,
        cp_interleave_size=interleave,
    )

    balanced = BalancedMla(heads, device, dcp)
    balanced.require(q, requests)
    key = balanced.key(q, requests)
    core, (counts, work) = balanced.cores[key], balanced.plans[key]
    # Each request's nonempty splits at stock's chunk, ceil(tiles / split) tiles.
    tiles = [-(-n // KV_TILE) for n in local]
    planned = [-(-t // -(-t // split)) for t in tiles]
    assert min(planned) > 1, "a single-split request stores its output directly"
    assert max(planned) <= core.capacity, (planned, core.capacity)
    entries = [
        ((tile * requests + b) << 8) | s
        for b, n in enumerate(planned)
        for s in range(n)
        for tile in range(core.q_tiles)
    ]
    counts[:requests] = as_tensor(planned)
    work.fill_(-1)
    work[: len(entries)] = as_tensor(entries)
    out = torch.empty(requests, q, heads, LATENT, dtype=torch.bfloat16, device=device)
    lse = torch.empty(requests, q, heads, dtype=torch.float32, device=device)
    extra = dict(cp=cp, totals=totals) if dcp else {}
    balanced.decode(
        query, cache, block_table, local_t, (softmax, output_scale), out, lse, **extra
    )
    torch.accelerator.synchronize()
    assert torch.equal(out.view(torch.int16), stock_out.view(torch.int16))
    assert torch.equal(lse.view(torch.int32), stock_lse.view(torch.int32))
