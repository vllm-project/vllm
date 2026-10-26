# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The Kimi K3 KV-balanced MLA decode against an FP32 reference."""

import math
from types import SimpleNamespace

import pytest
import torch

from vllm.platforms import current_platform
from vllm.v1.attention.backends.registry import AttentionBackendEnum

LATENT, ROPE, PAGE = 512, 64, 64
SCALE = 1.0 / math.sqrt(192)

requires_sm100 = pytest.mark.skipif(
    not current_platform.is_cuda()
    or not current_platform.is_device_capability_family(100),
    reason="the KV-balanced MLA decode runs on SM100-family GPUs",
)


def test_kv_balanced_backend_is_registered_and_opt_in():
    """KIMI_K3_KV_BALANCED_MLA resolves to the TOKENSPEED_MLA backend with the
    KV-balanced decode, and the platforms never pick it on their own."""
    from vllm.models.kimi_k3.nvidia.kv_balanced_mla.backend import (
        KvBalancedMLABackend,
        KvBalancedMLAImpl,
    )
    from vllm.platforms.cuda import _get_backend_priorities
    from vllm.platforms.interface import DeviceCapability
    from vllm.v1.attention.backends.mla.tokenspeed_mla import (
        TokenspeedMLABackend,
        TokenspeedMLAImpl,
    )

    backend = AttentionBackendEnum.KIMI_K3_KV_BALANCED_MLA
    assert backend.get_class() is KvBalancedMLABackend
    assert KvBalancedMLABackend.get_name() == backend.name
    assert issubclass(KvBalancedMLABackend, TokenspeedMLABackend)
    assert issubclass(KvBalancedMLABackend.get_impl_cls(), TokenspeedMLAImpl)
    assert KvBalancedMLABackend.get_impl_cls() is KvBalancedMLAImpl
    for major in (9, 10, 12):
        assert backend not in _get_backend_priorities(
            use_mla=True, device_capability=DeviceCapability(major, 0)
        )


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
        attention_config=SimpleNamespace(tokenspeed_mla_min_split_kv=1),
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
        """The decode, or None for a step left to TokenSpeed's decode."""
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
def test_kv_balanced_mqa_plans_once_per_forward_and_falls_back(
    monkeypatch, workspace_init
):
    """The per-layer dispatch: the first layer of a forward plans and later layers
    reuse it; steps the kernel can't run are left to TokenSpeed's decode."""
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
    assert mqa(query.to(torch.bfloat16), metadata) is None
    many = SimpleNamespace(
        **{**vars(metadata), "num_decodes": 5, "num_decode_tokens": 5 * q}
    )
    assert mqa(_cat([query] * 3)[: 5 * q], many) is None
    longer = SimpleNamespace(
        **{**vars(metadata), "num_decode_tokens": requests * (q + 1)}
    )  # more query rows per request than the reserved buffers hold
    assert mqa(_cat([query, query[:requests]]), longer) is None
    # The heads of a wider (padded) query buffer: not packed.
    wide = torch.cat([query.view(torch.uint8)] * 2, dim=1).view(torch.float8_e4m3fn)
    assert mqa(wide[:, : query.shape[1]], metadata) is None


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
