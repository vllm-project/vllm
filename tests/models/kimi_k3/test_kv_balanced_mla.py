# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The Kimi K3 KV-balanced MLA decode against an FP32 reference."""

import math
from types import SimpleNamespace

import pytest
import torch

from vllm.models.kimi_k3.nvidia.kv_balanced_mla.layer import (
    AUTO,
    KV_BALANCED,
    mla_decode_backend,
)
from vllm.platforms import current_platform

LATENT, ROPE, PAGE = 512, 64, 64
SCALE = 1.0 / math.sqrt(192)

requires_sm100 = pytest.mark.skipif(
    not current_platform.is_cuda()
    or not current_platform.is_device_capability_family(100),
    reason="the KV-balanced MLA decode runs on SM100-family GPUs",
)


@pytest.mark.parametrize(
    ("extra", "expected"),
    [(None, AUTO), ({}, AUTO), ({"mla_decode_backend": "kv_balanced"}, KV_BALANCED)],
)
def test_mla_decode_backend(extra, expected):
    assert mla_decode_backend(SimpleNamespace(additional_config=extra)) == expected


def test_mla_decode_backend_rejects_unknown():
    with pytest.raises(ValueError, match="mla_decode_backend"):
        mla_decode_backend(
            SimpleNamespace(additional_config={"mla_decode_backend": "flash"})
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


def _reference(query, cache, block_table, local, lengths, causal, dcp):
    """FP32 latent output and log2 LSE of each query row; an empty request gives a
    zero output and an LSE of -inf."""
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
            position = torch.arange(n, device=kv.device) * dcp
            scores = scores.masked_fill(
                position[None, None, :] > visible[:, None, None], float("-inf")
            )
        outs.append(torch.einsum("qhk,kd->qhd", scores.softmax(-1), kv[:, :LATENT]))
        lses.append(torch.logsumexp(scores, -1) / math.log(2))
    return torch.stack(outs), torch.stack(lses)


def _run(query, cache, block_table, local, totals, causal, dcp):
    from vllm.models.kimi_k3.nvidia.kv_balanced_mla.core import KvBalancedMla

    requests, q, heads, _ = query.shape
    device = query.device
    core = KvBalancedMla(heads, q, max_requests=8, causal=causal, device=device)
    counts = torch.zeros(core.max_requests, dtype=torch.int32, device=device)
    work = torch.zeros(core.list_len, dtype=torch.int32, device=device)
    workspace = torch.empty(core.workspace_bytes, dtype=torch.int8, device=device)
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
        causal_lens=totals if dcp > 1 else None,
        cp=(dcp, 0, 1),
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
    ("lengths", "heads", "q", "causal", "dcp"),
    [
        ([130, 4097, 30000], 96, 5, True, 1),  # DEP verify, unequal lengths
        ([64, 20000, 0, 900, 9000, 70000, 7], 96, 5, True, 1),  # with an empty row
        ([5000, 16000], 64, 4, False, 1),  # DSpark draft block
        ([2000, 64000, 12345], 96, 5, True, 8),  # DCP8 shard
    ],
)
def test_kv_balanced_decode_matches_reference(lengths, heads, q, causal, dcp):
    torch.manual_seed(0)
    device = torch.device("cuda")
    local = [-(-n // dcp) for n in lengths]
    cache, block_table = _paged(
        [torch.randn(n, LATENT + ROPE, device=device) / 2 for n in local], device
    )
    query = (torch.randn(len(lengths), q, heads, LATENT + ROPE, device=device) / 2).to(
        torch.float8_e4m3fn
    )
    as_tensor = lambda x: torch.tensor(x, dtype=torch.int32, device=device)  # noqa: E731
    local_t, totals = as_tensor(local), as_tensor(lengths)
    out, lse = _run(query, cache, block_table, local_t, totals, causal, dcp)
    ref_out, ref_lse = _reference(
        query, cache, block_table, local_t, totals, causal, dcp
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
    out, lse = _run(query, cache, block_table, local, local, False, 1)
    ref_out, ref_lse = _reference(query, cache, block_table, local, local, False, 1)
    _check(out, lse, ref_out, ref_lse)


@requires_sm100
def test_kv_balanced_mqa_plans_once_per_forward_and_falls_back(
    monkeypatch, workspace_init
):
    """The per-layer dispatch: the first layer of a forward plans and later layers
    reuse it; steps the kernel can't run use the backend decode."""
    from vllm.models.kimi_k3.nvidia.kv_balanced_mla import core, layer

    torch.manual_seed(0)
    device = torch.device("cuda")
    heads, q, lengths = 96, 5, [3000, 21000]
    config = SimpleNamespace(
        parallel_config=SimpleNamespace(
            decode_context_parallel_size=1, cp_kv_cache_interleave_size=1
        ),
        scheduler_config=SimpleNamespace(max_num_seqs=4),
    )
    monkeypatch.setattr(layer, "get_current_vllm_config", lambda: config)
    monkeypatch.setattr(layer, "max_decode_query_len", lambda _: q)
    context = [object()]
    monkeypatch.setattr(layer, "get_forward_context", lambda: context[0])
    plans = []
    plan = core.KvBalancedMla.plan
    monkeypatch.setattr(
        core.KvBalancedMla,
        "plan",
        lambda self, *args: plans.append(1) or plan(self, *args),
    )

    cache, block_table = _paged(
        [torch.randn(n, LATENT + ROPE, device=device) / 2 for n in lengths], device
    )
    lens = torch.tensor(lengths, dtype=torch.int32, device=device)
    impl = SimpleNamespace(
        scale=SCALE, dcp_world_size=1, dcp_rank=0, need_to_return_lse_for_decode=False
    )
    mla = SimpleNamespace(
        num_local_heads=heads,
        impl=impl,
        _q_scale_float=1.0,
        _k_scale_float=1.0,
        _attn_read_kv_cache=lambda: cache,
    )
    # The backend decode that the kernel defers to.
    impl.forward_mqa = lambda *args: "backend"
    metadata = SimpleNamespace(
        decode=SimpleNamespace(
            seq_lens=lens, block_table=block_table, dcp_tot_seq_lens=None
        ),
        num_decodes=len(lengths),
        num_decode_tokens=len(lengths) * q,
        causal=True,
    )
    query = (torch.randn(len(lengths) * q, heads, LATENT + ROPE, device=device) / 2).to(
        torch.float8_e4m3fn
    )
    mqa = layer.TokenspeedKvBalancedMqa(mla)

    out, lse = mqa(mla, query, metadata)
    assert lse is None
    ref_out, _ = _reference(
        query.view(len(lengths), q, heads, -1), cache, block_table, lens, lens, True, 1
    )
    assert (out.view_as(ref_out).float() - ref_out).norm() / ref_out.norm() < 0.03
    mqa(mla, query, metadata)  # a later layer of the same forward
    assert len(plans) == 1
    context[0] = object()  # the next forward
    mqa(mla, query, metadata)
    assert len(plans) == 2

    assert mqa(mla, query.to(torch.bfloat16), metadata) == "backend"
    many = SimpleNamespace(
        **{**vars(metadata), "num_decodes": 5, "num_decode_tokens": 25}
    )
    assert mqa(mla, _cat([query] * 3)[:25], many) == "backend"
    longer = SimpleNamespace(
        **{**vars(metadata), "num_decode_tokens": len(lengths) * (q + 1)}
    )  # more query rows per request than the reserved buffers hold
    assert mqa(mla, _cat([query, query[: len(lengths)]]), longer) == "backend"
