# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DSpark draft dense MLA decode with the block's query tokens folded into heads
(specialized_decode/drafter/attention.py) against the stock one-row-per-token
FlashInfer call and an exact FP32 reference over the dequantized FP8 inputs."""

import pytest
import torch

from vllm.platforms import current_platform

if not current_platform.is_device_capability_family(100):
    pytest.skip("trtllm-gen MLA decode needs SM100", allow_module_level=True)

KV_LORA, ROPE, NOPE = 512, 64, 128
D = KV_LORA + ROPE
PAGE = 64
SCALE = 192**-0.5


def _case(R, ctxs, heads, k, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    pages = [-(-c // PAGE) for c in ctxs]
    width = -(-max(pages) // 2) * 2  # trtllm-gen: block_num % (128 / page) == 0
    num_pages = sum(pages) + 1
    kv = (torch.randn(num_pages, PAGE, D, device="cuda", generator=g) * 0.5).to(
        torch.float8_e4m3fn
    )
    perm = torch.randperm(num_pages, device="cuda", generator=g)
    bt = torch.zeros(R, width, dtype=torch.int32, device="cuda")
    off = 0
    for r, n in enumerate(pages):
        bt[r, :n] = perm[off : off + n]
        off += n
    seq = torch.tensor(ctxs, dtype=torch.int32, device="cuda")
    q = (torch.randn(R * k, heads, D, device="cuda", generator=g) * 0.5).to(
        torch.float8_e4m3fn
    )
    return q, kv, bt, seq


def _reference(q, kv, bt, seq, k):
    out = []
    for r in range(bt.shape[0]):
        n = int(seq[r])
        rows = kv[bt[r]].reshape(-1, D)[:n].float()  # [n, D]
        qr = q[r * k : (r + 1) * k].float()  # [k, H, D]
        s = torch.einsum("thd,nd->thn", qr, rows) * SCALE
        p = s.softmax(-1)
        out.append(torch.einsum("thn,nl->thl", p, rows[:, :KV_LORA]))
    return torch.cat(out)


def _stock(q, kv, bt, seq, k, ws, dcp=None):
    from flashinfer.decode import trtllm_batch_decode_with_kv_cache_mla

    from vllm.v1.attention.backends.mla.flashinfer_mla import (
        _select_mla_decode_backend,
    )

    row_req = torch.arange(q.shape[0], device="cuda") // k
    extra = {}
    if dcp is not None:
        extra = dict(
            enable_dcp=True,
            cp_world=dcp[0],
            cp_rank=0,
            causal_seqlens_kv_global=dcp[1][row_req],
            backend="cute-dsl",
            return_lse=True,
        )
    else:
        b = _select_mla_decode_backend(q.shape[1])
        if b is not None:
            extra["backend"] = b
    o = trtllm_batch_decode_with_kv_cache_mla(
        query=q.unsqueeze(1),
        kv_cache=kv.unsqueeze(1),
        workspace_buffer=ws,
        qk_nope_head_dim=NOPE,
        kv_lora_rank=KV_LORA,
        qk_rope_head_dim=ROPE,
        block_tables=bt[row_req].contiguous(),
        seq_lens=seq[row_req],
        max_seq_len=int(seq.max()),
        bmm1_scale=SCALE,
        bmm2_scale=1.0,
        **extra,
    )
    if dcp is not None:
        o, lse = o
        return o.reshape(q.shape[0], q.shape[1], KV_LORA), lse.reshape(q.shape[:2])
    return o.reshape(q.shape[0], q.shape[1], KV_LORA)


@pytest.mark.parametrize("backend", [None, "auto"])
@pytest.mark.parametrize("heads", [8, 64])
@pytest.mark.parametrize("k", [4, 5, 7])
@pytest.mark.parametrize("ctxs", [[37], [1000, 3, 9000]])
def test_folded_matches_stock_and_reference(backend, heads, k, ctxs):
    from vllm.models.kimi_k3.nvidia.specialized_decode.drafter.attention import (
        folded_decode,
    )

    R = len(ctxs)
    q, kv, bt, seq = _case(R, ctxs, heads, k, seed=heads * 100 + k)
    ws = torch.zeros(128 << 20, dtype=torch.int8, device="cuda")
    o, lse = folded_decode(
        q,
        kv,
        bt,
        seq,
        int(seq.max()),
        k,
        kv_lora_rank=KV_LORA,
        qk_nope_head_dim=NOPE,
        qk_rope_head_dim=ROPE,
        bmm1_scale=SCALE,
        bmm2_scale=1.0,
        workspace=ws,
        max_reqs=64,
        backend=backend,
    )
    assert lse is None and o.shape == (R * k, heads, KV_LORA)
    ref = _reference(q, kv, bt, seq, k)
    stock = _stock(q, kv, bt, seq, k, ws).float()
    o = o.float()

    def rel(a, b):
        return ((a - b).norm() / b.norm()).item()

    # Both kernels round P to FP8 for the PV MMA and write BF16; the folded call
    # must be as close to the exact result as the stock one is.
    e_fold, e_stock = rel(o, ref), rel(stock, ref)
    assert e_fold < 0.03, e_fold
    assert e_fold < 1.5 * e_stock + 2e-3, (e_fold, e_stock)
    assert rel(o, stock) < 0.03


@pytest.mark.parametrize("k", [4, 5])
def test_dcp_fold_matches_stock(k):
    """DCP (LSE returned): k=4 at 64 heads folds token pairs into 128-head rows;
    output and LSE match the stock one-row-per-token call (same cute-dsl kernel)."""
    from vllm.models.kimi_k3.nvidia.specialized_decode.drafter.attention import (
        fold_plan,
        folded_decode,
    )

    heads, cp = 64, 8
    assert fold_plan(4, heads, True, True) == (2, "cute-dsl")
    assert fold_plan(5, heads, True, True) == (1, "cute-dsl")
    ctxs = [4100, 700]
    q, kv, bt, seq = _case(len(ctxs), ctxs, heads, k, seed=7)
    tot = seq * cp
    ws = torch.zeros(256 << 20, dtype=torch.int8, device="cuda")
    o, lse = folded_decode(
        q,
        kv,
        bt,
        seq,
        int(seq.max()),
        k,
        kv_lora_rank=KV_LORA,
        qk_nope_head_dim=NOPE,
        qk_rope_head_dim=ROPE,
        bmm1_scale=SCALE,
        bmm2_scale=1.0,
        workspace=ws,
        return_lse=True,
        dcp_world_size=cp,
        dcp_rank=0,
        dcp_tot_seq_lens=tot,
    )
    so, slse = _stock(q, kv, bt, seq, k, ws, dcp=(cp, tot))
    assert ((o.float() - so.float()).norm() / so.float().norm()).item() < 0.02
    assert (lse - slse).abs().max().item() < 1e-2


def test_foldable_policy():
    from types import SimpleNamespace

    from vllm.models.kimi_k3.nvidia.specialized_decode.drafter.attention import (
        foldable,
    )

    bt = torch.zeros(4, 2)
    dec = SimpleNamespace(block_table=bt, max_query_len=5)
    md = SimpleNamespace(causal=False, decode=dec, num_decodes=4, num_decode_tokens=20)
    assert foldable(md, 20) == 5
    assert foldable(SimpleNamespace(**{**md.__dict__, "causal": True}), 20) == 0
    assert (
        foldable(SimpleNamespace(**{**md.__dict__, "num_decode_tokens": 19}), 19) == 0
    )
    assert foldable(SimpleNamespace(**{**md.__dict__, "num_decode_tokens": 4}), 4) == 0
    assert foldable(md, 16) == 0
    # Padded requests without query tokens (8 requests, 4 x 4 real tokens): the
    # 2-token average is not the block length; never fold it as k=2.
    padded = SimpleNamespace(
        causal=False,
        decode=SimpleNamespace(block_table=torch.zeros(8, 2), max_query_len=4),
        num_decodes=8,
        num_decode_tokens=16,
    )
    assert foldable(padded, 16) == 0


def test_cake_workspace_overflow_falls_back():
    """128 requests x 5 tokens x 64 heads at 8K: the CAKE plan needs more split
    workspace than the drafter's fixed buffer; the call must still run (trtllm-gen
    fold) and match the stock one-row-per-token call."""
    from vllm.models.kimi_k3.nvidia.specialized_decode.drafter import attention

    R, k, heads = 128, 5, 64
    ctxs = [8192 - 37 * (r % 7) for r in range(R)]
    q, kv, bt, seq = _case(R, ctxs, heads, k, seed=11)
    ws = torch.zeros(attention.CAKE_WORKSPACE_BYTES, dtype=torch.int8, device="cuda")
    o, _ = attention.folded_decode(
        q,
        kv,
        bt,
        seq,
        int(seq.max()),
        k,
        kv_lora_rank=KV_LORA,
        qk_nope_head_dim=NOPE,
        qk_rope_head_dim=ROPE,
        bmm1_scale=SCALE,
        bmm2_scale=1.0,
        workspace=ws,
        max_reqs=R,
    )
    assert (R, heads * k, int(seq.max()), bt.shape[1]) in (
        attention._CAKE_WORKSPACE_TOO_SMALL
    )
    stock = _stock(q, kv, bt, seq, k, ws).float()
    assert ((o.float() - stock).norm() / stock.norm()).item() < 0.03


def test_reserved_cake_workspace_runs_the_large_batch():
    """The same 128 x 5 x 64-head batch with the workspace reserved at load for
    the deployment's largest draft batch: CAKE runs it (no trtllm-gen fallback)
    and matches the stock call and the FP32 reference."""
    from vllm.models.kimi_k3.nvidia.specialized_decode.drafter import attention

    R, k, heads = 128, 5, 64
    ctxs = [8192 - 37 * (r % 7) for r in range(R)]
    q, kv, bt, seq = _case(R, ctxs, heads, k, seed=11)
    max_len = int(seq.max())
    sms = torch.cuda.get_device_properties(0).multi_processor_count
    need = attention.cake_workspace_bytes(R, k, heads, max_len, sms)
    assert need > attention.CAKE_WORKSPACE_BYTES
    ws = torch.zeros(need, dtype=torch.int8, device="cuda")
    attention._CAKE_WORKSPACE_TOO_SMALL.clear()
    o, _ = attention.folded_decode(
        q, kv, bt, seq, max_len, k, kv_lora_rank=KV_LORA,
        qk_nope_head_dim=NOPE, qk_rope_head_dim=ROPE, bmm1_scale=SCALE,
        bmm2_scale=1.0, workspace=ws, max_reqs=R,
    )  # fmt: skip
    assert not attention._CAKE_WORKSPACE_TOO_SMALL
    stock = _stock(q, kv, bt, seq, k, ws).float()
    ref = _reference(q, kv, bt, seq, k)
    err = ((o.float() - ref).norm() / ref.norm()).item()
    err_stock = ((stock - ref).norm() / ref.norm()).item()
    assert err <= 1.5 * err_stock + 2e-3, (err, err_stock)


@pytest.mark.parametrize("heads,k", [(8, 4), (64, 4), (64, 7)])
def test_padded_requests_zero_len(heads, k):
    """CUDA-graph padding: requests past the real batch have seq_len 0 (and the
    padded rows' output is discarded). The fold must not fault on them and the
    real rows must be unchanged."""
    from vllm.models.kimi_k3.nvidia.specialized_decode.drafter.attention import (
        folded_decode,
    )

    ctxs = [9000, 40, 1, 1, 1]
    q, kv, bt, seq = _case(len(ctxs), ctxs, heads, k, seed=5)
    seq[2:] = 0
    ws = torch.zeros(256 << 20, dtype=torch.int8, device="cuda")
    o, _ = folded_decode(
        q,
        kv,
        bt,
        seq,
        9000,
        k,
        kv_lora_rank=KV_LORA,
        qk_nope_head_dim=NOPE,
        qk_rope_head_dim=ROPE,
        bmm1_scale=SCALE,
        bmm2_scale=1.0,
        workspace=ws,
        max_reqs=64,
    )
    torch.accelerator.synchronize()
    n = 2 * k
    ref = _reference(q[:n], kv, bt[:2], seq[:2], k)
    err = ((o[:n].float() - ref).norm() / ref.norm()).item()
    assert err < 0.03, err


def _balanced_args(ctxs, heads, k, cp, seed):
    q, kv, bt, seq = _case(len(ctxs), ctxs, heads, k, seed=seed)
    tot = seq * cp
    ws = torch.zeros(256 << 20, dtype=torch.int8, device="cuda")
    kwargs = dict(
        kv_lora_rank=KV_LORA,
        qk_nope_head_dim=NOPE,
        qk_rope_head_dim=ROPE,
        bmm1_scale=SCALE,
        bmm2_scale=1.0,
        workspace=ws,
        return_lse=True,
        dcp_world_size=cp,
        dcp_rank=0,
        dcp_tot_seq_lens=tot,
    )
    return (q, kv, bt, seq, int(seq.max()), k), kwargs


def test_balanced_plan_matches_reference():
    """The tier plan kernel: the longest rows first (ties: lower row first), moved
    when longer than a pair's share of the batch's keys; the rest keep their
    lengths in the unsplit launch."""
    from vllm.models.kimi_k3.nvidia.specialized_decode.drafter import attention as A

    g = torch.Generator().manual_seed(3)
    for rows, split, pairs in [(124, 28, 76), (40, 12, 76), (64, 28, 74)]:
        sl = torch.randint(1, 5000, (rows,), generator=g, dtype=torch.int32)
        sl[5] = sl[9] = sl.max() + 7  # a tie among the longest
        sl[rows - 1] = 0
        tot = sl * 8 + 3
        bt = torch.randint(0, 1 << 20, (rows, 33), generator=g, dtype=torch.int32)
        dev = [t.cuda() for t in (bt, sl, tot)]
        dst, rest, tbt, tsl, ttot = A._balanced_tables(None, *dev, split, pairs)
        order = sorted(range(rows), key=lambda i: (-int(sl[i]), i))[:split]
        total = int(sl.sum())
        num, den = A.MOVE
        moved = [i for i in order if int(sl[i]) * den * pairs > num * total]
        want_dst = [i if i in moved else rows for i in order]
        assert dst.tolist() == want_dst
        want_rest = sl.clone()
        want_rest[moved] = 0
        assert torch.equal(rest.cpu(), want_rest)
        assert tsl.tolist() == [int(sl[i]) if i in moved else 0 for i in order]
        assert ttot.tolist() == [int(tot[i]) for i in order]
        for j, i in enumerate(order):
            if i in moved:
                assert torch.equal(tbt[j].cpu(), bt[i])


@pytest.mark.parametrize("ctxs", ["skewed", "uniform"])
def test_balanced_dcp_decode_matches_one_launch(ctxs):
    """Wide DCP draft batches (64 heads x k=4: 2 fold rows of 128 heads per
    request; 31 requests = 62 decode rows): the balanced launches against the one
    launch. Rows that stay unsplit are bit-equal; moved rows differ only by the
    kernel's own split-KV (FP32 partials merged, one BF16 rounding); a uniform
    batch wider than the SM pairs (40 requests = 80 rows) moves nothing
    (bit-equal). CUDA-graph replay equals eager."""
    from vllm.models.kimi_k3.nvidia.specialized_decode.drafter import attention as A

    heads, k, cp = 64, 4, 8
    reqs = 31 if ctxs == "skewed" else 40
    if ctxs == "skewed":
        g = torch.Generator().manual_seed(11)
        lens = torch.randint(64, 3000, (reqs,), generator=g).tolist()
        lens[3], lens[17], lens[20] = 60000, 41000, 25000
    else:
        lens = [6000] * reqs
    args, kwargs = _balanced_args(lens, heads, k, cp, seed=5)
    rows = reqs * 2
    assert A.balanced_tiers(args[0].device, rows) == (4, 8, 16)
    policy = A.balanced_tiers
    try:
        A.balanced_tiers = lambda device, n: ()
        o0, l0 = A.folded_decode(*args, **kwargs, table_cache={})
    finally:
        A.balanced_tiers = policy
    o1, l1 = A.folded_decode(*args, **kwargs, table_cache={})
    torch.accelerator.synchronize()
    d_o = (o1.float() - o0.float()).abs().view(rows, -1).amax(1)
    d_l = (l1 - l0).abs().view(rows, -1).amax(1)
    same = (d_o == 0) & (d_l == 0)
    moved_reqs = {3, 17, 20} if ctxs == "skewed" else set()
    moved = torch.tensor([r // 2 in moved_reqs for r in range(rows)], device="cuda")
    assert bool(same[~moved].all())
    if moved_reqs:
        assert float(d_o[moved].max()) < 2e-3
        assert float(d_l[moved].max()) < 1e-4
        rel = (o1.float() - o0.float()).norm() / o0.float().norm()
        assert float(rel) < 1e-2
    assert torch.isfinite(o1.float()).all() and torch.isfinite(l1).all()

    # Graph capture: the same launches replayed (cache built inside the graph).
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        A.folded_decode(*args, **kwargs, table_cache={})
    torch.cuda.current_stream().wait_stream(s)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        go, gl = A.folded_decode(*args, **kwargs, table_cache={})
    graph.replay()
    torch.accelerator.synchronize()
    assert torch.equal(go, o1) and torch.equal(gl, l1)
