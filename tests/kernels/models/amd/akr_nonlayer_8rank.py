# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Eight-rank checks of the Kimi-K3.1 non-layer amd_kernels routes against the stock
paths, on random weights at the real shapes (vllm/models/kimi_k3/amd/akr_*.py):

  embed   `akr_embed.install` vs `VocabParallelEmbedding.forward` (eager + graph replay)
  lmhead  `akr_lmhead` all-gather mode (target eager, draft in a graph) and owner mode
          (`owned_logits`) vs `LogitsProcessor` (GEMM + all-gather)
  ctx     `akr_draft.ContextProj` (eager + graph path) vs context_proj + context_norm

Run (spawns 8 ranks, needs 8 GPUs):
    python tests/kernels/models/amd/akr_nonlayer_8rank.py [embed lmhead ctx]
"""

import sys
from types import SimpleNamespace

import torch
import torch.multiprocessing as mp

HIDDEN, VOCAB = 7168, 163840


def _free_gb():
    torch.accelerator.synchronize()
    return torch.accelerator.get_memory_info()[0] / 2**30


def _cos_rel(a, b):
    a, b = a.float().flatten(), b.float().flatten()
    return float(torch.nn.functional.cosine_similarity(a, b, dim=0)), float(
        (a - b).norm() / b.norm()
    )


def test_embed(rank):
    from vllm.model_executor.layers.vocab_parallel_embedding import (
        VocabParallelEmbedding,
    )
    from vllm.models.kimi_k3.amd import akr_embed

    torch.manual_seed(100 + rank)
    emb = VocabParallelEmbedding(VOCAB, HIDDEN)
    emb.weight.data.copy_(
        torch.randn_like(emb.weight, dtype=torch.float32).to(torch.bfloat16)
    )
    torch.manual_seed(7)
    batches = [
        torch.randint(0, VOCAB, (m,), device="cuda") for m in (1, 12, 80, 192, 256)
    ]
    batches.append(torch.tensor([0, 20479, 20480, 163839, 5], device="cuda"))
    ref = [emb(b.clone()).clone() for b in batches]
    assert akr_embed.install(emb), "embedding kernel not installed"
    for dt in (torch.int64, torch.int32):
        for b, r in zip(batches, ref):
            out = emb(b.to(dt))
            assert out.shape == r.shape and out.dtype == r.dtype
            assert torch.equal(out, r), (rank, b.shape, dt)
    # 2-D ids keep their shape; a non-contiguous id view takes the stock path
    b = batches[1]
    assert torch.equal(emb(b.view(3, 4)), ref[1].view(3, 4, HIDDEN))
    assert torch.equal(emb(torch.stack([b, b], 1)[:, 0]), ref[1])
    # graph replay with new ids (stock results recomputed through the stock forward)
    stock = emb.__class__.forward
    ids = batches[2].clone()
    emb(ids)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        out = emb(ids)
    for b in batches[2:4] + batches[1:2]:
        ids.copy_(
            b[: ids.numel()] if b.numel() >= ids.numel() else torch.zeros_like(ids)
        )
        g.replay()
        torch.accelerator.synchronize()
        assert torch.equal(out, stock(emb, ids.clone()))
    akr_embed.close()
    return "embed ok"


def _vllm_config():
    return SimpleNamespace(
        scheduler_config=SimpleNamespace(max_num_seqs=48),
        speculative_config=SimpleNamespace(num_speculative_tokens=3),
    )


def test_lmhead(rank):
    from vllm.distributed import get_tp_group
    from vllm.model_executor.layers.layernorm import RMSNorm
    from vllm.model_executor.layers.logits_processor import LogitsProcessor
    from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead
    from vllm.models.kimi_k3.amd import akr_lmhead

    torch.manual_seed(200 + rank)
    lm_head = ParallelLMHead(VOCAB, HIDDEN)
    lm_head.weight.data.copy_(
        (torch.randn_like(lm_head.weight, dtype=torch.float32) * 0.02).to(
            torch.bfloat16
        )
    )
    lp = LogitsProcessor(VOCAB)
    norm = RMSNorm(HIDDEN, eps=1e-5).to(torch.bfloat16).cuda()
    cfg = _vllm_config()
    f0 = _free_gb()
    assert akr_lmhead.setup("target", lm_head, lp, cfg)
    f1 = _free_gb()
    assert akr_lmhead.setup("draft", lm_head, lp, cfg)
    f2 = _free_gb()
    if rank == 0:
        print(
            f"lm_head exchange memory per rank: target {f0 - f1:.3f} GB, draft "
            f"{f1 - f2:.3f} GB",
            flush=True,
        )
    torch.manual_seed(5)
    # all-gather mode, eager (target; consecutive calls alternate buffers)
    for rows in (1, 8, 12, 16, 17, 40, 80, 192):
        x = torch.randn(rows, HIDDEN, device="cuda").to(torch.bfloat16)
        ref = lp(lm_head, x)
        out = akr_lmhead.logits("target", lm_head, x)
        assert out is not None and out.shape == ref.shape == (rows, VOCAB)
        cos, rel = _cos_rel(out, ref)
        assert cos > 0.99999 and rel < 2e-3, (rows, cos, rel)
    assert (
        akr_lmhead.logits(
            "target", lm_head, torch.randn(81, HIDDEN, device="cuda").to(torch.bfloat16)
        )
        is not None
    )
    assert (
        akr_lmhead.logits(
            "target", lm_head, torch.randn(81, HIDDEN, device="cuda").float()
        )
        is None
    )
    # two consecutive calls: both results stay valid (two buffers)
    x1 = torch.randn(40, HIDDEN, device="cuda").to(torch.bfloat16)
    x2 = torch.randn(40, HIDDEN, device="cuda").to(torch.bfloat16)
    r1 = akr_lmhead.logits("target", lm_head, x1)
    r2 = akr_lmhead.logits("target", lm_head, x2)
    assert (
        _cos_rel(r1, lp(lm_head, x1))[0] > 0.99999
        and _cos_rel(r2, lp(lm_head, x2))[0] > 0.99999
    )
    # the draft's exchange in a CUDA graph (buffer 0), replayed with new inputs
    for rows in (12, 60):
        xs = torch.randn(rows, HIDDEN, device="cuda").to(torch.bfloat16)
        akr_lmhead.logits("draft", lm_head, xs)
        torch.accelerator.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            gout = akr_lmhead.logits("draft", lm_head, xs)
        for i in range(4):
            xs.copy_(torch.randn(rows, HIDDEN, device="cuda").to(torch.bfloat16))
            graph.replay()
            torch.accelerator.synchronize()
            cos, rel = _cos_rel(gout, lp(lm_head, xs))
            assert cos > 0.99999 and rel < 2e-3, (rows, i, cos, rel)
    # owner mode: rows sorted by owner, each rank receives its own rows
    tp = get_tp_group().world_size
    for rows in (8, 24, 80, 192):
        counts = [rows // tp + int(r < rows % tp) for r in range(tp)]
        if rows == 24:
            counts = [3, 0, 5, 1, 7, 2, 0, 6]
        assert sum(counts) == rows
        md = SimpleNamespace(num_logits_per_rank=counts, num_local_logits=counts[rank])
        hidden = torch.randn(rows, HIDDEN, device="cuda").to(torch.bfloat16)
        perm = torch.randperm(rows, device="cuda")
        out = akr_lmhead.owned_logits(lm_head, norm, hidden, perm, md)
        assert out is not None and out.shape == (counts[rank], VOCAB)
        full = lp(lm_head, norm(hidden[perm], None))
        start = sum(counts[:rank])
        if counts[rank]:
            cos, rel = _cos_rel(out, full[start : start + counts[rank]])
            assert cos > 0.99999 and rel < 2e-3, (rows, cos, rel)
    # hidden rows beyond 256: the rows are gathered before the norm
    rows = 40
    counts = [5] * 8
    md = SimpleNamespace(num_logits_per_rank=counts, num_local_logits=5)
    hidden = torch.randn(300, HIDDEN, device="cuda").to(torch.bfloat16)
    sel = torch.randperm(300, device="cuda")[:rows]
    out = akr_lmhead.owned_logits(lm_head, norm, hidden, sel, md)
    full = lp(lm_head, norm(hidden[sel], None))
    assert _cos_rel(out, full[5 * rank : 5 * rank + 5])[0] > 0.99999
    return "lmhead ok"


def test_ctx(rank):
    from vllm.model_executor.layers.layernorm import RMSNorm
    from vllm.models.kimi_k3.amd import akr_draft

    torch.manual_seed(300)  # replicated weights: the same on every rank
    w = (torch.randn(HIDDEN, 35840, device="cuda") * 0.01).to(torch.bfloat16)
    norm = RMSNorm(HIDDEN, eps=1e-5).to(torch.bfloat16).cuda()
    norm.weight.data.copy_((torch.rand(HIDDEN, device="cuda") + 0.5).to(torch.bfloat16))
    model = SimpleNamespace(
        model=SimpleNamespace(
            context_proj=SimpleNamespace(weight=SimpleNamespace(data=w)),
            context_norm=norm,
        )
    )
    f0 = _free_gb()
    ctx = akr_draft.ContextProj(model, torch.device("cuda"))
    ctx.warm_up(torch.device("cuda"))
    if rank == 0:
        print(
            f"context_proj extra memory per rank: {f0 - _free_gb():.3f} GB (packed "
            "shard + staging + slabs + mailbox)",
            flush=True,
        )

    def stock(aux):
        return norm(torch.cat(aux, dim=-1) @ w.t())

    def exact(aux):
        # the kernel's contract (reference.py): fp32 accumulation, the projection
        # rounded to bf16, then the norm in fp32 with one rounding (AITER's rms_norm;
        # vLLM's native RMSNorm rounds twice)
        y = (torch.cat(aux, dim=-1).float() @ w.float().t()).to(torch.bfloat16).float()
        y = (
            y
            * torch.rsqrt(y.pow(2).mean(-1, keepdim=True) + 1e-5)
            * norm.weight.float()
        )
        return y.to(torch.bfloat16)

    torch.manual_seed(9)
    for rows in (12, 40, 60, 65, 144, 192):
        aux = [
            torch.randn(rows + 5, HIDDEN, device="cuda").to(torch.bfloat16)
            for _ in range(5)
        ]
        out = torch.empty(rows + 5, HIDDEN, dtype=torch.bfloat16, device="cuda")
        ctx.combine_eager(aux, 3, 3 + rows, out[3 : 3 + rows])
        part = [h[3 : 3 + rows] for h in aux]
        cos, rel = _cos_rel(out[3 : 3 + rows], exact(part))
        assert cos > 0.99999 and rel < 1e-3, (rows, cos, rel)
        assert _cos_rel(out[3 : 3 + rows], stock(part))[1] < 6e-3
    # the graph path: pack eagerly, project in the graph
    rows = 60
    aux = [
        torch.randn(rows, HIDDEN, device="cuda").to(torch.bfloat16) for _ in range(5)
    ]
    out = torch.zeros(rows, HIDDEN, dtype=torch.bfloat16, device="cuda")
    ctx.pack_graph(aux, rows)
    ctx.combine_graph(rows, out)
    torch.accelerator.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        ctx.combine_graph(rows, out)
    for i in range(3):
        aux = [
            torch.randn(rows, HIDDEN, device="cuda").to(torch.bfloat16)
            for _ in range(5)
        ]
        ctx.pack_graph(aux, rows)
        graph.replay()
        torch.accelerator.synchronize()
        cos, rel = _cos_rel(out, exact(aux))
        assert cos > 0.99999 and rel < 1e-3, (i, cos, rel)
    # the speculator hooks: graph rows packed eagerly + projected in the graph step,
    # the remaining rows eagerly (<= 192) or by the stock combine (> 192)
    spec = SimpleNamespace(
        _akr_ctx=ctx,
        _aux_staging=object(),
        hidden_states=torch.zeros(512, HIDDEN, dtype=torch.bfloat16, device="cuda"),
        _num_graph_context_tokens=lambda n: n * 4,
        prepare_context_anchor=lambda *a: None,
        target_input_buffers=SimpleNamespace(query_start_loc=None),
        _num_rejected=None,
        _precompute_context_kv=lambda a, b: None,
        supports_decode_metadata_update=False,
        _update_draft_decode_metadata=lambda *a: None,
        model=SimpleNamespace(combine_hidden_states=lambda x: norm(x @ w.t())),
    )
    for num_tokens, num_graph in (
        (50, 40),
        (40, 40),
        (30, 40),
        (300, 48),
        (400, 0),
        (60, 0),
    ):
        aux = [
            torch.randn(num_tokens, HIDDEN, device="cuda").to(torch.bfloat16)
            for _ in range(5)
        ]
        spec.hidden_states.zero_()
        assert akr_draft.store_hidden_states(spec, num_tokens, None, aux, num_graph)
        start = min(num_tokens, num_graph)
        if start:
            assert akr_draft.prepare_graph_forward(spec, num_graph // 4, None)
        torch.accelerator.synchronize()
        got = spec.hidden_states[:num_tokens]
        want = exact(aux)
        cos, rel = _cos_rel(got, want)
        assert cos > 0.99999 and rel < 6e-3, (num_tokens, num_graph, cos, rel)
    ctx.comm.close()
    return "ctx ok"


TESTS = {"embed": test_embed, "lmhead": test_lmhead, "ctx": test_ctx}


def worker(rank, port, names):
    torch.accelerator.set_device_index(rank)
    torch.set_default_dtype(torch.bfloat16)
    torch.set_default_device(f"cuda:{rank}")
    from tests.utils import init_test_distributed_environment
    from vllm.config import VllmConfig, set_current_vllm_config

    with set_current_vllm_config(VllmConfig()):
        init_test_distributed_environment(8, 1, rank, str(port))
        for n in names:
            msg = TESTS[n](rank)
            torch.accelerator.synchronize()
            torch.distributed.barrier()
            if rank == 0:
                print(msg, flush=True)


if __name__ == "__main__":
    names = sys.argv[1:] or list(TESTS)
    port = 29500 + (hash(tuple(names)) % 400)
    mp.spawn(worker, args=(port, names), nprocs=8)
    print("PASS", names)
