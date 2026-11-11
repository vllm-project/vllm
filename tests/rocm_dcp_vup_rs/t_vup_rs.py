# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# 8-GPU test + microbench of the v_up-first DCP combine (VLLM_ROCM_DCP_VUP_RS=1) against the
# a2a combine + local v_up it replaces, using the real mum MLADCPManager / communicator stack.
# Run: python -m torch.distributed.run --nproc-per-node 8 --master-port 29671 t_vup_rs.py
# env: MODE=test|bench|both, TS=64,100,125, NW=52 (graph calls), REP=4
import os, json, statistics, math, torch, torch.distributed as dist

os.environ.setdefault("VLLM_ROCM_USE_AITER", "1")
os.environ.setdefault("VLLM_ROCM_DCP_AITER_ALLGATHER", "1")
os.environ.setdefault("VLLM_ROCM_DCP_VUP_RS", "1")
rank = int(os.environ["RANK"])
W = int(os.environ["WORLD_SIZE"])
lr = int(os.environ["LOCAL_RANK"])
torch.cuda.set_device(lr)
dev = torch.device("cuda", lr)

from vllm.config import VllmConfig, ParallelConfig, set_current_vllm_config
from vllm.distributed.parallel_state import (
    init_distributed_environment,
    initialize_model_parallel,
    get_dcp_group,
    graph_capture,
)

pc = ParallelConfig(
    tensor_parallel_size=W, decode_context_parallel_size=W, dcp_comm_backend="a2a"
)
vc = VllmConfig(parallel_config=pc)
MODE = os.environ.get("MODE", "both")
TS = [int(x) for x in os.environ.get("TS", "64,100,125").split(",")]
NW = int(os.environ.get("NW", "52"))
REP = int(os.environ.get("REP", "4"))
OUTJ = os.environ.get("OUT")  # optional JSON dump of the results
RES = {}
BF = torch.bfloat16
H, HL, L, V, HP = (
    96,
    96 // W,
    512,
    128,
    128,
)  # HP: ASM-padded head count of the partial output
FAIL = []


def log(*a):
    if rank == 0:
        print(*a, flush=True)


with set_current_vllm_config(vc):
    init_distributed_environment(
        world_size=W, rank=rank, local_rank=lr, distributed_init_method="env://"
    )
    initialize_model_parallel(
        tensor_model_parallel_size=W, decode_context_model_parallel_size=W
    )
    from vllm.v1.attention.ops.dcp import MLADCPManager

    mgr = MLADCPManager(
        vllm_config=vc,
        device=dev,
        num_heads=HL,
        query_head_dim=L + 64,
        output_head_dim=L,
        query_dtype=BF,
        output_dtype=BF,
        padded_num_heads=None,
        is_lse_base_on_e=True,
        use_pcp=False,
    )
log(f"use_vup_rs={mgr.use_vup_rs} use_a2a={mgr.use_a2a} direct_a2a={mgr.direct_a2a}")
assert mgr.use_vup_rs
dcp = get_dcp_group()
assert dcp.rank_in_group == rank

# full W_UV identical on all ranks; local = this rank's heads; W_UV_dcp built like process_weights_after_loading
gw = torch.Generator(device=dev)
gw.manual_seed(1234)
WFULL = (torch.randn(H, L, V, device=dev, generator=gw) * 0.05).to(BF)
WLOC = WFULL[rank * HL : (rank + 1) * HL].clone()
WDCP = dcp.all_gather(WLOC.contiguous(), dim=0)
assert torch.equal(WDCP, WFULL), "W_UV_dcp gather order"


def make_inputs(T, nreq, pad, seed, all_empty_req=3, nan=True):
    """Per-rank partial out [T,HP,L] (heads >= H garbage), lse as a transposed [HP,T] buffer, local seq_lens,
    query_start_loc with `pad` graph-padding rows at the end. Empty shards: req 0 on odd ranks, req 7 on
    ranks >= 6, req `all_empty_req` on all ranks; their rows (and padding rows) get NaN partials/lse."""
    g = torch.Generator(device=dev)
    g.manual_seed(seed * 100 + rank)
    A = (torch.randn(T, HP, L, device=dev, generator=g) * 0.5).to(BF)
    lseb = torch.randn(HP, T, device=dev, generator=g) * 1.5 + 3.0
    lse = lseb.t()  # [T, HP] with strides (1, T): an attention-backend style layout
    rows = (T - pad) // nreq
    qsl = torch.arange(nreq + 1, device=dev, dtype=torch.int32) * rows
    qsl[-1] = T - pad
    seql = torch.full((nreq,), 1000, device=dev, dtype=torch.int32)
    if rank % 2 == 1:
        seql[0] = 0
    if rank >= 6 and nreq > 7:
        seql[7] = 0
    if all_empty_req is not None and nreq > all_empty_req:
        seql[all_empty_req] = 0
    return A, lse, seql, qsl


def reference(A, lse, seql, qsl, T):
    A_all = torch.empty(W, T, HP, L, device=dev, dtype=BF)
    dist.all_gather_into_tensor(A_all, A.contiguous())
    L_all = torch.empty(W, T, HP, device=dev)
    dist.all_gather_into_tensor(L_all, lse.contiguous())
    S_all = torch.empty(W, seql.numel(), device=dev, dtype=torch.int32)
    dist.all_gather_into_tensor(S_all, seql)
    rid = torch.searchsorted(
        qsl[1:], torch.arange(T, device=dev, dtype=torch.int32), right=True
    ).clamp(max=seql.numel() - 1)
    pad = torch.arange(T, device=dev) >= qsl[-1]
    emp = (S_all[:, rid] == 0) | pad[None, :]  # [W,T]
    h0, h1 = rank * HL, (rank + 1) * HL
    lm = L_all[:, :, h0:h1].masked_fill(emp[:, :, None], float("-inf"))
    wts = torch.nan_to_num(torch.softmax(lm, 0), nan=0.0)
    Ah = A_all[:, :, h0:h1].float().masked_fill(emp[:, :, None, None], 0.0)
    pr = torch.einsum("sthl,hlv->sthv", Ah, WFULL[h0:h1].float())
    return (wts[..., None] * pr).sum(0), emp


def inject_nan(A, lse, emp):
    m = emp[rank]
    A[m] = float("nan")
    lse[m] = float("nan")


def old_path(A, lse, seql, qsl, out):
    # the backend hands the combine a [:, :H] view of the head-padded buffers
    lat = mgr.combine(
        A[:, :H], lse[:, :H], seq_lens=seql, query_start_loc=qsl
    )  # [T,HL,L]
    torch.bmm(lat.transpose(0, 1), WLOC_cur[0], out=out.transpose(0, 1))


def new_path(A, lse, seql, qsl, out):
    ok = mgr.vup_rs_combine(
        A[:, :H], lse[:, :H], WDCP_cur[0], out, seq_lens=seql, query_start_loc=qsl
    )
    assert ok, "vup_rs_combine fell back"


WLOC_cur = [WLOC]
WDCP_cur = [WDCP]


def err(out, ref):
    d = out.float() - ref
    nan = torch.isnan(out.float()).any().float()
    e = torch.tensor(
        [
            d.nan_to_num(1e9).abs().max().item(),
            (d.nan_to_num(1e9).norm() / ref.norm()).item(),
            nan.item(),
        ],
        device=dev,
    )
    dist.all_reduce(e, op=dist.ReduceOp.MAX)
    return e.tolist()


def check(tag, out, ref, tol=8e-3):
    mx, rl, nan = err(out, ref)
    ok = rl < tol and nan == 0
    log(
        f"{'PASS' if ok else 'FAIL'} {tag:58s} maxabs {mx:.4f} relL2 {rl:.2e} nan {int(nan)}"
    )
    RES[tag] = [mx, rl, nan]
    if not ok:
        FAIL.append(tag)


# ------------------------------------------------------------------------------------------------ tests
if MODE in ("test", "both"):
    for T, nreq, pad in (
        (125, 25, 0),
        (100, 24, 4),
        (64, 16, 0),
        (61, 15, 1),
        (4, 1, 0),
        (8, 2, 2),
    ):
        A, lse, seql, qsl = make_inputs(T, nreq, pad, seed=T)
        ref, emp = reference(A, lse, seql, qsl, T)
        inject_nan(A, lse, emp)
        for name, fn in (("old", old_path), ("new", new_path)):
            out = torch.full((T, HL, V), 7.0, device=dev, dtype=BF)
            fn(A, lse, seql, qsl, out)
            torch.cuda.synchronize()
            check(f"eager {name} T={T} nreq={nreq} pad={pad}", out, ref)
            # all-empty / padded rows must be exactly 0
            z = out[emp.all(0)].float().abs().max().item() if emp.all(0).any() else 0.0
            zt = torch.tensor([z], device=dev)
            dist.all_reduce(zt, op=dist.ReduceOp.MAX)
            if zt.item() != 0.0:
                FAIL.append(f"eager {name} T={T} all-empty rows nonzero {zt.item()}")
            log(
                f"     all-empty/pad rows: {int(emp.all(0).sum())}  max|out| there {zt.item()}"
            )
    # num_seqs == 0 (every row padding): output must be all zeros
    T = 16
    A, lse, _, _ = make_inputs(T, 1, 0, seed=5)
    seql = torch.zeros(0, device=dev, dtype=torch.int32)
    qsl = torch.zeros(1, device=dev, dtype=torch.int32)
    out = torch.full((T, HL, V), 7.0, device=dev, dtype=BF)
    new_path(A, lse, seql, qsl, out)
    torch.cuda.synchronize()
    zt = torch.tensor([out.float().abs().max().item()], device=dev)
    dist.all_reduce(zt, op=dist.ReduceOp.MAX)
    log(
        f"{'PASS' if zt.item() == 0 else 'FAIL'} new num_seqs=0 -> zeros (max {zt.item()})"
    )
    if zt.item() != 0:
        FAIL.append("num_seqs=0")

    # CUDA graph capture/replay: capture with one input set, replay with others copied in place.
    for T, nreq, pad in ((125, 25, 0), (100, 24, 4), (64, 16, 3)):
        A, lse, seql, qsl = make_inputs(T, nreq, pad, seed=1)
        lse_buf = lse  # transposed view of a persistent buffer
        outs = {k: torch.zeros(T, HL, V, device=dev, dtype=BF) for k in ("old", "new")}
        graphs = {}
        with graph_capture(dev) as ctx:
            for k, fn in (("old", old_path), ("new", new_path)):
                fn(
                    A, lse_buf, seql, qsl, outs[k]
                )  # warmup inside the capture context (aiter warmup path)
                torch.cuda.synchronize()
                g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g, stream=ctx.stream):
                    fn(A, lse_buf, seql, qsl, outs[k])
                graphs[k] = g
        torch.cuda.synchronize()
        dist.barrier()
        for it, (seed, aer) in enumerate(((11, 3), (12, None), (13, 0))):
            A2, lse2, seql2, qsl2 = make_inputs(
                T, nreq, pad, seed=seed, all_empty_req=aer
            )
            ref, emp = reference(A2, lse2, seql2, qsl2, T)
            inject_nan(A2, lse2, emp)
            A.copy_(A2)
            lse_buf.copy_(lse2)
            seql.copy_(seql2)
            qsl.copy_(qsl2)
            for k, g in graphs.items():
                outs[k].fill_(7.0)
                g.replay()
                torch.cuda.synchronize()
                check(f"graph {k} T={T} replay{it} (all-empty req {aer})", outs[k], ref)
        del graphs
        torch.cuda.synchronize()
        dist.barrier()
    log("TESTS", "ALL PASS" if not FAIL else f"FAILURES: {FAIL}")


# ------------------------------------------------------------------------------------------------ bench
def measure(graphs, ncalls):
    ts = {k: [] for k in graphs}
    e0 = torch.cuda.Event(enable_timing=True)
    e1 = torch.cuda.Event(enable_timing=True)
    for _ in range(7):
        for k, g in graphs.items():
            dist.barrier()
            torch.cuda.synchronize()
            e0.record()
            for _ in range(REP):
                g.replay()
            e1.record()
            torch.cuda.synchronize()
            ts[k].append(e0.elapsed_time(e1) * 1000 / (ncalls * REP))
    out = {}
    for k in graphs:
        u = torch.tensor([statistics.median(ts[k])], device=dev)
        dist.all_reduce(u, op=dist.ReduceOp.MAX)
        out[k] = u.item()
    return out


if MODE in ("bench", "both"):
    WD = [
        (torch.randn(H, L, V, device=dev, generator=gw) * 0.05).to(BF)
        for _ in range(NW)
    ]  # >= 640 MB rotation
    WLs = [w[rank * HL : (rank + 1) * HL] for w in WD]
    log(f"bench: W_UV rotation {NW} x {H * L * V * 2 / 1e6:.1f} MB")
    for T in TS:
        nreq = 16 if T == 64 else 25
        A, lse, seql, qsl = make_inputs(
            T, nreq, T - (T // nreq) * nreq, seed=3, all_empty_req=None
        )
        outs = {k: torch.zeros(T, HL, V, device=dev, dtype=BF) for k in ("old", "new")}
        graphs = {}
        with graph_capture(dev) as ctx:
            for k, fn in (("old", old_path), ("new", new_path)):

                def call(i, fn=fn, k=k):
                    WLOC_cur[0] = WLs[i]
                    WDCP_cur[0] = WD[i]
                    fn(A, lse, seql, qsl, outs[k])

                call(0)
                call(1)
                torch.cuda.synchronize()
                g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g, stream=ctx.stream):
                    for i in range(NW):
                        call(i)
                graphs[k] = g
        torch.cuda.synchronize()
        dist.barrier()
        for k, g in graphs.items():
            g.replay()
        torch.cuda.synchronize()
        dist.barrier()
        r = measure(graphs, NW)
        for k, u in r.items():
            RES[f"bench T={T} {k}"] = u
        log(
            f"BENCH T={T}: old (a2a D512 + torch.bmm 12h) {r['old']:.2f} us | new (vup-first + AG + scale + RS) {r['new']:.2f} us | "
            f"saving {r['old'] - r['new']:.2f} us/layer -> x29 layers {29 * (r['old'] - r['new']) / 1000:.3f} ms/step"
        )
        del graphs
        torch.cuda.synchronize()
        dist.barrier()
    WLOC_cur[0] = WLOC
    WDCP_cur[0] = WDCP

dist.barrier()
if rank == 0:
    if OUTJ:
        os.makedirs(os.path.dirname(OUTJ) or ".", exist_ok=True)
        json.dump(RES, open(OUTJ, "w"), indent=1)
    print("DONE", "FAIL" if FAIL else "OK", flush=True)
