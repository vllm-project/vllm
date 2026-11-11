# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""VLLM_ROCM_DSA_MULTISTREAM: one real Kimi-K3.1 DSA layer on 8 GPUs (TP8/DCP8).

torchrun --nproc-per-node 8 tests/kimi_dsams/run_dsams.py [--bench]

Builds a real ROCm ``KimiK3DSAAttention`` (Kimi-K3.1-Pro dims, random bf16
weights, fp8_ds_mla MLA cache, MXFP4 indexer cache, nst4 decode threshold) and
compares the serial path (``MultiHeadLatentAttentionWrapper.forward``) with the
multi-stream wrapper on decode steps:
  * pre-gate attention and the merged CSR top-k slots are bitwise equal;
  * the final output (gate * attn -> o_proj -> TP all-reduce) is close to the
    serial one and both are compared against an fp32 reference;
  * edge cases: empty DCP shards / all-empty local rows (tiny sequences),
    mixed decode query lengths, padded requests (query 0, seq 0);
  * FULL CUDA graph capture inside ``graph_capture()`` and replay with new
    inputs (hidden states and cache contents), old vs new;
  * --bench: layer graph replay time old vs new at T = 64, 100, 125.
"""

import argparse
import dataclasses
import json
import math
import os
import sys
import time

os.environ["VLLM_ROCM_DSA_MULTISTREAM"] = "1"
os.environ.setdefault("VLLM_ROCM_DCP_AITER_ALLGATHER", "1")
os.environ.setdefault("VLLM_USE_DIRECT_DCP_A2A", "0")
os.environ.setdefault("VLLM_USE_DIRECT_DCP_Q_GATHER", "0")
os.environ.setdefault("VLLM_USE_DIRECT_DCP_KV_GATHER", "0")
os.environ.setdefault("VLLM_ROCM_USE_AITER_FLYDSL_SPARSE_MLA", "1")
# --det: deterministic GEMMs (no ROCm skinny GEMMs, no AITER tuned GEMMs, and
# the batch-invariant Triton overrides of aten mm/linear/bmm). The default GEMMs
# (wvSplitKrc, tgemm) are not run-to-run reproducible, and with random weights
# a 1-ulp change of q_c flips top-k selections, which moves the output by ~10%.
DET = "--det" in sys.argv
if DET:
    os.environ["VLLM_ROCM_USE_SKINNY_GEMM"] = "0"
    os.environ["VLLM_ROCM_USE_AITER_LINEAR"] = "0"

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.distributed as dist  # noqa: E402

MODEL = "/share_nfs/models/moonshotai/Kimi-K3.1-Pro"
DRAFT = "/share_nfs/models/Inferact/Kimi-K3.1-Pro-DSpark-preview"
NST = 4
PAGE = 64
NUM_BLOCKS = 1536
MAX_MODEL_LEN = 65536
FAILS: list[str] = []


def log(rank, *a):
    if rank == 0:
        print(*a, flush=True)


def check(rank, ok, msg):
    if not ok:
        FAILS.append(msg)
        print(f"[rank {rank}] FAIL: {msg}", flush=True)


def make_config():
    from vllm.config import AttentionConfig
    from vllm.engine.arg_utils import EngineArgs

    ea = EngineArgs(
        model=MODEL,
        trust_remote_code=True,
        tensor_parallel_size=8,
        decode_context_parallel_size=8,
        cp_kv_cache_interleave_size=1,
        dcp_comm_backend="a2a",
        kv_cache_dtype="fp8",
        attention_config=AttentionConfig(indexer_kv_dtype="mxfp4"),
        max_num_seqs=32,
        max_num_batched_tokens=8192,
        max_model_len=MAX_MODEL_LEN,
        load_format="dummy",
        skip_tokenizer_init=True,
        language_model_only=True,
        enforce_eager=True,
        speculative_config={
            "method": "dspark",
            "model": DRAFT,
            "num_speculative_tokens": NST,
            "attention_backend": "ROCM_AITER_MLA",
            "kv_cache_dtype": "fp8",
        },
    )
    return ea.create_engine_config()


class Harness:
    def __init__(self, config, rank, world, device):
        from vllm.models.kimi_k3.amd import dsa as kdsa
        from vllm.models.kimi_k3.common.dsa_cache import KimiK3IndexerMetadataBuilder
        from vllm.v1.attention.backends.mla.rocm_aiter_mla_sparse import (
            ROCMAiterMLASparseMetadataBuilder,
        )
        from vllm.v1.worker.block_table import get_block_table_width

        self.config, self.rank, self.world, self.device = config, rank, world, device
        hf = config.model_config.hf_text_config
        self.hf = hf
        self.topk_buf = torch.empty(
            (config.scheduler_config.max_num_batched_tokens, hf.dsa_config.index_topk),
            dtype=torch.int32,
            device=device,
        )
        torch.set_default_dtype(torch.bfloat16)
        with torch.device(device):
            attn = kdsa.KimiK3DSAAttention(
                config=hf,
                hidden_size=hf.hidden_size,
                num_heads=hf.num_attention_heads,
                qk_nope_head_dim=hf.qk_nope_head_dim,
                qk_rope_head_dim=hf.qk_rope_head_dim,
                v_head_dim=hf.v_head_dim,
                q_lora_rank=hf.q_lora_rank,
                kv_lora_rank=hf.kv_lora_rank,
                cache_config=config.cache_config,
                quant_config=None,
                prefix="model.layers.4.self_attn",
                topk_indices_buffer=self.topk_buf,
            )
        torch.set_default_dtype(torch.float32)
        assert isinstance(attn.mla_attn, kdsa.KimiK3DSAMultiStreamMLAWrapper), type(
            attn.mla_attn
        )
        self.attn = attn
        self.wrapper = attn.mla_attn
        self.layer = attn.mla_attn.mla_attn
        self.ind = attn.dsa_indexer
        # Random weights, identical on every rank (replicated ones must be).
        gen = torch.Generator(device=device).manual_seed(1234)
        for name, p in sorted(attn.named_parameters()):
            with torch.no_grad():
                if p.dim() == 1:
                    p.copy_(
                        1.0
                        + 0.05
                        * torch.randn(p.shape, generator=gen, device=device).to(p.dtype)
                    )
                else:
                    std = 1.0 / math.sqrt(p.shape[-1])
                    p.copy_(
                        (std * torch.randn(p.shape, generator=gen, device=device)).to(
                            p.dtype
                        )
                    )
        self.w_g = attn.g_proj.weight.detach().float().clone()
        self.w_o = attn.o_proj.weight.detach().float().clone()
        from vllm.model_executor.model_loader.utils import (
            process_weights_after_loading,
        )

        process_weights_after_loading(attn, config.model_config, device)

        # Caches (64-token pages for both), filled with valid random content.
        spec = dataclasses.replace(
            self.layer.get_kv_cache_spec(config), block_size=PAGE
        )
        self.spec = spec
        row = spec.page_size_bytes // PAGE
        self.mla_cache = torch.zeros(
            (NUM_BLOCKS, PAGE, row), dtype=torch.uint8, device=device
        )
        ispec = dataclasses.replace(
            self.ind.k_cache.get_kv_cache_spec(config), block_size=PAGE
        )
        self.ispec = ispec
        irow = ispec.page_size_bytes // PAGE
        self.index_cache = torch.zeros(
            (NUM_BLOCKS, PAGE, irow), dtype=torch.uint8, device=device
        )
        self.layer.kv_cache = self.mla_cache
        self.ind.k_cache.kv_cache = self.index_cache
        self.fill_caches(seed=7)

        self.mla_builder = ROCMAiterMLASparseMetadataBuilder(
            spec, [self.layer.layer_name], config, device
        )
        self.index_builder = KimiK3IndexerMetadataBuilder(
            ispec,
            [self.ind.k_cache.prefix],
            config,
            device,
            block_table_width=get_block_table_width(
                ispec.max_num_blocks_per_req(config, MAX_MODEL_LEN), PAGE
            ),
        )
        self.bt_width = get_block_table_width(
            ispec.max_num_blocks_per_req(config, MAX_MODEL_LEN), PAGE
        )
        # Pre-gate attention of the last call (forward hook).
        self.pre_gate = None
        self.layer.register_forward_hook(self._hook)
        # merge_dcp_scores recorder: outputs, the stream it ran on, and the
        # inputs (cloned on that stream, so exactly what the merge read).
        from vllm.v1.attention.backends.mla import rocm_aiter_mla_sparse as be

        self.be = be
        self.orig_merge = be.merge_dcp_scores
        self.merges: list = []
        self.record_merges = False
        self.main_stream = None

        def merge(*args, **kwargs):
            out = self.orig_merge(*args, **kwargs)
            if self.record_merges:
                aux = torch.cuda.current_stream() != self.main_stream
                ins = tuple(a.clone() if torch.is_tensor(a) else a for a in args)
                self.merges.append((out[0].clone(), out[1].clone(), aux, ins))
            return out

        be.merge_dcp_scores = merge

    def _hook(self, mod, args, out):
        if torch.cuda.is_current_stream_capturing():
            self.cap_pre = out
        else:
            self.pre_gate = out.clone()

    def fill_caches(self, seed):
        from vllm.models.kimi_k3.amd.ops.fp4_indexer import store_keys

        g = torch.Generator(device=self.device).manual_seed(seed * 100 + self.rank)
        n = NUM_BLOCKS * PAGE
        chunk = 16384
        hf = self.hf
        for s in range(0, n, chunk):
            e = min(n, s + chunk)
            slots = torch.arange(s, e, device=self.device, dtype=torch.int64)
            kv_c = torch.randn(
                (e - s, hf.kv_lora_rank), generator=g, device=self.device
            ).bfloat16()
            k_pe = torch.randn(
                (e - s, 1, hf.qk_rope_head_dim), generator=g, device=self.device
            ).bfloat16()
            self.layer.impl.do_kv_cache_update(
                kv_c, k_pe, self.mla_cache, slots, "fp8_ds_mla", self.layer._k_scale
            )
            keys = torch.randn(
                (e - s, hf.dsa_config.index_head_dim), generator=g, device=self.device
            ).bfloat16()
            store_keys(keys, self.index_cache, slots)
        torch.accelerator.synchronize()

    def metadata(self, lens, pad_reqs=0):
        """lens: [(context, new)] of real requests; pad_reqs trailing padded
        requests (query 0, seq 0), as the runner pads a FULL graph batch."""
        from vllm.forward_context import create_forward_context
        from vllm.v1.attention.backend import CommonAttentionMetadata
        from vllm.v1.attention.backends.utils import get_dcp_local_seq_lens

        new = [n for _, n in lens] + [0] * pad_reqs
        seq = [c + n for c, n in lens] + [0] * pad_reqs
        nreq = len(new)
        qsl = torch.tensor(np.r_[0, np.cumsum(new)], dtype=torch.int32)
        seq_lens = torch.tensor(seq, dtype=torch.int32)
        num_tokens = int(qsl[-1])
        local = get_dcp_local_seq_lens(seq_lens, self.world, self.rank)
        rng = np.random.default_rng(11 + self.rank)
        perm = rng.permutation(NUM_BLOCKS - 1) + 1
        bt = torch.zeros((nreq, self.bt_width), dtype=torch.int32)
        used = 0
        for r in range(nreq):
            nb = -(-int(local[r]) // PAGE)
            assert used + nb <= NUM_BLOCKS - 1
            bt[r, :nb] = torch.from_numpy(perm[used : used + nb].astype(np.int32))
            used += nb
        cm = CommonAttentionMetadata(
            query_start_loc=qsl.to(self.device),
            query_start_loc_cpu=qsl,
            seq_lens=seq_lens.to(self.device),
            num_reqs=nreq,
            num_actual_tokens=num_tokens,
            max_query_len=max(new),
            max_seq_len=max(seq),
            block_table_tensor=bt.to(self.device),
            # The step's own MLA cache rows are not written (not under test);
            # the indexer derives its slots from the block table.
            slot_mapping=torch.full(
                (num_tokens,), -1, dtype=torch.int64, device=self.device
            ),
            seq_lens_cpu_upper_bound=seq_lens,
            dcp_local_seq_lens=local.to(self.device),
            dcp_local_seq_lens_cpu_upper_bound=local,
        )
        md = self.mla_builder.build(0, cm)
        imd = self.index_builder.build(0, cm)
        return (
            create_forward_context(
                {self.layer.layer_name: md, self.ind.k_cache.prefix: imd}, self.config
            ),
            md,
            imd,
            num_tokens,
        )

    def run(self, ctx, x, new: bool):
        from vllm.forward_context import override_forward_context
        from vllm.model_executor.layers.mla import MultiHeadLatentAttentionWrapper

        pos = torch.zeros(x.shape[0], dtype=torch.int64, device=self.device)
        self.main_stream = torch.cuda.current_stream()
        with override_forward_context(ctx):
            if new:
                return self.wrapper(pos, x)
            return MultiHeadLatentAttentionWrapper.forward(self.wrapper, pos, x)

    def reference(self, pre_gate, x):
        """fp32 gate * attention -> o_proj -> TP all-reduce of one path's own
        pre-gate attention (checks the gate/join/o_proj stage of that path)."""
        gate = torch.sigmoid(x.float() @ self.w_g.t())
        ref = (pre_gate.float() * gate) @ self.w_o.t()
        dist.all_reduce(ref)
        return ref


def rel(a, b):
    return float((a.float() - b.float()).norm() / b.float().norm().clamp_min(1e-30))


def csr_equal(s1, i1, s2, i2):
    """CSR equality over the valid prefix only: ``packed`` is allocated with
    rows * topk entries and only ``indptr[-1]`` of them are written."""
    if not torch.equal(i1, i2):
        return False
    nnz = int(i1[-1]) if i1.numel() else 0
    return torch.equal(s1[:nnz], s2[:nnz])


def merge_selfcheck(h, rec):
    """Re-run the merge on main from the recorded inputs: equal output means
    the merge saw consistent inputs and was not raced."""
    slots, indptr, _, ins = rec
    s2, i2 = h.orig_merge(*ins)
    torch.accelerator.synchronize()
    return csr_equal(slots, indptr, s2, i2)


def numerics(h, rank, name, lens, pad_reqs=0, seed=0):
    """Eager old (serial) vs new (multi-stream) on one decode step.

    Bitwise old == new is only meaningful when the serial path itself is
    run-to-run deterministic (``DET``); the default ROCm skinny GEMM
    (wvSplitKrc) is not, and any 1-ulp change of q_c flips top-k selections.
    """
    ctx, md, imd, n = h.metadata(lens, pad_reqs)
    T = max(n, 1)
    g = torch.Generator(device=h.device).manual_seed(1000 + seed)
    x = torch.randn((T, h.hf.hidden_size), generator=g, device=h.device).bfloat16()
    h.merges.clear()
    h.record_merges = True
    out_old = h.run(ctx, x, new=False).clone()
    pre_old = h.pre_gate
    out_old2 = h.run(ctx, x, new=False).clone()
    torch.accelerator.synchronize()
    det = torch.equal(out_old[:n], out_old2[:n])
    m_old = list(h.merges)
    h.merges.clear()
    out_new = h.run(ctx, x, new=True).clone()
    pre_new = h.pre_gate
    torch.accelerator.synchronize()
    m_new = list(h.merges)
    h.record_merges = False
    check(
        rank,
        len(m_new) == 1 and m_new[0][2],
        f"{name}: multi-stream merge did not run on aux",
    )
    check(
        rank,
        len(m_old) == 2 and not m_old[0][2],
        f"{name}: serial merge missing/not on main",
    )
    sc = bool(m_new) and merge_selfcheck(h, m_new[0])
    check(rank, sc, f"{name}: aux merge != merge recomputed from its inputs")
    sc_old = bool(m_old) and merge_selfcheck(h, m_old[0])
    check(rank, sc_old, f"{name}: serial merge != merge recomputed from its inputs")
    bit = None
    e_emu = 0.0
    if DET:
        # Gathered scores and the merged CSR slots must be bitwise equal; the
        # pre-gate attention too unless the serial path itself is not
        # reproducible for this case (then only within its own noise).
        sel = csr_equal(
            m_old[0][0], m_old[0][1], m_new[0][0], m_new[0][1]
        ) and torch.equal(m_old[0][3][0], m_new[0][3][0])
        if det:
            check(
                rank,
                sel,
                f"{name}: gathered scores / merged slots not bitwise old == new",
            )
        else:
            # The serial path itself is not reproducible here: compare the
            # gathered scores of new vs serial with serial run 1 vs run 2.
            s_oo = float((m_old[0][3][0] - m_old[1][3][0]).abs().max())
            s_on = float((m_new[0][3][0] - m_old[0][3][0]).abs().max())
            print(
                f"[rank {rank}] {name}: serial not reproducible; gathered-score "
                f"max|diff| old-vs-old={s_oo:.3e} new-vs-old={s_on:.3e}",
                flush=True,
            )
            check(
                rank,
                s_on <= 3 * s_oo + 1e-6,
                f"{name}: scores new-vs-old {s_on} >> {s_oo}",
            )
        pre_eq = torch.equal(pre_old[:n], pre_new[:n])
        if det:
            check(rank, pre_eq, f"{name}: pre-gate attention not bitwise old == new")
        # The join multiplies in fp32 (one rounding, like the fused compiled
        # gate); the eager serial path rounds sigmoid to bf16 first. Emulate
        # the join on the serial pre-gate output: must match the new output.
        with torch.no_grad():
            gl = h.attn.g_proj(x)[0]
            y = (pre_old.float() * gl.float().sigmoid()).to(pre_old.dtype)
            emu = h.attn.o_proj(y)[0]
        torch.accelerator.synchronize()
        e_emu = rel(out_new[:n], emu[:n])
        if det:
            check(
                rank,
                e_emu < 1e-4,
                f"{name}: new != serial pre-gate + fp32 gate ({e_emu})",
            )
        bit = sel and pre_eq
    ref_old = h.reference(pre_old[:n], x[:n])
    ref_new = h.reference(pre_new[:n], x[:n])
    e_old = rel(out_old[:n], ref_old)
    e_new = rel(out_new[:n], ref_new)
    e_on = rel(out_new[:n], out_old[:n])
    e_oo = rel(out_old2[:n], out_old[:n])
    check(rank, bool(torch.isfinite(out_new).all()), f"{name}: non-finite output")
    check(
        rank, e_new < 1e-2 and e_new <= e_old * 1.5 + 1e-4, f"{name}: relL2 new {e_new}"
    )
    if DET:
        check(rank, e_on <= 1e-2, f"{name}: new vs old relL2 {e_on}")
    else:
        # Both are noisy samples of the same computation.
        check(
            rank,
            e_on <= max(3 * e_oo, 1e-2),
            f"{name}: new-vs-old {e_on} >> old-vs-old {e_oo}",
        )
    nnz = int(m_new[0][1][-1]) if m_new else -1
    empty_rows = int(((m_new[0][1][1:] - m_new[0][1][:-1]) == 0).sum()) if m_new else -1
    stats = torch.tensor([e_old, e_new, e_on, e_oo, e_emu], device=h.device)
    dist.all_reduce(stats, op=dist.ReduceOp.MAX)
    cnt = torch.tensor([nnz, empty_rows], device=h.device, dtype=torch.int64)
    allc = [torch.zeros_like(cnt) for _ in range(h.world)]
    dist.all_gather(allc, cnt)
    flags = torch.tensor(
        [int(det), int(sc and sc_old), int(bool(bit))], device=h.device
    )
    dist.all_reduce(flags, op=dist.ReduceOp.MIN)
    log(
        rank,
        f"[num] {name}: rows={n} relL2 vs own fp32: old={stats[0]:.3e} new={stats[1]:.3e}; "
        f"new-vs-old={stats[2]:.3e} old-vs-old={stats[3]:.3e}; serial deterministic="
        f"{bool(flags[0])} merge selfcheck(old,new)={bool(flags[1])} bitwise old==new="
        f"{bool(flags[2]) if DET else 'n/a'} (scores+slots+pre-gate); new vs serial+fp32-gate="
        f"{stats[4]:.1e}; merge (nnz, empty local rows)/rank="
        f"{[tuple(c.tolist()) for c in allc]}",
    )


GRAPH_CASES = {
    "T64": (
        [(int(c), 4) for c in np.random.default_rng(1).integers(2000, 9000, 16)],
        0,
        64,
    ),
    "T100": (
        [(int(c), 4) for c in np.random.default_rng(2).integers(2000, 9000, 25)],
        0,
        100,
    ),
    "T125": (
        [(int(c), 5) for c in np.random.default_rng(3).integers(2000, 9000, 25)],
        0,
        125,
    ),
    # 23 real requests + 2 padded (query 0, seq 0); graph rows = 100.
    "T100pad": (
        [(int(c), 4) for c in np.random.default_rng(4).integers(2000, 9000, 23)],
        2,
        100,
    ),
}


def graph_case(h, rank, name, bench, iters):
    """Capture old and new FULL graphs of one case (inside graph_capture(),
    so the aux comm registers its captured buffers), replay them with new
    hidden states and cache contents, compare with eager; optional timing.

    Metadata is built right before capture and kept for the replays: the
    builders write into persistent buffers, so another case's build would
    change what these graphs read.
    """
    from vllm.distributed.parallel_state import graph_capture

    lens, pad, T = GRAPH_CASES[name]
    ctx, md, imd, n = h.metadata(lens, pad)
    x = torch.zeros((T, h.hf.hidden_size), dtype=torch.bfloat16, device=h.device)
    graphs, outs = {}, {}
    pool = torch.cuda.graph_pool_handle()
    with graph_capture(h.device) as gctx:
        for new in (False, True):
            x.normal_()
            with torch.cuda.stream(gctx.stream):
                for _ in range(2):
                    h.run(ctx, x, new)
            torch.accelerator.synchronize()
            gr = torch.cuda.CUDAGraph()
            with torch.cuda.graph(gr, pool=pool, stream=gctx.stream):
                o = h.run(ctx, x, new)
            graphs[new], outs[new] = gr, o
    torch.accelerator.synchronize()
    dist.barrier()
    for step in range(2):
        h.fill_caches(seed=20 + step)
        g = torch.Generator(device=h.device).manual_seed(500 + step)
        x.copy_(torch.randn(x.shape, generator=g, device=h.device).bfloat16())
        eager_old = h.run(ctx, x, new=False).clone()
        pre_old = h.pre_gate.clone()
        eager_old2 = h.run(ctx, x, new=False).clone()
        eager_new = h.run(ctx, x, new=True).clone()
        pre_new = h.pre_gate.clone()
        torch.accelerator.synchronize()
        res = {}
        for new in (False, True):
            graphs[new].replay()
            torch.accelerator.synchronize()
            res[new] = outs[new][:n].clone()
        # Replay twice more: identical results when deterministic.
        graphs[True].replay()
        torch.accelerator.synchronize()
        rep2 = outs[True][:n].clone()
        e = torch.tensor(
            [
                rel(res[False], eager_old[:n]),
                rel(res[True], eager_new[:n]),
                rel(res[True], res[False]),
                rel(eager_old2[:n], eager_old[:n]),
                rel(eager_new[:n], h.reference(pre_new[:n], x[:n])),
                rel(eager_old[:n], h.reference(pre_old[:n], x[:n])),
                rel(rep2, res[True]),
            ],
            device=h.device,
        )
        check(rank, bool(torch.isfinite(res[True]).all()), f"graph {name}: non-finite")
        if DET:
            check(
                rank,
                float(e[0]) == 0.0,
                f"graph {name}: old graph != old eager ({e[0]})",
            )
            check(
                rank,
                float(e[1]) == 0.0,
                f"graph {name}: new graph != new eager ({e[1]})",
            )
            # new vs old differs only by the fp32 gate rounding (see numerics).
            check(
                rank, float(e[2]) < 1e-2, f"graph {name}: new graph vs old graph {e[2]}"
            )
            check(
                rank,
                float(e[6]) == 0.0,
                f"graph {name}: new graph replays differ ({e[6]})",
            )
        else:
            tol = max(3 * float(e[3]), 1e-2)
            for i, what in (
                (0, "old graph vs old eager"),
                (1, "new graph vs new eager"),
                (2, "new graph vs old graph"),
            ):
                check(rank, float(e[i]) <= tol, f"graph {name}: {what} {e[i]} > {tol}")
        check(rank, float(e[4]) < 1e-2, f"graph {name}: new eager vs fp32 {e[4]}")
        dist.all_reduce(e, op=dist.ReduceOp.MAX)
        log(
            rank,
            f"[graph] step{step} {name}: rows={n}/{T} relL2 graph-vs-eager old={e[0]:.2e} "
            f"new={e[1]:.2e}; new-vs-old graph={e[2]:.2e}; eager old-vs-old={e[3]:.2e}; "
            f"new replay-vs-replay={e[6]:.2e}; eager vs own fp32 new={e[4]:.3e} old={e[5]:.3e}",
        )
    result = None
    if bench and name in ("T64", "T100", "T125"):
        times = {False: [], True: []}
        for new in (False, True):
            for _ in range(20):
                graphs[new].replay()
        torch.accelerator.synchronize()
        dist.barrier()
        for _ in range(iters):
            for new in (False, True):
                s = torch.cuda.Event(enable_timing=True)
                e2 = torch.cuda.Event(enable_timing=True)
                s.record()
                for _ in range(10):
                    graphs[new].replay()
                e2.record()
                e2.synchronize()
                times[new].append(s.elapsed_time(e2) * 100.0)  # us per replay
        med = torch.tensor(
            [float(np.median(times[False])), float(np.median(times[True]))],
            device=h.device,
        )
        dist.all_reduce(med, op=dist.ReduceOp.MAX)
        result = (float(med[0]), float(med[1]))
        log(
            rank,
            f"[bench] {name}: old {med[0]:.1f} us  new {med[1]:.1f} us  "
            f"delta {med[1] - med[0]:+.1f} us ({(med[1] / med[0] - 1) * 100:+.1f}%)",
        )
    del graphs, outs, gr, o
    torch.accelerator.synchronize()
    dist.barrier()
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", action="store_true")
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--det", action="store_true")
    ap.add_argument("--no-graph", action="store_true")
    args = ap.parse_args()
    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    torch.accelerator.set_device_index(rank)
    device = torch.device("cuda", rank)
    from vllm.config import set_current_vllm_config
    from vllm.distributed import (
        cleanup_dist_env_and_memory,
        init_distributed_environment,
        initialize_model_parallel,
    )
    from vllm.v1.worker.workspace import init_workspace_manager

    t0 = time.time()
    config = make_config()
    log(
        rank,
        f"[cfg] cache={config.cache_config.cache_dtype} "
        f"indexer={config.attention_config.indexer_kv_dtype} "
        f"nst={config.speculative_config.num_speculative_tokens} "
        f"dcp={config.parallel_config.decode_context_parallel_size} "
        f"skinny_gemm={os.environ.get('VLLM_ROCM_USE_SKINNY_GEMM', '1')} DET={DET}",
    )
    with set_current_vllm_config(config):
        init_distributed_environment(
            world_size=world,
            rank=rank,
            local_rank=rank,
            distributed_init_method="env://",
        )
        initialize_model_parallel(
            tensor_model_parallel_size=world, decode_context_model_parallel_size=world
        )
        init_workspace_manager(device)
        try:
            h = Harness(config, rank, world, device)
            if DET:
                from vllm.model_executor.determinism.batch_invariant import (
                    enable_batch_invariant_mode,
                )

                enable_batch_invariant_mode()
            log(rank, f"[setup] {time.time() - t0:.1f}s")
            rng = np.random.default_rng(5)
            numerics(
                h,
                rank,
                "T100 (25x4)",
                [(int(c), 4) for c in rng.integers(1000, 9000, 25)],
                seed=1,
            )
            numerics(
                h,
                rank,
                "T125 (25x5)",
                [(int(c), 5) for c in rng.integers(1000, 9000, 25)],
                seed=2,
            )
            numerics(
                h,
                rank,
                "T64 (16x4)",
                [(int(c), 4) for c in rng.integers(1000, 9000, 16)],
                seed=3,
            )
            numerics(
                h,
                rank,
                "long ctx (8x4, ~20k)",
                [(int(c), 4) for c in rng.integers(16000, 24000, 8)],
                seed=4,
            )
            # Tiny sequences: ranks >= seq_len hold no keys of a request
            # (empty shards; their local top-k rows are all -1).
            numerics(
                h,
                rank,
                "empty shards (tiny seqs)",
                [(0, 4), (1, 4), (2, 4), (3, 4), (5, 4), (8, 4), (13, 4), (100, 4)],
                seed=5,
            )
            numerics(
                h, rank, "all tiny (seq<=4)", [(0, 1), (0, 2), (1, 3), (0, 4)], seed=6
            )
            numerics(
                h,
                rank,
                "mixed query lens",
                [(0, 1), (3, 2), (4000, 5), (10, 3), (2047, 4), (2048, 1)],
                seed=7,
            )
            numerics(
                h,
                rank,
                "padded reqs (23x4 + 2 pad)",
                [(int(c), 4) for c in rng.integers(1000, 9000, 23)],
                pad_reqs=2,
                seed=8,
            )
            numerics(h, rank, "single row", [(5000, 1)], seed=9)
            bench = {}
            if not args.no_graph:
                for name in GRAPH_CASES:
                    r = graph_case(h, rank, name, args.bench, args.iters)
                    if r is not None:
                        bench[name] = r
            if rank == 0 and bench:
                print("[bench-json]", json.dumps(bench), flush=True)
        finally:
            fails = torch.tensor([len(FAILS)], device=device)
            dist.all_reduce(fails)
            log(rank, "RESULT:", "PASS" if int(fails) == 0 else f"FAIL ({int(fails)})")
            cleanup_dist_env_and_memory()
    sys.exit(0 if not FAILS else 1)


if __name__ == "__main__":
    main()
