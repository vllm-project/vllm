# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""amd_kernels on one real Kimi-K3.1 DSA layer, 8 GPUs (TP8/DCP8).

torchrun --nproc-per-node 8 tests/kimi_dsams/run_dsa_akr.py \
    [--bench] [--no-graph] [--only a,b]

Builds two real ROCm ``KimiK3DSAAttention`` layers (Kimi-K3.1-Pro dims, random bf16
weights, fp8_ds_mla MLA cache, MXFP4 indexer cache): ``akr`` (all VLLM_KIMI_AKR_<NAME>
switches on, extended qkv_a weight, packed q_b / replicated W_UK / transposed W_UV) and
``stock`` (all off). Stage tests (same inputs through the integrated and the stock op
sequence, real weights and caches): prologue, indexer (fused score+top-k), score
gather+merge, q gather, sparse MLA, combine (+ gate). End-to-end: the multi-stream
wrapper of both layers, eager and FULL graph (padded rows), per-switch bisection, and
--bench: layer graph replay time stock vs integrated.
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
os.environ.setdefault("VLLM_ROCM_DCP_VUP_RS", "1")
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

# Hugging Face ids or local paths (only the configs are read: load_format="dummy").
MODEL = os.environ.get("KIMI_K3_MODEL", "moonshotai/Kimi-K3.1-Pro")
DRAFT = os.environ.get("KIMI_K3_DRAFT", "Inferact/Kimi-K3.1-Pro-DSpark-preview")
NST = 4
PAGE = 64
NUM_BLOCKS = 1536
MAX_MODEL_LEN = 1 << 20
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
        max_num_seqs=48,
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


SWITCHES = (
    "PROLOGUE",
    "SCORE_TOPK",
    "GATHER_MERGE",
    "QGATHER",
    "SPARSE_MLA",
    "COMBINE",
)


class akr_env:
    """Turn the per-kernel switches on (`on`: all, or a set of names) or off."""

    def __init__(self, on):
        self.on = set(SWITCHES) if on is True else set(on or ())

    def __enter__(self):
        self.old = {n: os.environ.get(f"VLLM_KIMI_AKR_{n}") for n in SWITCHES}
        for n in SWITCHES:
            os.environ[f"VLLM_KIMI_AKR_{n}"] = "1" if n in self.on else "0"

    def __exit__(self, *a):
        for n, v in self.old.items():
            if v is None:
                os.environ.pop(f"VLLM_KIMI_AKR_{n}", None)
            else:
                os.environ[f"VLLM_KIMI_AKR_{n}"] = v


class Harness:
    def __init__(self, config, rank, world, device, prefix, akr):
        from vllm.models.kimi_k3.amd import dsa as kdsa
        from vllm.models.kimi_k3.common.dsa_cache import KimiK3IndexerMetadataBuilder
        from vllm.v1.attention.backends.mla.rocm_aiter_mla_sparse import (
            ROCMAiterMLASparseMetadataBuilder,
        )
        from vllm.v1.worker.block_table import get_block_table_width

        self.config, self.rank, self.world, self.device = config, rank, world, device
        self.akr = akr
        hf = config.model_config.hf_text_config
        self.hf = hf
        self.topk_buf = torch.empty(
            (config.scheduler_config.max_num_batched_tokens, hf.dsa_config.index_topk),
            dtype=torch.int32,
            device=device,
        )
        torch.set_default_dtype(torch.bfloat16)
        with akr_env(akr), torch.device(device):
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
                prefix=prefix,
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
        assert self.wrapper.akr_pro == akr and self.wrapper.akr_qg == akr, (
            self.wrapper.akr_pro,
            self.wrapper.akr_qg,
        )
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

        before = torch.accelerator.memory_allocated(device)
        with akr_env(akr):
            process_weights_after_loading(attn, config.model_config, device)
        self.post_load_bytes = torch.accelerator.memory_allocated(device) - before

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
        self.pre_gate = None
        self.layer.register_forward_hook(self._hook)

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
        requests (query 0, seq 0), as the runner pads a FULL graph batch. The
        MLA slot mapping is real (this rank's owned tokens), as in the server."""
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
        slots = np.full(num_tokens, -1, dtype=np.int64)
        t = 0
        for r in range(nreq):
            for j in range(new[r]):
                p = seq[r] - new[r] + j
                if p % self.world == self.rank:
                    li = p // self.world
                    slots[t] = int(bt[r, li // PAGE]) * PAGE + li % PAGE
                t += 1
        slots = torch.from_numpy(slots).to(self.device)
        cm = CommonAttentionMetadata(
            query_start_loc=qsl.to(self.device),
            query_start_loc_cpu=qsl,
            seq_lens=seq_lens.to(self.device),
            num_reqs=nreq,
            num_actual_tokens=num_tokens,
            max_query_len=max(new),
            max_seq_len=max(seq),
            block_table_tensor=bt.to(self.device),
            slot_mapping=slots,
            seq_lens_cpu_upper_bound=seq_lens,
            dcp_local_seq_lens=local.to(self.device),
            dcp_local_seq_lens_cpu_upper_bound=local,
        )
        md = self.mla_builder.build(0, cm)
        imd = self.index_builder.build(0, cm)
        return (
            create_forward_context(
                {self.layer.layer_name: md, self.ind.k_cache.prefix: imd},
                self.config,
                slot_mapping={self.layer.layer_name: slots},
            ),
            md,
            imd,
            num_tokens,
        )

    def run(self, ctx, x, new: bool = True):
        from vllm.forward_context import override_forward_context
        from vllm.model_executor.layers.mla import MultiHeadLatentAttentionWrapper

        pos = torch.zeros(x.shape[0], dtype=torch.int64, device=self.device)
        with override_forward_context(ctx):
            if new:
                return self.wrapper(pos, x)
            return MultiHeadLatentAttentionWrapper.forward(self.wrapper, pos, x)


def rel(a, b):
    return float((a.float() - b.float()).norm() / b.float().norm().clamp_min(1e-30))


def csr_equal(s1, i1, s2, i2):
    if not torch.equal(i1, i2):
        return False
    nnz = int(i1[-1]) if i1.numel() else 0
    return torch.equal(s1[:nnz], s2[:nnz])


def agree(rank, ok):
    """True when `ok` holds on every rank."""
    t = torch.tensor([int(bool(ok))], device="cuda")
    dist.all_reduce(t, op=dist.ReduceOp.MIN)
    return bool(t[0])


def stat(vals, device):
    """Max over ranks of a list of floats."""
    t = torch.tensor(vals, device=device, dtype=torch.float64)
    dist.all_reduce(t, op=dist.ReduceOp.MAX)
    return t.tolist()


# --------------------------------------------------------------------------
# stage tests
# --------------------------------------------------------------------------


def stage_prologue(ha, hs, rank, M):
    """K_A: norms + fp8_ds_mla KV insert + indexer key store + head weights vs the stock
    ops."""
    from vllm.models.kimi_k3.amd import akr_dsa as A
    from vllm.models.kimi_k3.amd.ops.fp4_indexer import store_keys

    dev = ha.device
    g = torch.Generator(device=dev).manual_seed(77 + M)
    qkv = (torch.randn((M, 2272), generator=g, device=dev) * 2).bfloat16()
    pages = 8
    slots = torch.randperm(pages * PAGE, device=dev, generator=g)[:M].to(torch.int64)
    slots[::5] = -1
    islots = slots.clone()
    islots[1::7] = -1
    kvc = ha.mla_cache[:pages].clone()
    indc = ha.index_cache[:pages].clone()
    kvr = kvc.clone()
    indr = indc.clone()
    q_c, kv_c, idx_w = A.prologue_a(ha.wrapper, qkv, kvc, slots, indc, islots)
    # stock: the AITER rms_norm contract (fp32 statistics, one bf16 rounding)
    w = hs.wrapper

    def rms(x, norm):
        xf = x.float()
        inv = torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + norm.variance_epsilon)
        return (xf * inv * norm.weight.float()).bfloat16()

    q_ref = rms(qkv[:, :1536], w.q_a_layernorm)
    kv_ref = rms(qkv[:, 1536:2048], w.kv_a_layernorm)
    k_pe = qkv[:, 2048:2112].unsqueeze(1)
    hs.layer.impl.do_kv_cache_update(
        kv_ref, k_pe, kvr, slots, "fp8_ds_mla", hs.layer._k_scale
    )
    key = hs.ind.k_index_norm(qkv[:, 2112:2240])
    store_keys(key[islots >= 0], indr, islots[islots >= 0])
    w_ref = qkv[:, 2240:2272].float() * (
        hs.ind.indexer_weight_scale * hs.ind.index_softmax_scale
    )
    torch.accelerator.synchronize()
    eq = lambda a, b: float((a == b).float().mean())  # noqa: E731
    r = stat(
        [
            1 - eq(q_c, q_ref),
            1 - eq(kv_c, kv_ref),
            1 - eq(kvc, kvr),
            1 - eq(indc, indr),
            float((idx_w - w_ref).abs().max()),
            rel(q_c, q_ref),
            rel(kv_c, kv_ref),
        ],
        dev,
    )
    ok = r[0] < 2e-3 and r[1] < 2e-3 and r[2] < 2e-3 and r[3] < 2e-3 and r[4] == 0
    check(rank, ok, f"prologue M={M}: mismatch {r}")
    log(
        rank,
        f"[stage] prologue M={M}: frac differing q={r[0]:.1e} kv={r[1]:.1e} "
        f"kv-cache={r[2]:.1e} "
        f"index-cache={r[3]:.1e}; max|w diff|={r[4]:.1e}; relL2 q={r[5]:.1e} "
        f"kv={r[6]:.1e}",
    )


def stage_indexer(ha, hs, rank, lens, pad, tag):
    """Fused score+top-k (+ prologue K_B) vs score_paged + select, same inputs, through
    _fp4_indexer."""
    from vllm.forward_context import override_forward_context
    from vllm.models.kimi_k3.amd import akr_dsa as A
    from vllm.models.kimi_k3.amd import dsa as kdsa

    dev = ha.device
    ctx, md, imd, n = ha.metadata(lens, pad)
    T = max(n, 1) + pad * 0
    g = torch.Generator(device=dev).manual_seed(900 + T)
    ind = ha.ind
    qidx = (torch.randn((T, 32, 128), generator=g, device=dev)).bfloat16()
    wv = torch.rand((T, 32), generator=g, device=dev).bfloat16()
    kraw = torch.randn((T, 128), generator=g, device=dev).bfloat16()
    dcp = kdsa.get_dcp_group()
    scale = ind.indexer_weight_scale * ind.index_softmax_scale
    topk = torch.full((T, 2048), 9, dtype=torch.int32, device=dev)
    scores = torch.zeros((T, 2048), dtype=torch.float32, device=dev)
    res = {}
    for name, akr_on in (("stock", False), ("akr", True)):
        out = topk.clone()
        sc = scores.clone()
        with akr_env(akr_on), override_forward_context(ctx):
            kdsa._fp4_indexer(
                qidx,
                ind.k_index_norm(kraw),
                wv,
                ind.k_cache.kv_cache,
                out,
                ind.k_cache.prefix,
                ind.max_model_len,
                dcp.rank_in_group,
                dcp.world_size,
                ind.cp_interleave,
                scale,
                sc,
            )
        torch.accelerator.synchronize()
        res[name] = (out, sc)
    (so, ss), (ao, as_) = res["stock"], res["akr"]

    # same selected key SET per row (ties at the k-th score may pick different keys:
    # compare the scores of the selected keys instead), scores of selected keys equal
    def selected_scores(o, s):
        v = torch.where(o >= 0, s, torch.full_like(s, float("-inf")))
        return v.sort(dim=1).values

    sv, av = selected_scores(so, ss), selected_scores(ao, as_)
    cnt_eq = bool(((so >= 0).sum(1) == (ao >= 0).sum(1)).all())
    val_eq = bool(torch.equal(sv, av))
    set_eq = float((so.sort(dim=1).values == ao.sort(dim=1).values).float().mean())
    check(
        rank,
        cnt_eq and val_eq,
        f"indexer {tag}: fused score+top-k differs (count {cnt_eq} values {val_eq})",
    )
    ok = agree(rank, cnt_eq and val_eq)
    # prepped (K_B) path
    with override_forward_context(ctx):
        qp, qs = A.prologue_b(ha.wrapper, qidx, ha.topk_buf, ind.k_cache.kv_cache)
    from vllm.models.kimi_k3.amd.ops.fp4_indexer import quantize_queries

    rp, rs = quantize_queries(qidx)
    pb = (
        torch.equal(qp, rp)
        and torch.equal(qs, rs)
        and bool((ha.topk_buf[:T] == -1).all())
    )
    check(
        rank,
        pb,
        f"indexer {tag}: K_B payload/scales/reset != quantize_queries / fill(-1)",
    )
    log(
        rank,
        f"[stage] indexer {tag}: rows={n}: counts equal={ok} selected-score values "
        f"equal={val_eq} "
        f"index sets equal (frac)={set_eq:.5f}; K_B bit-equal={pb}",
    )


def stage_gather_merge(ha, hs, rank, rows, kind="random"):
    """Score gather + merge vs all_gather + merge_dcp_scores (bitwise CSR)."""
    from vllm.distributed import get_dcp_group
    from vllm.models.kimi_k3.amd import akr_dsa as A
    from vllm.v1.attention.backends.mla import rocm_aiter_mla_sparse as be

    dev = ha.device
    g = torch.Generator(device=dev).manual_seed(31 * rows + rank)
    K = 2048
    width = 20000
    live = torch.randint(0, K + 1, (rows,), device=dev, generator=g)
    if kind == "dense":
        live[:] = K
    if kind == "skew" and rank % 2:
        live[:] = 0
    scores = torch.randn((rows, K), device=dev, generator=g)
    if kind == "ties":
        scores = (scores * 4).round() / 4
    cols = torch.arange(K, device=dev)[None]
    mask = cols < live[:, None]
    scores = torch.where(mask, scores, torch.full_like(scores, float("-inf")))
    idx = torch.stack(
        [torch.randperm(width, device=dev, generator=g)[:K] for _ in range(rows)]
    ).to(torch.int32)
    idx = torch.where(mask, idx, torch.full_like(idx, -1))
    blocks = -(-width // 64) + 1
    pages = torch.randint(0, NUM_BLOCKS, (rows, blocks), device=dev, generator=g).to(
        torch.int32
    )
    comm_grp = get_dcp_group()
    gathered = comm_grp.all_gather(scores.view(torch.int32), dim=1).view(torch.float32)
    s_ref, i_ref = be.merge_dcp_scores(gathered, idx, pages, rank, 8, 64, 64)
    with akr_env(True):
        merged = A.gather_merge(scores, idx, pages, 64, 64)
        assert merged is not None
        s_new, i_new = merged
    torch.accelerator.synchronize()
    ok = csr_equal(s_new, i_new, s_ref, i_ref)
    check(
        rank, ok, f"gather_merge {kind} rows={rows}: CSR differs from merge_dcp_scores"
    )
    ok = agree(rank, ok)
    log(
        rank,
        f"[stage] gather_merge {kind} rows={rows}: CSR bitwise equal on all ranks = "
        f"{ok}",
    )


def stage_qgather(ha, hs, rank, M):
    from vllm.distributed import get_dcp_group
    from vllm.models.kimi_k3.amd import akr_dsa as A

    dev = ha.device
    g = torch.Generator(device=dev).manual_seed(5 + M)
    q_c = (torch.randn((M, 1536), generator=g, device=dev) * 1.5).bfloat16()
    w = hs.wrapper
    with torch.no_grad():
        q = w.q_b_proj(q_c)[0].view(M, 12, 192)
        nope, pe = q[..., :128], q[..., 128:]
        lat = torch.bmm(nope.transpose(0, 1), hs.layer.W_UK_T).transpose(0, 1)
        qs = torch.cat((lat, pe), dim=-1)
    ref = get_dcp_group().all_gather(qs.contiguous(), dim=1)
    with akr_env(True):
        out = A.qgather(ha.wrapper, q_c).clone()
    torch.accelerator.synchronize()
    # fp32 reference chain (the kernel's contract): fp32 GEMMs, one bf16 rounding each
    with torch.no_grad():
        q32 = (q_c.float() @ w.q_b_proj.weight.float().t()).bfloat16().view(M, 12, 192)
        lat32 = (
            torch.bmm(q32[..., :128].transpose(0, 1).float(), hs.layer.W_UK_T.float())
            .bfloat16()
            .transpose(0, 1)
        )
        qs32 = torch.cat((lat32, q32[..., 128:]), dim=-1)
    ref32 = get_dcp_group().all_gather(qs32.contiguous(), dim=1)
    e32, e_stock32, e = stat([rel(out, ref32), rel(ref, ref32), rel(out, ref)], dev)
    pe_ok = float((out[..., 512:] == ref32[..., 512:]).float().mean())
    check(
        rank,
        e32 < 4e-3 and e32 <= e_stock32 * 1.5 + 1e-4,
        f"qgather M={M}: relL2 vs fp32 {e32} (stock {e_stock32})",
    )
    log(
        rank,
        f"[stage] qgather M={M}: relL2 akr-vs-stock={e:.2e}; vs the fp32 chain: "
        f"akr={e32:.2e} stock={e_stock32:.2e}; rope columns equal to it "
        f"(frac)={pe_ok:.4f}",
    )


def stage_sparse_combine(ha, hs, rank, M, reqs_empty=1):
    """Sparse MLA decode kernel vs Triton; combine (+gate) vs the stock v_up-first
    path."""
    from vllm._aiter_ops import rocm_aiter_ops
    from vllm.models.kimi_k3.amd import akr_dsa as A
    from vllm.v1.attention.backends.mla.sparse_utils import flat_kv_row_view

    dev = ha.device
    g = torch.Generator(device=dev).manual_seed(123 + M)
    q = (torch.randn((M, 96, 576), generator=g, device=dev) * 0.7).bfloat16()
    counts = torch.randint(0, 600, (M,), device=dev, generator=g).to(torch.int32)
    counts[::9] = 0
    counts[3::11] = 2048
    indptr = torch.zeros(M + 1, dtype=torch.int32, device=dev)
    indptr[1:] = counts.cumsum(0)
    nnz = int(indptr[-1])
    packed = torch.empty(M * 2048, dtype=torch.int32, device=dev)
    packed[:nnz] = torch.randint(
        0, NUM_BLOCKS * PAGE, (nnz,), device=dev, generator=g
    ).to(torch.int32)
    rows, stride_rows = flat_kv_row_view(ha.mla_cache, PAGE)
    scale = ha.layer.impl.scale
    bands, A.FLYDSL_SPARSE_ROWS = A.FLYDSL_SPARSE_ROWS, ()  # the kernel at every M
    try:
        with akr_env(True):
            res = A.sparse_mla_decode(q, rows, indptr, packed, scale, torch.bfloat16)
    finally:
        A.FLYDSL_SPARSE_ROWS = bands
    assert res is not None
    out, lse = res
    out_t = torch.empty((M, 96, 512), dtype=torch.bfloat16, device=dev)
    lse_t = rocm_aiter_ops.triton_sparse_mla_fwd(
        q,
        rows.view(torch.uint8).unsqueeze(1),
        out_t,
        scale,
        indptr,
        packed,
        kv_lora_rank=512,
        qk_rope_head_dim=64,
        has_invalid=False,
        return_lse=True,
    )
    torch.accelerator.synchronize()
    valid = counts > 0
    e_out = rel(out[valid], out_t[valid])
    e_lse = float((lse[valid] - lse_t[valid]).abs().max())
    check(
        rank,
        e_out < 6e-3 and e_lse < 2e-2,
        f"sparse M={M}: out relL2 {e_out} lse err {e_lse}",
    )
    log(
        rank,
        f"[stage] sparse M={M}: relL2 vs Triton out={e_out:.2e} max|lse "
        f"diff|={e_lse:.1e} "
        f"(empty rows lse akr={float(lse[~valid].max()) if (~valid).any() else 0:.2e})",
    )
    # combine: seq_lens / query_start_loc of 4-row requests, some empty shards and
    # padded tail
    nreq = (M + 3) // 4
    qsl = torch.arange(0, nreq + 1, device=dev, dtype=torch.int32) * 4
    qsl = qsl.clamp(
        max=M - 2 if M > 6 else M
    )  # the last request(s) end before M: padded rows
    seq = torch.randint(1, 5000, (nreq,), device=dev, generator=g).to(torch.int32)
    seq[::5] = 0
    gate = torch.randn((M, 1536), device=dev, generator=g).bfloat16()
    hst = hs.layer
    ref_out = torch.empty((M, 12, 128), dtype=torch.bfloat16, device=dev)
    done = hst.dcp_manager.vup_rs_combine(
        out, lse, hst.W_UV_dcp, ref_out, seq_lens=seq, query_start_loc=qsl
    )
    assert done
    ref_gated = (ref_out.view(M, 1536).float() * gate.float().sigmoid()).to(
        torch.bfloat16
    )
    st = ha.layer._akr_ms_state
    res_plain = torch.empty((M, 12, 128), dtype=torch.bfloat16, device=dev)
    res_gate = torch.empty((M, 12, 128), dtype=torch.bfloat16, device=dev)
    with akr_env(True):
        st.reset_akr()
        ok1 = A.combine(ha.layer, out, lse, res_plain, seq, qsl)
        st.reset_akr()
        st.gate, st.joined = gate, False
        st.ev_done.record()
        ok2 = A.combine(ha.layer, out, lse, res_gate, seq, qsl)
        folded = st.folded
        st.reset_akr()
        st.joined = True
    torch.accelerator.synchronize()
    e1 = rel(res_plain.view(M, 1536), ref_out.view(M, 1536))
    e2 = rel(res_gate.view(M, 1536), ref_gated)
    check(
        rank,
        ok1 and ok2 and folded and e1 < 1e-2 and e2 < 1e-2,
        f"combine M={M}: {ok1} {ok2} {folded} {e1} {e2}",
    )
    r = stat([e1, e2], dev)
    log(
        rank,
        f"[stage] combine M={M}: relL2 vs stock vup_rs plain={r[0]:.2e} "
        f"gated={r[1]:.2e} (gate folded={folded})",
    )


# --------------------------------------------------------------------------
# end to end
# --------------------------------------------------------------------------

CALLS: dict[str, int] = {}
SYNC = os.environ.get("AKR_SYNC") == "1"


def count_calls():
    from vllm.models.kimi_k3.amd import akr_dsa as A

    for name in (
        "prologue_a",
        "prologue_b",
        "qgather",
        "gather_merge",
        "sparse_mla_decode",
        "combine",
        "score_topk",
    ):
        orig = getattr(A, name)

        def f(*a, _o=orig, _n=name, **k):
            if SYNC:
                print(f"[call r{os.environ['RANK']}] {_n} start", flush=True)
            out = _o(*a, **k)
            # gather_merge / sparse_mla_decode return None, combine / score_topk False
            # when the stock path runs
            if out is not None and out is not False:
                CALLS[_n] = CALLS.get(_n, 0) + 1
            if SYNC:
                torch.accelerator.synchronize()
                print(f"[call r{os.environ['RANK']}] {_n} ok", flush=True)
            if _n == "combine" and a[0]._akr_ms_state.folded:
                CALLS["folded"] = CALLS.get("folded", 0) + 1
            return out

        setattr(A, name, f)


def reference_out(h, pre_gate, x):
    gate = torch.sigmoid(x.float() @ h.w_g.t())
    ref = (pre_gate.float() * gate) @ h.w_o.t()
    dist.all_reduce(ref)
    return ref


def e2e(ha, hs, rank, name, lens, pad=0, switches=True, expect=None):
    ctx_a, _, _, n = ha.metadata(lens, pad)
    ctx_s, _, _, _ = hs.metadata(lens, pad)
    T = max(n, 1)
    g = torch.Generator(device=ha.device).manual_seed(4242 + T)
    x = torch.randn((T, ha.hf.hidden_size), generator=g, device=ha.device).bfloat16()
    with akr_env(False):
        s1 = hs.run(ctx_s, x).clone()
        pre_s = hs.pre_gate
        s2 = hs.run(ctx_s, x).clone()
    CALLS.clear()
    with akr_env(switches):
        a1 = ha.run(ctx_a, x).clone()
    pre_a = ha.pre_gate
    torch.accelerator.synchronize()
    calls = dict(CALLS)
    e_noise = rel(s2[:n], s1[:n])
    e_sa = rel(a1[:n], s1[:n])
    if calls.get("folded"):
        # the combine kernel gated the attention output itself: pre_a is gated
        ref_a = pre_a[:n].float() @ ha.w_o.t()
        dist.all_reduce(ref_a)
        with torch.no_grad():
            gl = hs.attn.g_proj(x[:n])[0]
        pre_cmp = rel(
            pre_a[:n], (pre_s[:n].float() * gl.float().sigmoid()).to(pre_a.dtype)
        )
    else:
        ref_a = reference_out(ha, pre_a[:n], x[:n])
        pre_cmp = rel(pre_a[:n], pre_s[:n])
    e_gate_stage = rel(a1[:n], ref_a)
    r = stat([e_noise, e_sa, e_gate_stage, pre_cmp], ha.device)
    finite = agree(rank, bool(torch.isfinite(a1).all()))
    check(rank, finite, f"e2e {name}: non-finite")
    # selections can flip with 1-ulp changes upstream (random weights): the bound is
    # loose
    check(rank, r[2] < 1e-2, f"e2e {name}: gate/join/o_proj stage vs fp32 {r[2]}")
    check(
        rank,
        r[1] < max(0.35, 3 * r[0]),
        f"e2e {name}: akr vs stock relL2 {r[1]} (noise {r[0]})",
    )
    if expect is not None:
        from vllm.models.kimi_k3.amd import akr_dsa as A

        if A._flydsl_sparse_wins(torch.empty((T, 96, 576), device="meta")):
            # rows the hook leaves to the faster stock FlyDSL kernel
            expect = [k for k in expect if k != "sparse_mla_decode"]
        for k in expect:
            check(rank, calls.get(k, 0) >= 1, f"e2e {name}: {k} did not run ({calls})")
    log(
        rank,
        f"[e2e] {name}: rows={n}/{T} akr-vs-stock relL2={r[1]:.3e} (stock run-to-run "
        f"{r[0]:.3e}); "
        f"pre-gate akr-vs-stock={r[3]:.3e}; gate+o_proj stage vs fp32={r[2]:.2e}; "
        f"calls={calls}",
    )
    return calls


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
    "T100pad": (
        [(int(c), 4) for c in np.random.default_rng(4).integers(2000, 9000, 23)],
        2,
        100,
    ),
    "T12": (
        [(int(c), 4) for c in np.random.default_rng(5).integers(2000, 9000, 3)],
        0,
        12,
    ),
    "T192": (
        [(int(c), 4) for c in np.random.default_rng(6).integers(2000, 9000, 48)],
        0,
        192,
    ),
    "T240": (
        [(int(c), 5) for c in np.random.default_rng(7).integers(2000, 9000, 48)],
        0,
        240,
    ),
    "T240pad": (
        [(int(c), 5) for c in np.random.default_rng(8).integers(2000, 9000, 45)],
        3,
        240,
    ),
}


def graph_case(ha, hs, rank, name, bench, iters):
    from vllm.distributed.parallel_state import graph_capture

    lens, pad, T = GRAPH_CASES[name]
    ctx_a, _, _, n = ha.metadata(lens, pad)
    ctx_s, _, _, _ = hs.metadata(lens, pad)
    x = torch.zeros((T, ha.hf.hidden_size), dtype=torch.bfloat16, device=ha.device)
    graphs, outs = {}, {}
    pool = torch.cuda.graph_pool_handle()
    CALLS.clear()
    with graph_capture(ha.device) as gctx:
        for key, h, ctx, on in (("stock", hs, ctx_s, False), ("akr", ha, ctx_a, True)):
            x.normal_()
            with akr_env(on), torch.cuda.stream(gctx.stream):
                for _ in range(2):
                    h.run(ctx, x)
            torch.accelerator.synchronize()
            gr = torch.cuda.CUDAGraph()
            with akr_env(on), torch.cuda.graph(gr, pool=pool, stream=gctx.stream):
                o = h.run(ctx, x)
            graphs[key], outs[key] = gr, o
    torch.accelerator.synchronize()
    dist.barrier()
    for step in range(2):
        for h in (ha, hs):
            h.fill_caches(seed=20 + step)
        g = torch.Generator(device=ha.device).manual_seed(500 + step)
        x.copy_(torch.randn(x.shape, generator=g, device=ha.device).bfloat16())
        with akr_env(False):
            eager_s = hs.run(ctx_s, x).clone()
            eager_s2 = hs.run(ctx_s, x).clone()
        with akr_env(True):
            eager_a = ha.run(ctx_a, x).clone()
        torch.accelerator.synchronize()
        res = {}
        for key in ("stock", "akr"):
            graphs[key].replay()
            torch.accelerator.synchronize()
            res[key] = outs[key][:n].clone()
        graphs["akr"].replay()
        torch.accelerator.synchronize()
        rep2 = outs["akr"][:n].clone()
        e = stat(
            [
                rel(res["stock"], eager_s[:n]),
                rel(res["akr"], eager_a[:n]),
                rel(res["akr"], res["stock"]),
                rel(eager_s2[:n], eager_s[:n]),
                rel(rep2, res["akr"]),
                rel(eager_a[:n], eager_s[:n]),
            ],
            ha.device,
        )
        check(
            rank,
            agree(rank, bool(torch.isfinite(res["akr"]).all())),
            f"graph {name}: non-finite",
        )
        tol = max(3 * e[3], 0.35)
        check(rank, e[1] <= tol, f"graph {name}: akr graph vs akr eager {e[1]}")
        check(rank, e[2] <= tol, f"graph {name}: akr graph vs stock graph {e[2]}")
        log(
            rank,
            f"[graph] step{step} {name}: rows={n}/{T} relL2 graph-vs-eager "
            f"stock={e[0]:.2e} akr={e[1]:.2e}; "
            f"akr-graph-vs-stock-graph={e[2]:.2e}; stock eager run-to-run={e[3]:.2e}; "
            f"akr replay-vs-replay={e[4]:.2e}; "
            f"akr-eager-vs-stock-eager={e[5]:.2e}",
        )
    result = None
    del graphs, outs, gr, o
    torch.accelerator.synchronize()
    dist.barrier()
    return result


def bench_layers(ha, hs, rank, name, iters, only_switches=True):
    """Layer time in a FULL graph, stock vs integrated: the per-layer time is the
    difference of a graph of 5 and one of 1 consecutive layer calls over 4 (a one-layer
    graph carries 10-50 us of replay start skew across 8 ranks); min over `iters`
    timings, max over ranks."""
    from vllm.distributed.parallel_state import graph_capture

    lens, pad, T = GRAPH_CASES[name]
    ctx_a, _, _, n = ha.metadata(lens, pad)
    ctx_s, _, _, _ = hs.metadata(lens, pad)
    g = torch.Generator(device=ha.device).manual_seed(99)
    x = torch.randn((T, ha.hf.hidden_size), generator=g, device=ha.device).bfloat16()
    configs = [("stock", hs, ctx_s, set()), ("akr-all", ha, ctx_a, set(SWITCHES))]
    if only_switches:
        configs += [(f"only-{sw}", ha, ctx_a, {sw}) for sw in SWITCHES]
    out = {}
    for label, h, ctx, on in configs:
        graphs = {}
        for reps in (1, 5):
            with graph_capture(ha.device) as gctx:
                with akr_env(on), torch.cuda.stream(gctx.stream):
                    h.run(ctx, x)
                torch.accelerator.synchronize()
                gr = torch.cuda.CUDAGraph()
                with akr_env(on), torch.cuda.graph(gr, stream=gctx.stream):
                    for _ in range(reps):
                        h.run(ctx, x)
            graphs[reps] = gr
        torch.accelerator.synchronize()
        dist.barrier()
        for gr in graphs.values():
            for _ in range(5):
                gr.replay()
        torch.accelerator.synchronize()
        best = {1: 1e30, 5: 1e30}
        for _ in range(iters):
            for reps, gr in graphs.items():
                a = torch.cuda.Event(enable_timing=True)
                b = torch.cuda.Event(enable_timing=True)
                a.record()
                for _ in range(3):
                    gr.replay()
                b.record()
                b.synchronize()
                best[reps] = min(best[reps], a.elapsed_time(b) * 1000.0 / 3)
        per = (best[5] - best[1]) / 4
        r = stat([per, best[1], best[5]], ha.device)
        out[label] = r[0]
        log(
            rank,
            f"[bench] {name} {label}: layer {r[0]:.1f} us  (graphs of 1 / 5 layers: "
            f"{r[1]:.1f} / {r[2]:.1f} us)",
        )
        del graphs, gr
        dist.barrier()
    base = out["stock"]
    log(
        rank,
        f"[bench] {name} summary (us per layer, min of {iters}, max over ranks): "
        + ", ".join(f"{k} {v:.1f} ({v - base:+.1f})" for k, v in out.items()),
    )
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", action="store_true")
    ap.add_argument("--iters", type=int, default=15)
    ap.add_argument("--no-graph", action="store_true")
    ap.add_argument("--bench-only", action="store_true")
    ap.add_argument("--one", default="")
    ap.add_argument("--pad", type=int, default=0)
    ap.add_argument("--lens", default="")
    ap.add_argument("--cases", default="")
    ap.add_argument("--only", default="")
    ap.add_argument("--graph-cases", default="T64,T100,T100pad")
    args = ap.parse_args()
    only = set(filter(None, args.only.split(",")))
    run = lambda k: not only or k in only  # noqa: E731
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
    log(rank, f"[t] config {time.time() - t0:.1f}s")
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
        log(rank, f"[t] dist {time.time() - t0:.1f}s")
        try:
            from amd_kernels import warm_up

            warm_up.set_workers(4)
            count_calls()
            ha = Harness(
                config, rank, world, device, "model.layers.4.self_attn", akr=True
            )
            log(
                rank,
                f"[setup] akr layer {time.time() - t0:.1f}s; post-load extra bytes "
                "(this layer, incl. temporaries "
                f"released) = {ha.post_load_bytes / 2**20:.1f} MiB; akr_w "
                f"extra={ha.wrapper.akr_w.extra_bytes / 2**20:.1f} MiB",
            )
            hs = Harness(
                config, rank, world, device, "model.layers.8.self_attn", akr=False
            )
            log(
                rank,
                f"[setup] stock layer {time.time() - t0:.1f}s; post-load bytes "
                f"{hs.post_load_bytes / 2**20:.1f} MiB",
            )
            rng = np.random.default_rng(5)
            if run("stage"):
                for M in (12, 65, 192):
                    stage_prologue(ha, hs, rank, M)
                stage_indexer(
                    ha,
                    hs,
                    rank,
                    [(int(c), 4) for c in rng.integers(2000, 9000, 25)],
                    0,
                    "25x4",
                )
                stage_indexer(
                    ha,
                    hs,
                    rank,
                    [(int(c), 4) for c in rng.integers(2000, 9000, 3)],
                    0,
                    "3x4",
                )
                stage_indexer(
                    ha,
                    hs,
                    rank,
                    [(int(c), 5) for c in rng.integers(16000, 24000, 8)],
                    0,
                    "8x5 ~20k",
                )
                stage_indexer(
                    ha,
                    hs,
                    rank,
                    [(int(c), 4) for c in rng.integers(2000, 9000, 23)],
                    2,
                    "23x4 + 2 padded requests",
                )
                for rows in (12, 40, 100, 192):
                    stage_gather_merge(ha, hs, rank, rows)
                stage_gather_merge(ha, hs, rank, 64, "dense")
                stage_gather_merge(ha, hs, rank, 64, "ties")
                stage_gather_merge(ha, hs, rank, 64, "skew")
                for M in (12, 40, 100, 192, 240, 256):
                    stage_qgather(ha, hs, rank, M)
                for M in (12, 40, 100, 192, 200, 240, 256):
                    stage_sparse_combine(ha, hs, rank, M)
            if run("gm"):
                # the merge's receive regions must be clean after every kind of row
                # (skewed rows take the histogram / bin-B rounds)
                for kind, rows in (
                    ("skew", 64),
                    ("random", 92),
                    ("skew", 12),
                    ("random", 192),
                    ("skew", 40),
                    ("dense", 40),
                    ("skew", 192),
                    ("random", 64),
                    ("ties", 100),
                    ("random", 12),
                ):
                    stage_gather_merge(ha, hs, rank, rows, kind)
                for lens, pad, tag in (
                    (
                        [(0, 1), (3, 2), (4000, 5), (10, 3), (2047, 4), (2048, 1)],
                        0,
                        "mixed",
                    ),
                    (
                        [
                            (0, 4),
                            (1, 4),
                            (2, 4),
                            (3, 4),
                            (5, 4),
                            (8, 4),
                            (13, 4),
                            (100, 4),
                        ],
                        0,
                        "tiny",
                    ),
                    (
                        [(int(c), 4) for c in rng.integers(2000, 9000, 23)],
                        2,
                        "23x4 + 2 padded",
                    ),
                    ([(int(c), 4) for c in rng.integers(2000, 9000, 25)], 0, "25x4"),
                ):
                    stage_indexer(ha, hs, rank, lens, pad, "seq " + tag)
            lens_arg = (
                [tuple(int(v) for v in t.split(":")) for t in args.lens.split(",")]
                if args.lens
                else None
            )
            if args.one:
                sw = (
                    set(filter(None, args.one.split("+")))
                    if args.one != "none"
                    else set()
                )
                e2e(
                    ha,
                    hs,
                    rank,
                    f"one: {args.one}",
                    lens_arg
                    or [(int(c), 4) for c in rng.integers(1000, 9000, 25 - args.pad)],
                    pad=args.pad,
                    switches=sw,
                )
            elif run("e2e"):
                every = [
                    "prologue_a",
                    "prologue_b",
                    "qgather",
                    "gather_merge",
                    "sparse_mla_decode",
                    "combine",
                ]
                draw = lambda n: [  # noqa: E731
                    (int(c), 4) for c in rng.integers(1000, 9000, n)
                ]
                lens25 = draw(25)
                cases: list[tuple] = [
                    ("T100 (25x4)", lens25, 0, True, every),
                    ("T12 (3x4)", draw(3), 0, True, every),
                    ("T192 (48x4)", draw(48), 0, True, every),
                    (
                        "empty shards (tiny seqs)",
                        [
                            (0, 4),
                            (1, 4),
                            (2, 4),
                            (3, 4),
                            (5, 4),
                            (8, 4),
                            (13, 4),
                            (100, 4),
                        ],
                        0,
                        True,
                        None,
                    ),
                    (
                        "mixed query lens",
                        [(0, 1), (3, 2), (4000, 5), (10, 3), (2047, 4), (2048, 1)],
                        0,
                        True,
                        None,
                    ),
                    ("padded reqs (23x4 + 2 pad)", draw(23), 2, True, None),
                    ("single row", [(5000, 1)], 0, True, None),
                    ("T200 (50x4)", draw(50), 0, True, every),
                    (
                        # the prologue (and so the fused score + top-k) takes one
                        # indexer decode tile: decode_tile_size at 1M context = 233 rows
                        "T240 (48x5)",
                        [(int(c), 5) for c in rng.integers(1000, 9000, 48)],
                        0,
                        True,
                        ["qgather", "gather_merge", "sparse_mla_decode", "combine"],
                    ),
                ]
                cases += [(f"only {sw}", lens25, 0, {sw}, None) for sw in SWITCHES]
                cases += [("all off (akr layer, stock paths)", lens25, 0, False, None)]
                pick = (
                    [int(i) for i in args.cases.split(",")]
                    if args.cases
                    else range(len(cases))
                )
                for i in pick:
                    name, lens, pad, sw, expect = cases[i]
                    e2e(ha, hs, rank, name, lens, pad=pad, switches=sw, expect=expect)
            bench = {}
            if not args.no_graph and run("graph") and not args.bench_only:
                for name in args.graph_cases.split(","):
                    graph_case(ha, hs, rank, name, False, args.iters)
            if args.bench and run("bench"):
                for name in args.graph_cases.split(","):
                    bench[name] = bench_layers(ha, hs, rank, name, args.iters)
            if rank == 0 and bench:
                print("[bench-json]", json.dumps(bench), flush=True)
            late = warm_up.late_compiles()
            check(rank, late == 0, f"{late} amd_kernels compiles after the warm-up")
            log(rank, f"[warm-up] late compiles after the layer's warm-up: {late}")
        finally:
            fails = torch.tensor([len(FAILS)], device=device)
            dist.all_reduce(fails)
            log(rank, "RESULT:", "PASS" if int(fails) == 0 else f"FAIL ({int(fails)})")
            cleanup_dist_env_and_memory()
    sys.exit(0 if not FAILS else 1)


if __name__ == "__main__":
    main()
