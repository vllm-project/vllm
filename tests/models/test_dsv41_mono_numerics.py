# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Numerics of the DeepSeek-V4.1 mono decode kernels at TP2 on CDNA4 against the
ops vLLM runs for the same decode step: random weights in the checkpoint's
formats, the step's KV caches as vLLM's insert writes them.

- The whole layer (K1 + K2) against the mHC seams, the ROCm attention op chain,
  the TP all-reduce and ``DeepseekV4MoE``.
- The FFN launch against the same FFN half, from wo_b's unreduced output.

Each stage is compared on its own input, so a failure names the stage; the MoE
is checked on the kernels' own FFN input, where routing cannot differ."""

import pytest
import ray
import torch

from vllm.platforms import current_platform

from ..utils import (
    ensure_model_parallel_initialized,
    init_test_distributed_environment,
    multi_process_parallel,
)

MODEL = "deepseek-ai/DeepSeek-V4.1-Flash"
HIDDEN, Q_RANK, HEAD_DIM, NOPE, ROPE = 5120, 1280, 512, 448, 64
N_HEADS, O_GROUPS, O_RANK = 64, 8, 1024
WINDOW, TOPK, SWA_BLOCK, CACHE_BLOCK = 128, 512, 32, 128
RECORD, DATA, ALIGN = 584, 576, 576
REPLAYS = 300
HC = 4


def _on_cdna4() -> bool:
    if not current_platform.is_rocm():
        return False
    from vllm.platforms.rocm import get_cdna_version

    return get_cdna_version() == 4


def _cos(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.float().reshape(-1), b.float().reshape(-1)
    return torch.nn.functional.cosine_similarity(a, b, dim=0).item()


def _rel(a: torch.Tensor, b: torch.Tensor) -> float:
    """Relative L2 error of ``a`` against ``b``: unlike the cosine, it sees a
    wrong scale."""
    a, b = a.float().reshape(-1), b.float().reshape(-1)
    return ((a - b).norm() / b.norm()).item()


# ---------------------------------------------------------------- caches


def _paged_cache(num_blocks: int, block: int, device) -> torch.Tensor:
    """[blocks, block, 584] u8 over pages of round_up(block * 584, 576) bytes,
    the way vLLM binds a packed V4 cache."""
    page = -(-block * RECORD // ALIGN) * ALIGN
    raw = torch.zeros(num_blocks * page, dtype=torch.uint8, device=device)
    return torch.as_strided(raw, (num_blocks, block, RECORD), (page, RECORD, 1))


def _rope(kv: torch.Tensor, positions: torch.Tensor, cos_sin: torch.Tensor):
    """GPT-J RoPE of the last 64 dims (fp32 math, bf16 out)."""
    cs = cos_sin[positions.long()]
    cos, sin = cs[:, :32], cs[:, 32:]
    x = kv.float()
    e, o = x[:, NOPE::2].clone(), x[:, NOPE + 1 :: 2].clone()
    x[:, NOPE::2] = e * cos - o * sin
    x[:, NOPE + 1 :: 2] = e * sin + o * cos
    return x.to(torch.bfloat16)


def _write_records(cache, block: int, slots, kv):
    """fp8_ds_mla records of roped bf16 kv rows at ``slots``: the 448 nope dims
    in e4m3 with a UE8M0 per 64, the rope dims in bf16, as vLLM's insert."""
    nope = kv.float()[:, :NOPE].view(-1, 7, 64)
    exp = torch.ceil(torch.log2(nope.abs().amax(-1).clamp_min(1e-4) / 448.0))
    q = (nope * torch.exp2(-exp)[..., None]).clamp(-448, 448).reshape(-1, NOPE)
    data = torch.cat(
        [
            q.to(torch.float8_e4m3fn).view(torch.uint8),
            kv[:, NOPE:].contiguous().view(torch.uint8),
        ],
        1,
    )
    scales = torch.zeros(kv.shape[0], 8, dtype=torch.uint8, device=kv.device)
    scales[:, :7] = (exp + 127).clamp(0, 255).to(torch.uint8)
    n = cache.untyped_storage().nbytes() - cache.storage_offset()
    flat = torch.as_strided(cache, (n,), (1,))
    slots = slots.to(kv.device).long()
    b, o = slots // block, slots % block
    d0 = b * cache.stride(0) + o * DATA
    s0 = b * cache.stride(0) + block * DATA + o * 8
    ar = torch.arange(DATA, device=kv.device)
    flat[(d0[:, None] + ar).reshape(-1)] = data.reshape(-1)
    ar = torch.arange(8, device=kv.device)
    flat[(s0[:, None] + ar).reshape(-1)] = scales.reshape(-1)


def _normed_kv(n: int, kv_norm, g, device):
    x = torch.randn(n, HEAD_DIM, generator=g).to(device)
    x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True)) * kv_norm.float()
    return x.to(torch.bfloat16)


def _decode_step(ratio: int, R: int, T: int, kv_norm, cos_sin, seed: int):
    """R requests x T tokens past random contexts: the step's metadata as
    vLLM's builders lay it out, its window rows and (ratio > 0) the compressed
    cache filled, a random top-512 a token."""
    g = torch.Generator().manual_seed(seed)
    dev = kv_norm.device
    ctx = [int(c) for c in torch.randint(2000, 40000, (R,), generator=g)]
    M = R * T
    spans = [
        (max(0, c - WINDOW - T) // SWA_BLOCK, (c + T - 1) // SWA_BLOCK) for c in ctx
    ]
    n_swa = 1 + sum(hi - lo + 1 for lo, hi in spans)
    perm = (torch.randperm(n_swa - 1, generator=g) + 1).tolist()
    bt = torch.zeros(R, max(hi for _, hi in spans) + 1, dtype=torch.int64)
    for r, (lo, hi) in enumerate(spans):
        for lb in range(lo, hi + 1):
            bt[r, lb] = perm.pop()
    swa = _paged_cache(n_swa, SWA_BLOCK, dev)

    def slot(r, p):
        return int(bt[r, p // SWA_BLOCK]) * SWA_BLOCK + p % SWA_BLOCK

    pos = [ctx[r] + t for r in range(R) for t in range(T)]
    t2r = [r for r in range(R) for _ in range(T)]
    swa_idx = torch.full((M, WINDOW), -1, dtype=torch.int32)
    swa_len = torch.zeros(M, dtype=torch.int32)
    for i in range(M):
        ps = range(max(pos[i] - WINDOW + 1, 0), pos[i] + 1)
        swa_idx[i, : len(ps)] = torch.tensor([slot(t2r[i], p) for p in ps])
        swa_len[i] = len(ps)
    hist = [
        (r, p) for r in range(R) for p in range(max(0, ctx[r] - WINDOW - T), ctx[r])
    ]
    hp = torch.tensor([p for _, p in hist], device=dev)
    kv = _rope(_normed_kv(len(hist), kv_norm, g, dev), hp, cos_sin)
    _write_records(swa, SWA_BLOCK, torch.tensor([slot(r, p) for r, p in hist]), kv)

    step = dict(
        M=M,
        positions=torch.tensor(pos, dtype=torch.int64, device=dev),
        slot_mapping=torch.tensor(
            [slot(t2r[i], pos[i]) for i in range(M)], dtype=torch.int64, device=dev
        ),
        token_to_req=torch.tensor(t2r, dtype=torch.int32, device=dev),
        swa_indices=swa_idx.to(dev),
        swa_lens=swa_len.to(dev),
        swa_cache=swa,
        topk=None,
        comp_cache=None,
        comp_block_table=None,
        comp_entries=0,
    )
    if ratio:
        entries = CACHE_BLOCK // ratio
        lens = [(c + T) // ratio for c in ctx]
        nblk = [-(-n // entries) for n in lens]
        perm = (torch.randperm(sum(nblk), generator=g) + 1).tolist()
        cbt = torch.zeros(R, max(nblk), dtype=torch.int32)
        for r in range(R):
            for b in range(nblk[r]):
                cbt[r, b] = perm.pop()
        comp = _paged_cache(1 + sum(nblk), entries, dev)
        for r in range(R):
            idx = torch.arange(lens[r])
            slots = cbt[r, idx // entries].long() * entries + idx % entries
            kv = _rope(
                _normed_kv(lens[r], kv_norm, g, dev), (idx * ratio).to(dev), cos_sin
            )
            _write_records(comp, entries, slots, kv)
        topk = torch.full((M, TOPK), -1, dtype=torch.int32)
        for i in range(M):
            cand = (pos[i] + 1) // ratio
            n = min(cand, TOPK)
            topk[i, :n] = torch.randperm(cand, generator=g)[:n].to(torch.int32)
        step.update(
            topk=topk.to(dev),
            comp_cache=comp,
            comp_block_table=cbt.to(dev),
            comp_entries=entries,
        )
    return step


# ---------------------------------------------------------------- weights


def _mxfp8(n: int, k: int, g, device):
    """A random MXFP8 weight [n, k] e4m3 and its E8M0 32 x 32 block scales."""
    w = torch.randn(n, k, generator=g).clamp(-6, 6).to(torch.float8_e4m3fn)
    s = torch.randint(118, 123, (n // 32, k // 32), generator=g, dtype=torch.uint8)
    return w.to(device), s.to(device)


def _random_moe(moe, gd, gr, inter: int) -> None:
    """Random parameters in the loaded layout, before post-load processing: this
    rank's shards (``gd``, a device generator) of the MXFP4 routed experts
    (random nibbles, E8M0 scales; the intermediate past ``inter`` zero, as the
    loader pads it) and the MXFP8 shared expert (one E8M0 a 32 x 32 block); the
    replicated router from ``gr``, seeded alike on every rank."""
    u8 = torch.uint8
    for lin in (moe.shared_experts.gate_up_proj, moe.shared_experts.down_proj):
        n = lin.weight.shape[0]
        w = torch.randn(lin.weight.shape, generator=gd, device=gd.device)
        lin.weight.data.copy_(w.clamp(-6, 6).to(lin.weight.dtype))
        sc = lin.weight_scale.data.view(u8)
        rows = -(-n // 32) if sc.shape[0] == n else sc.shape[0]
        blocks = torch.randint(
            118, 123, (rows, sc.shape[1]), generator=gd, device=gd.device, dtype=u8
        )
        sc.copy_(blocks.repeat_interleave(32, 0)[:n] if sc.shape[0] == n else blocks)
    e = moe.experts.routed_experts
    for name, lo, hi in (("weight", 0, 256), ("weight_scale", 118, 122)):
        for w in ("w13", "w2"):
            t = getattr(e, f"{w}_{name}").data.view(u8)
            t.copy_(
                torch.randint(lo, hi, t.shape, generator=gd, device=gd.device, dtype=u8)
            )
    padded = e.w13_weight.shape[1] // 2
    if padded > inter:
        w13, w2 = e.w13_weight.data.view(u8), e.w2_weight.data.view(u8)
        for half in (0, padded):
            w13[:, half + inter : half + padded] = 0
        w2[..., inter // 2 :] = 0
    for t in (moe.gate.weight, moe.gate.e_score_correction_bias):
        t.data.copy_(torch.randn(t.shape, generator=gr, device=gr.device) * 0.02)


# ---------------------------------------------------------------- the worker


@ray.remote(num_gpus=1, max_calls=1)
def _mono_numerics(monkeypatch, tp_size, pp_size, rank, distributed_init_port):
    from vllm._aiter_ops import rocm_aiter_ops
    from vllm.v1.worker.workspace import init_workspace_manager

    with monkeypatch.context() as m:
        m.delenv("CUDA_VISIBLE_DEVICES", raising=False)
        # the checkpoint's MXFP4 experts run on AITER's MoE; its flags are read
        # at import, so refresh them
        m.setenv("VLLM_ROCM_USE_AITER", "1")
        m.setenv("VLLM_ROCM_USE_AITER_MOE", "1")
        rocm_aiter_ops.refresh_env_variables()
        device = torch.device(f"cuda:{rank}")
        torch.accelerator.set_device_index(device)
        init_test_distributed_environment(tp_size, pp_size, rank, distributed_init_port)
        ensure_model_parallel_initialized(tp_size, pp_size)
        init_workspace_manager(device)
        _check_layers(rank, tp_size, device)


def _check_layers(rank: int, tp: int, device) -> None:
    from vllm._aiter_ops import rocm_aiter_ops
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.distributed import get_tp_group, tensor_model_parallel_all_reduce
    from vllm.engine.arg_utils import EngineArgs
    from vllm.forward_context import set_forward_context
    from vllm.model_executor.kernels.linear.mxfp8.rocm_block32_gemm import (
        rocm_mxfp8_block32_gemm,
    )
    from vllm.model_executor.layers.mhc import MHCPreDelayedOp
    from vllm.model_executor.layers.quantization.utils.mxfp8_utils import (
        mxfp8_e4m3_quantize,
    )
    from vllm.model_executor.model_loader.utils import process_weights_after_loading
    from vllm.models.common.ops import fused_q_kv_rmsnorm
    from vllm.models.deepseek_v41.amd.model import DeepseekV4MoE
    from vllm.models.deepseek_v41.amd.mono.layer import scratch_layout
    from vllm.models.deepseek_v41.amd.mono.runner import (
        AttnWeights,
        DSV41MonoLayer,
        MonoLayerWeights,
    )
    from vllm.models.deepseek_v41.amd.rocm import (
        compute_global_topk_ragged_indices_and_indptr,
    )
    from vllm.models.deepseek_v41.common.rope import build_deepseek_v4_rope
    from vllm.utils.torch_utils import set_default_torch_dtype
    from vllm.v1.attention.ops.rocm_aiter_mla_sparse import (
        build_ragged_indices_from_dense,
        rocm_inverse_rope_mxfp8_rows,
    )

    vc = EngineArgs(
        model=MODEL,
        tensor_parallel_size=tp,
        moe_backend="aiter",
        load_format="dummy",
        max_model_len=65536,
        language_model_only=True,
        attention_config={"indexer_kv_dtype": "mxfp4", "indexer_sparse_logits": True},
    ).create_engine_config()
    cfg = vc.model_config.hf_config
    cfg = getattr(cfg, "text_config", cfg)
    # replicated tensors alike on every rank; each rank's own TP shards
    g = torch.Generator().manual_seed(1000)
    gs = torch.Generator().manual_seed(1001 + rank)
    H, G = N_HEADS // tp, O_GROUPS // tp
    bf = torch.bfloat16

    with set_current_vllm_config(vc), set_default_torch_dtype(bf), device:
        moe = DeepseekV4MoE(vc, prefix="model.layers.21.ffn")
        mhc = MHCPreDelayedOp()
    _random_moe(
        moe,
        torch.Generator(device=device).manual_seed(2001 + rank),
        torch.Generator(device=device).manual_seed(2000),
        inter=2304 // tp,
    )
    process_weights_after_loading(moe, vc.model_config, device)

    def f32(*shape, s=1.0, gen=g):
        return (torch.randn(*shape, generator=gen) * s).to(device)

    mix = (2 + HC) * HC
    seams = {
        sub: dict(
            fn=f32(mix, HC * HIDDEN, s=0.01),
            scale=f32(3, s=0.5).abs() + 0.5,
            base=f32(mix, s=0.1),
            norm=(1 + f32(HIDDEN, s=0.1)).to(bf),
        )
        for sub in ("attn", "ffn")
    }

    def seam(sub, residual, x, post, comb, pre):
        s = seams[sub]
        return mhc(
            residual, s["fn"], s["scale"], s["base"], cfg.rms_norm_eps,
            cfg.hc_eps, cfg.hc_eps, 2.0, cfg.hc_sinkhorn_iters,
            pre_mix=pre, sublayer_out=x, post_layer_mix=post, comb_res_mix=comb,
            norm_weight=s["norm"], norm_eps=cfg.rms_norm_eps,
        )  # fmt: skip

    def moe_out(x):
        with set_forward_context(None, vc):
            ids = torch.zeros(x.shape[0], dtype=torch.int64, device=device)
            return moe(x.clone(), ids)

    wqkv, wqkv_s = _mxfp8(1792, HIDDEN, g, device)
    wq_b, wq_b_s = _mxfp8(H * HEAD_DIM, Q_RANK, gs, device)
    wo_a, wo_a_s = _mxfp8(G * O_RANK, H // G * HEAD_DIM, gs, device)
    wo_b, wo_b_s = _mxfp8(HIDDEN, G * O_RANK, gs, device)
    q_norm, kv_norm = (1 + f32(Q_RANK, s=0.1)).to(bf), (1 + f32(HEAD_DIM, s=0.1)).to(bf)
    sink = f32(H, gen=gs)
    e, sh = moe.experts.routed_experts, moe.shared_experts
    u8 = lambda t: t.view(torch.uint8)  # noqa: E731
    runner = DSV41MonoLayer(tp, rank, get_tp_group().cpu_group, device)
    gw, bias = moe.gate.weight.float(), moe.gate.e_score_correction_bias.float()

    def top6(x):
        """The top-6 experts as vLLM routes: sqrt(softplus(logits)) + bias."""
        sc = torch.nn.functional.softplus(x.float() @ gw.T).sqrt() + bias
        return sc.topk(6, dim=1).indices.sort(1).values

    steps = []  # each case's launches, on fixed buffers, for the replay stress
    for ratio, R, T in ((1, 1, 6), (2, 8, 6)):
        with set_current_vllm_config(VllmConfig()):
            rope = build_deepseek_v4_rope(
                cfg, head_dim=HEAD_DIM, rope_head_dim=ROPE,
                max_position_embeddings=cfg.max_position_embeddings,
                compress_ratio=ratio,
            )  # fmt: skip
        cos_sin = rope.cos_sin_cache.to(device=device, dtype=torch.float32)
        s = _decode_step(ratio, R, T, kv_norm, cos_sin, seed=7 + ratio)
        M = s["M"]
        mw = MonoLayerWeights(
            attn=AttnWeights(
                layer_id=21, wqkv=wqkv, wqkv_scale=wqkv_s, q_norm=q_norm,
                kv_norm=kv_norm, wq_b=wq_b, wq_b_scale=wq_b_s, wo_a=wo_a,
                wo_a_scale=wo_a_s, wo_b=wo_b, wo_b_scale=wo_b_s, attn_sink=sink,
                cos_sin=cos_sin, ratio=ratio,
            ),
            hc_attn_fn=seams["attn"]["fn"], hc_attn_scale=seams["attn"]["scale"],
            hc_attn_base=seams["attn"]["base"], attn_norm=seams["attn"]["norm"],
            hc_ffn_fn=seams["ffn"]["fn"], hc_ffn_scale=seams["ffn"]["scale"],
            hc_ffn_base=seams["ffn"]["base"], ffn_norm=seams["ffn"]["norm"],
            gate_w=moe.gate.weight, bias=moe.gate.e_score_correction_bias,
            w13=e.w13_weight, w13_s=e.w13_weight_scale, w2=e.w2_weight,
            w2_s=e.w2_weight_scale, sgu=sh.gate_up_proj.weight,
            sgu_s=u8(sh.gate_up_proj.weight_scale), sw2=sh.down_proj.weight,
            sw2_s=u8(sh.down_proj.weight_scale),
        )  # fmt: skip

        # the layer's inputs at the attention seam (the same on every rank)
        gi = torch.Generator().manual_seed(11)
        res0 = torch.randn(M, HC, HIDDEN, generator=gi)
        res0 = (
            (res0 * torch.rsqrt(res0.pow(2).mean(-1, keepdim=True))).to(bf).to(device)
        )
        x0 = (torch.randn(M, HIDDEN, generator=gi) * 0.5).to(bf).to(device)
        comb0 = torch.rand(M, HC, HC, generator=gi) + 0.1
        for _ in range(20):
            comb0 = comb0 / comb0.sum(-1, keepdim=True)
            comb0 = comb0 / comb0.sum(-2, keepdim=True)
        comb0 = comb0.to(device)
        post0 = (torch.sigmoid(torch.randn(M, HC, 1, generator=gi)) * 2).to(device)
        pre0 = (torch.sigmoid(torch.randn(M, HC, generator=gi)) + 1e-6).to(device)

        # vLLM's decode chain
        res, post, comb, xa, attn_pre = seam("attn", res0, x0, post0, comb0, pre0)
        swa_ind, swa_ptr = build_ragged_indices_from_dense(
            s["swa_indices"].reshape(M, -1), s["swa_lens"]
        )
        top_ind = top_ptr = None
        if ratio:
            valid = torch.ones(M, dtype=torch.bool, device=device)
            top_ind, top_ptr, _ = compute_global_topk_ragged_indices_and_indptr(
                s["topk"], s["token_to_req"], s["comp_block_table"],
                s["comp_entries"], valid,
            )  # fmt: skip
        x8, xs = mxfp8_e4m3_quantize(xa)
        qr_kv = rocm_mxfp8_block32_gemm(x8, xs, wqkv, wqkv_s, bf)
        qr, kv = fused_q_kv_rmsnorm(
            qr_kv[:, :Q_RANK], qr_kv[:, Q_RANK:], q_norm, kv_norm, cfg.rms_norm_eps
        )
        q8, qs = mxfp8_e4m3_quantize(qr)
        q = rocm_mxfp8_block32_gemm(q8, qs, wq_b, wq_b_s, bf).view(M, H, HEAD_DIM)
        swa2d = s["swa_cache"].view(s["swa_cache"].shape[0], -1)
        q_roped = torch.ops._C.fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert(
            q, kv, swa2d, s["slot_mapping"], s["positions"], cos_sin, H,
            cfg.rms_norm_eps, SWA_BLOCK, False, False, True, False,
        )  # fmt: skip
        attn = torch.empty(M, H, HEAD_DIM, dtype=bf, device=device)
        rocm_aiter_ops.triton_sparse_mla_fwd(
            q_roped, s["swa_cache"], attn, HEAD_DIM**-0.5, swa_ptr, swa_ind,
            attn_sink=sink, extra_kv_buffer=s["comp_cache"],
            extra_kv_indptr=top_ptr, extra_kv_indices=top_ind,
        )  # fmt: skip
        a8 = torch.empty(M, H * HEAD_DIM, dtype=torch.float8_e4m3fn, device=device)
        a_s = torch.empty(M, H * HEAD_DIM // 32, dtype=torch.uint8, device=device)
        rocm_inverse_rope_mxfp8_rows(attn, s["positions"], cos_sin, ROPE, a8, a_s)
        z = torch.ops.vllm.rocm_dsv41_mxfp8_wo_a_bmm(a8, a_s, wo_a, wo_a_s, G, O_RANK)
        z8, zs = mxfp8_e4m3_quantize(z)
        part = rocm_mxfp8_block32_gemm(z8, zs, wo_b, wo_b_s, bf).contiguous()
        a = tensor_model_parallel_all_reduce(part.clone())
        res2, post2, comb2, xf, ffn_pre = seam("ffn", res, a, post, comb, attn_pre)
        out = moe_out(xf)

        # ---- the whole layer: K1 + K2
        o = runner.forward(
            mw, x0, res0, post0, comb0, pre0, s["positions"], s["slot_mapping"],
            s["swa_cache"], s["swa_indices"], s["swa_lens"], s["token_to_req"],
            topk_indices=s["topk"], comp_cache=s["comp_cache"],
            comp_block_table=s["comp_block_table"],
        )  # fmt: skip
        off = scratch_layout(M, tp)["normed"][0]
        normed = runner.scratch(M)[off : off + M * HIDDEN * 2].view(bf).view(M, HIDDEN)
        tag = f"ratio {ratio}, {R} x {T}"
        assert _cos(runner.res_mid[:M], res) > 0.99999, tag
        for got, want in (
            (runner.post_a, post),
            (runner.comb_a, comb),
            (runner.pre_a, attn_pre),
        ):
            torch.testing.assert_close(
                got[:M].view_as(want), want, rtol=1e-4, atol=1e-5
            )
        same_in = moe_out(normed)
        # relative errors about 3x what the kernels show (0.3-0.5%)
        assert _cos(o[1], res2) > 0.9999 and _rel(o[1], res2) < 0.01, tag
        assert _cos(normed, xf) > 0.999 and _rel(normed, xf) < 0.012, tag
        assert _cos(o[0], same_in) > 0.9999 and _rel(o[0], same_in) < 0.015, tag
        # per token, where both FFN inputs pick the same experts
        same = (top6(xf) == top6(normed)).all(1)
        assert same.float().mean() >= 0.5, tag
        tok = torch.nn.functional.cosine_similarity(o[0].float(), out.float(), dim=1)
        assert tok[same].min() > 0.99, tag

        # ---- the FFN launch, from the same unreduced wo_b output
        f = runner.ffn(mw, part, res, post, comb, attn_pre)
        torch.testing.assert_close(f[1], res2, rtol=2**-8, atol=1e-6)
        torch.testing.assert_close(normed, xf, rtol=2**-7, atol=1e-6)
        assert _cos(f[0], out) > 0.9999 and _rel(f[0], out) < 0.015, tag
        torch.accelerator.synchronize()

        whole, ffn = runner._outs(M, res0), runner._outs(M, res)
        args = (s["positions"], s["slot_mapping"], s["swa_cache"], s["swa_indices"])
        meta = dict(
            topk_indices=s["topk"], comp_cache=s["comp_cache"],
            comp_block_table=s["comp_block_table"],
        )  # fmt: skip

        def step(mw=mw, x0=x0, res0=res0, post0=post0, comb0=comb0, pre0=pre0,
                 args=args, s=s, meta=meta, whole=whole, part=part, res=res,
                 post=post, comb=comb, attn_pre=attn_pre, ffn=ffn):  # fmt: skip
            runner.forward(
                mw, x0, res0, post0, comb0, pre0, *args, s["swa_lens"],
                s["token_to_req"], **meta, outs=whole,
            )  # fmt: skip
            runner.ffn(mw, part, res, post, comb, attn_pre, outs=ffn)
            return [*whole, *ffn]

        steps.append(step)

    # ---- the hand-off protocol under back-to-back launches: both step widths
    # alternating in one graph, every replay bit for bit the first's
    outs = [t for step in steps for t in step()]
    torch.accelerator.synchronize()
    ref = [t.clone() for t in outs]
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for step in steps:
            step()
    for i in range(REPLAYS):
        graph.replay()
        if i % 50 == 49 or i == REPLAYS - 1:
            torch.accelerator.synchronize()
            bad = [
                j for j, (t, r) in enumerate(zip(outs, ref)) if not torch.equal(t, r)
            ]
            assert not bad, f"replay {i}: outputs {bad} differ from the first run"


@pytest.mark.skipif(not _on_cdna4(), reason="the mono kernels target CDNA4")
@pytest.mark.parametrize("tp", [2, 4])
def test_mono_decode_layer_matches_vllm(monkeypatch: pytest.MonkeyPatch, tp: int):
    if torch.accelerator.device_count() < tp:
        pytest.skip(f"needs {tp} GPUs")
    pytest.importorskip("flydsl")
    multi_process_parallel(monkeypatch, tp, 1, _mono_numerics)
