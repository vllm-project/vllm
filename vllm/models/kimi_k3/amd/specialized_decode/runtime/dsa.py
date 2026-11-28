# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kimi-K3.1's DSA layers at DCP1 (`VLLM_KIMI_SD_DSA`): the prologue, then K2,
which runs the split-K sparse attention over the indexer's top-k first, merges
its slices (W_UV and the gate in its M1b), then o_proj and the MoE.

The prologue is one `dsa_front` launch that also scores the MXFP4 indexer cache
and selects the exact top-k, or vLLM's query, KV and indexer ops (`prologue`)
where the front cannot serve. Pure decode steps of one request only
(the layer runs vLLM's ops otherwise).
"""

from typing import TYPE_CHECKING, Any

import torch

from vllm.logger import init_logger
from vllm.models.kimi_k3.amd.specialized_decode.runtime.layers import (
    MlaOpsLayer,
    _k2,
    _k2_key,
)
from vllm.models.kimi_k3.amd.specialized_decode.runtime.runner import MonoRunner

if TYPE_CHECKING:
    from vllm.models.kimi_k3.amd.specialized_decode.runtime.decode import Step

logger = init_logger(__name__)

SPLITS = 16
# heads a rank, kv_lora_rank, rope, nope, v head dims, the indexer's top-k, the
# fp8_ds_mla record's bytes
DSA_SHAPES = (12, 512, 64, 128, 128, 2048, 656)


def enabled() -> bool:
    import vllm.envs as envs

    return envs.VLLM_KIMI_SD_DSA == "auto"


def dsa_key(width: int):
    from vllm.models.kimi_k3.amd.specialized_decode.mono.attention import dsa_attn

    return dsa_attn.DsaAttnBuild(
        tokens=width,
        splits=SPLITS,
        chunk=dsa_attn.TOPK // SPLITS,
        fold=True,
        standalone=False,
    )


def layer_shapes(layer) -> tuple:
    w = layer.self_attn.mla_attn
    a = w.mla_attn
    rec = getattr(a.impl, "get_fp8_ds_mla_row_bytes", None)
    return (
        a.num_heads,
        a.kv_lora_rank,
        a.qk_rope_head_dim,
        a.qk_nope_head_dim,
        a.v_head_dim,
        getattr(w, "topk_tokens", None),
        rec() if rec is not None and a.kv_cache_dtype == "fp8_ds_mla" else None,
    )


def dsa_reasons(runner: MonoRunner, layer) -> list[str]:
    """Why this DSA layer cannot run `DsaM1Layer` (empty: it can)."""
    if not enabled():
        return ["VLLM_KIMI_SD_DSA=off"]
    w: Any = getattr(layer.self_attn, "mla_attn", None)
    a: Any = getattr(w, "mla_attn", None)
    if a is None or w.indexer is None or not w.is_sparse:
        return ["a layer without DSA's sparse MLA"]
    topk = getattr(a.impl, "topk_indices_buffer", None)
    w_v = getattr(a, "W_UV", None)
    checks = (
        (
            topk is None
            or topk.dtype != torch.int32
            or topk.dim() != 2
            or topk.stride() != (DSA_SHAPES[5], 1),
            f"a top-k buffer not [tokens, {DSA_SHAPES[5]}] dense int32",
        ),
        (runner.dcp != 1, f"DCP{runner.dcp} (DCP1 only)"),
        (a.kv_cache_dtype != "fp8_ds_mla", f"a {a.kv_cache_dtype} KV cache"),
        (layer_shapes(layer) != DSA_SHAPES, f"DSA shapes other than {DSA_SHAPES}"),
        (
            a.is_aiter_triton_fp4_bmm_enabled or a.is_aiter_triton_fp8_bmm_enabled,
            "AITER's FP4 / FP8 MLA bmms (W_UV not bf16)",
        ),
        (
            w_v is None
            or w_v.dtype != torch.bfloat16
            or tuple(w_v.shape) != (DSA_SHAPES[0], DSA_SHAPES[1], DSA_SHAPES[4])
            or w_v.stride(1) != 1
            or w_v.data_ptr() % 16,
            "a W_UV not bf16 [12, 512, 128] with its 512 dense",
        ),
        (a.q_pad_num_heads is not None, "a padded query (q_pad_num_heads)"),
        (w.rotary_emb is not None, "RoPE (the prologue applies none)"),
        (w.q_lora_rank is None, "MLA without q-LoRA"),
        (w.g_proj is None, "MLA without the output gate (g_proj)"),
        (a.hisparse_cache is not None, "HiSparse"),
        (a.use_pcp, "prefill context parallelism"),
        (w.dcp_q_replicate, "DCP query replication"),
        (getattr(a.impl, "sinks", None) is not None, "attention sinks"),
    )
    return [why for bad, why in checks if bad]


def width_served(runner: MonoRunner, width: int) -> bool:
    """One request a step, its slices on resident CTAs."""
    return width == runner.num_spec + 1 and fits(width)


def fits(width: int) -> bool:
    from vllm.models.kimi_k3.amd.specialized_decode.mono.attention import dsa_attn

    try:
        dsa_attn.check(dsa_key(width))
    except AssertionError:
        return False
    return True


def spec_reasons(vllm_config, q: int, seqs: int, check_geometry: bool) -> list[str]:
    """Why speculative DCP1 steps of `seqs` requests at qlen `q` are not all
    served by `DsaM1Layer` (empty: they are): `dsa_reasons`' deployment checks,
    made before the model is built."""
    import vllm.envs as envs
    from vllm.models.kimi_k3.amd.specialized_decode.runtime import TP
    from vllm.v1.attention.backends.registry import AttentionBackendEnum

    text = vllm_config.model_config.hf_text_config
    cache = vllm_config.cache_config.cache_dtype
    backend = vllm_config.attention_config.backend
    dims = ("kv_lora_rank", "qk_rope_head_dim", "qk_nope_head_dim", "v_head_dim")
    shapes = (
        getattr(text, "num_attention_heads", 0) // TP,
        *(getattr(text, n, None) for n in dims),
        getattr(text.dsa_config, "index_topk", None),
    )
    checks = (
        (not enabled(), "VLLM_KIMI_SD_DSA=off"),
        (seqs != 1, f"max_num_seqs {seqs} (one request a step: max_num_seqs 1)"),
        (
            backend not in (None, AttentionBackendEnum.ROCM_AITER_MLA_SPARSE),
            "the ROCM_AITER_MLA_SPARSE attention backend",
        ),
        (cache != "fp8_ds_mla", f"KV cache dtype {cache} (fp8_ds_mla only)"),
        (
            envs.VLLM_ROCM_USE_AITER
            and (envs.VLLM_ROCM_USE_AITER_FP4BMM or envs.VLLM_ROCM_USE_AITER_FP8BMM),
            "AITER's FP4 / FP8 MLA bmms (W_UV not bf16)",
        ),
        (shapes != DSA_SHAPES[:6], f"DSA shapes other than {DSA_SHAPES[:6]}"),
        (check_geometry and not fits(q), f"{q} rows a step"),
    )
    return [why for bad, why in checks if bad]


def prologue(w, positions, hidden, sd_q, sd_gate, md, kv_cache, slots) -> None:
    """VLLM's ops up to the sparse attention: `sd_q` [S, 12, 576] gets the bf16
    query `concat_mla_q` writes, `sd_gate` [S, 1536] the gate's sigmoid; the
    KV cache and the indexer's top-k are written as they write them."""
    a = w.mla_attn
    s = hidden.size(0)
    qkv_lora = w.fused_qkv_a_proj(hidden)[0]
    q_c, kv_lora = qkv_lora.split(
        [w.q_lora_rank, w.kv_lora_rank + w.qk_rope_head_dim], dim=-1
    )
    if not w.fuse_qkv_rmsnorm:
        q_c = w.q_a_layernorm(q_c)
    kv_c, k_pe = kv_lora.split([w.kv_lora_rank, w.qk_rope_head_dim], dim=-1)
    if w.fuse_qkv_rmsnorm:
        from vllm.models.common.ops import fused_q_kv_rmsnorm

        q_c, kv_c_normed = fused_q_kv_rmsnorm(
            q_c,
            kv_c,
            w.q_a_layernorm.weight.data,
            w.kv_a_layernorm.weight.data,
            w.q_a_layernorm.variance_epsilon,
        )
    else:
        kv_c_normed = w.kv_a_layernorm(kv_c)
    k_pe = k_pe.unsqueeze(1)
    q = w.q_b_proj(q_c)[0].view(-1, w.num_heads, w.qk_head_dim)
    if not w.skip_topk:
        w.indexer(hidden, q_c, positions, w.indexer_rope_emb)
    a.impl.record_logical_topk_ready()
    a.prepare_kv_cache_update(md)
    a.update_kv_cache(
        kv_c_normed, k_pe, kv_cache, slots, md, a.kv_cache_dtype, a._k_scale
    )
    nope, rope = q.split([w.qk_nope_head_dim, w.qk_rope_head_dim], dim=-1)
    torch.bmm(
        nope.transpose(0, 1),
        a.W_UK_T,
        out=sd_q[:s, :, : a.kv_lora_rank].transpose(0, 1),
    )
    sd_q[:s, :, a.kv_lora_rank :].copy_(rope)
    torch.sigmoid(w.g_proj(hidden)[0], out=sd_gate[:s])


def dsa_inputs(w, key, sd_q, kv_cache, md):
    """`dsa_attn`'s inputs (`host_args`') and new partials, o [splits, S, 12,
    512] and LSE [splits, S, 12]: the indexer's top-k and the request's block
    table row, folded to cache rows in the kernel."""
    from vllm.models.kimi_k3.amd.specialized_decode.mono.attention import dsa_attn

    a = w.mla_attn
    so, sl = dsa_attn.partial_shapes(key)
    assert md.topk_tokens == dsa_attn.TOPK
    rows, pitch = dsa_attn.kv_rows(kv_cache, md.block_size)
    dsa_in = dict(
        q=sd_q,
        kv=rows,
        iptr=md.paged_kv_indptr,
        scale=a.impl.scale,
        topk=a.impl.topk_indices_buffer,
        bt=md.block_table[0],
        bsr=pitch,
        bs=md.block_size,
    )
    po = sd_q.new_empty(so)
    return dsa_in, po, torch.empty(sl, dtype=torch.float32, device=sd_q.device)


def front_reasons(layer, runner: MonoRunner) -> list[str]:
    """Why `dsa_front` cannot run this layer's prologue (empty: it can)."""
    from vllm.models.kimi_k3.amd.dsa import KimiK3DSAIndexerKNorm

    w = layer.self_attn.mla_attn
    ix = w.indexer
    bf = torch.bfloat16
    shapes = (
        (w.fused_qkv_a_proj.weight, (2112, 7168)),
        (w.g_proj.weight, (1536, 7168)),
        (w.q_b_proj.weight, (2304, 1536)),
        (ix.wq_index.weight, (4096, 1536)),
        (ix.wk_index.weight, (128, 7168)),
        (ix.wv_index.weight, (32, 7168)),
        (w.mla_attn.W_UK_T, (12, 128, 512)),
    )
    checks = (
        (
            any(
                t is None or tuple(t.shape) != shape or t.dtype != bf
                for t, shape in shapes
            ),
            "weights other than bf16 at K3.1's shapes",
        ),
        (not getattr(ix, "use_fp4", False), "an FP8 indexer cache (MXFP4 only)"),
        (
            not isinstance(ix.k_index_norm, KimiK3DSAIndexerKNorm),
            "an indexer key norm other than the LayerNorm",
        ),
        (
            (ix.index_num_heads, ix.index_head_dim) != (32, 128),
            "indexer heads other than 32 x 128",
        ),
        (ix.dcp_scores is not None or w.skip_topk, "a shared or DCP top-k"),
        (w.indexer_rope_emb is not None, "the indexer's RoPE"),
        (layer.is_block_write_layer, "a block-write layer"),
        (runner.num_spec + 1 > 8, "more than 8 rows"),
    )
    return [why for bad, why in checks if bad]


def front_buffers(runner: MonoRunner, key, width: int, pages: int) -> dict:
    """The step's `dsa_front` buffers, one set a runner, each grown to the
    largest of the builds it serves (the scratch grows with `nblocks`, the rest
    with the rows and the block table's pages)."""
    from vllm.models.kimi_k3.amd.specialized_decode.mono.attention import dsa_front

    nb = max(
        dsa_front.scratch_bytes(dsa_front.DsaFrontBuild(**{**key, "delta": d}))
        for d in (False, True)
    )
    u8, f32 = torch.uint8, torch.float32
    want = {
        "scratch": ((nb,), u8),
        "iq_pay": ((width, 32, 64), u8),
        "iq_sc": ((width, 256), u8),
        "iq_w": ((width, 32), f32),
        "scores": ((width, pages * 64), f32),
    }
    dev = runner.epoch.device
    if runner.dsa_front is None:
        runner.dsa_front = {}
    bufs = runner.dsa_front
    for name, (shape, dtype) in want.items():
        old = bufs.get(name)
        if old is not None and all(o >= n for o, n in zip(old.shape, shape)):
            continue
        if old is not None:
            shape = tuple(max(o, n) for o, n in zip(old.shape, shape))
        if name == "scores":
            rows, ld = shape
            flat = torch.zeros(rows * ld + 16384, dtype=dtype, device=dev)
            bufs[name] = flat[: rows * ld].view(rows, ld)
        else:
            bufs[name] = torch.zeros(shape, dtype=dtype, device=dev)
    return bufs


class DsaFront:
    """A layer's `dsa_front` build and the runner's step buffers. Weights are
    views taken at every launch; W_UK's K-dense copy and the index key norm's
    bf16 weights are the attention's derived tensors, refilled in place by its
    post-load hook (`KimiK3DSAAttention.build_sd_front_weights`)."""

    def __init__(self, layer, runner: MonoRunner, width: int):
        w = layer.self_attn.mla_attn
        ix = w.indexer
        layer.self_attn.build_sd_front_weights()
        pages = -(-ix.max_model_len // 64)
        pages = -(-pages // 128) * 128
        self.kw = dict(
            tokens=width,
            nblocks=layer.prev_valid_blocks,
            eps=layer.self_attention_res_norm.variance_epsilon,
            out_eps=layer.input_layernorm.variance_epsilon,
            q_eps=w.q_a_layernorm.variance_epsilon,
            kv_eps=w.kv_a_layernorm.variance_epsilon,
            k_eps=ix.k_index_norm.eps,
            wscale=float(
                torch.tensor(
                    ix.indexer_weight_scale * ix.index_softmax_scale,
                    dtype=torch.float32,
                )
            ),
            max_pages=pages,
        )
        self.bufs = front_buffers(runner, self.kw, width, pages)
        self.pages = pages

    def key(self, delta: bool):
        from vllm.models.kimi_k3.amd.specialized_decode.mono.attention import (
            dsa_front,
        )

        return dsa_front.DsaFrontBuild(delta=delta, **self.kw)

    def step_reasons(self, ix, im, kv_cache, s: int) -> list[str]:
        """Why this step runs vLLM's prologue (fixed at capture)."""
        if im is None or im.decode is None or im.block_table is None:
            return ["no indexer decode metadata"]
        dec, bt, cache = im.decode, im.block_table, ix.k_cache.kv_cache
        ends = dec.seq_lens.reshape(-1)
        checks = (
            (
                im.num_prefills or im.num_decode_tokens != s,
                "indexer rows other than the step's decode rows",
            ),
            (
                dec.query_start_loc is None
                or dec.requires_padding
                or ends.numel() != s
                or ends.dtype != torch.int32,
                "padded decode rows",
            ),
            (bt.dtype != torch.int32 or bt.stride(1) != 1, "a block table layout"),
            (bt.shape[1] > self.pages, "a block table past max_model_len"),
            (
                im.slot_mapping.dtype != torch.int64 or im.slot_mapping.numel() != s,
                "indexer slots",
            ),
            (
                cache.dtype != torch.uint8
                or tuple(cache.shape[1:]) != (64, 68)
                or cache.stride()[1:] != (68, 1),
                "an indexer cache not MXFP4 [pages, 64, 68]",
            ),
            (kv_cache.view(torch.uint8).stride(-1) != 1, "a strided KV record"),
        )
        return [why for bad, why in checks if bad]

    def launch(
        self, layer, runner, prefix, blocks, delta, sd_q, gate, kv_cache, slots, im
    ) -> None:
        from vllm.models.kimi_k3.amd.specialized_decode.mono.attention import (
            dsa_front,
        )
        from vllm.models.kimi_k3.amd.specialized_decode.mono.attention.kda import (
            HIDDEN,
        )

        w = layer.self_attn.mla_attn
        ix = w.indexer
        attn = layer.self_attn
        s = prefix.size(0)
        b = self.bufs
        args = runner.launch_args()
        args["scratch"] = b["scratch"]
        dsa_front.dsa_front(
            self.key(delta is not None),
            prefix=prefix,
            delta=delta,
            blocks=blocks,
            ares_nw=layer.self_attention_res_norm.weight,
            ares_qk=layer.self_attention_res_proj.weight.view(HIDDEN),
            in_nw=layer.input_layernorm.weight,
            w_qkv=w.fused_qkv_a_proj.weight,
            w_g=w.g_proj.weight,
            w_wk=ix.wk_index.weight,
            w_wv=ix.wv_index.weight,
            q_nw=w.q_a_layernorm.weight,
            kv_nw=w.kv_a_layernorm.weight,
            w_qb=w.q_b_proj.weight,
            w_iq=ix.wq_index.weight,
            w_uk=attn._w_uk_kdense,
            ln_w=attn._k_norm_bf16[0],
            ln_b=attn._k_norm_bf16[1],
            kv_cache=kv_cache.view(torch.uint8),
            idx_cache=ix.k_cache.kv_cache,
            slots=slots[:s],
            islots=im.slot_mapping,
            gate_out=gate,
            q_out=sd_q,
            iq_pay=b["iq_pay"],
            iq_sc=b["iq_sc"],
            iq_w=b["iq_w"],
            idx_bt=im.block_table[0],
            row_ends=im.decode.seq_lens.reshape(-1),
            scores=b["scores"],
            topk_out=ix.topk_indices_buffer,
            layer=layer.layer_idx,
            **args,
        )
        w.mla_attn.impl.record_logical_topk_ready()


class DsaM1Layer(MlaOpsLayer):
    """A DSA layer: the prologue (vLLM's ops or `dsa_front`), then K2 running
    `dsa_attn`'s slices first, merging them (W_UV and the gate in its M1b), then
    o_proj and the MoE."""

    def __init__(self, layer, runner: MonoRunner, width: int):
        super().__init__(layer, runner, width)
        self.wrapper = layer.self_attn.mla_attn
        self.key = dsa_key(width)
        self.front = None
        if enabled():
            why = front_reasons(layer, runner)
            if why:
                logger.info_once(
                    "Kimi-K3 specialized decode: the DSA layers run vLLM's "
                    "prologue (%s)",
                    "; ".join(why),
                )
            else:
                self.front = DsaFront(layer, runner, width)

    def builds(self) -> list:
        """Every launch this layer can make: the DSA K2, the front at either
        delta, and `MlaOpsLayer`'s for the steps it hands to vLLM's ops."""
        k2 = _k2_key(self.layer, self.runner, self.width, False, True, self.key)
        fronts = (
            [] if self.front is None else [self.front.key(d) for d in (True, False)]
        )
        return [*fronts, k2, *super().builds()]

    def launch(self, step: "Step", positions, prefix_sum, block_residual, prefix_delta):
        from vllm.models.kimi_k3.amd.linear import _apply_attn_res
        from vllm.models.kimi_k3.amd.specialized_decode.mono.attention import (
            dsa_attn,
        )
        from vllm.models.kimi_k3.amd.specialized_decode.runtime import mla

        layer, w, runner = self.layer, self.wrapper, self.runner
        a = w.mla_attn
        md, _, kv_cache, slots = mla.context(w)
        s = prefix_sum.size(0)
        if (
            md.num_prefills
            or md.num_decodes != 1
            or md.num_decode_tokens != s
            or md.num_actual_tokens != s
        ):
            return super().launch(
                step, positions, prefix_sum, block_residual, prefix_delta
            )
        if self.first:
            runner.step_begin()
        sd_q = prefix_sum.new_empty(s, dsa_attn.NH, dsa_attn.ENTRY)
        gate = prefix_sum.new_empty(s, dsa_attn.NH * a.v_head_dim)
        im = None
        if self.front is not None:
            im = step.attn_metadata.get(w.indexer.k_cache.prefix)
            why = self.front.step_reasons(w.indexer, im, kv_cache, s)
            if why:
                logger.info_once(
                    "Kimi-K3 specialized decode: a DSA step runs vLLM's prologue (%s)",
                    "; ".join(why),
                )
                im = None
        if im is not None:
            assert self.front is not None
            a.prepare_kv_cache_update(md)
            self.front.launch(
                layer,
                runner,
                prefix_sum,
                block_residual,
                prefix_delta,
                sd_q,
                gate,
                kv_cache,
                slots,
                im,
            )
        else:
            hidden_states = _apply_attn_res(
                prefix_sum,
                block_residual,
                layer.self_attention_res_proj,
                layer.self_attention_res_norm,
                layer.prev_valid_blocks,
                delta=prefix_delta,
                output_norm=layer.input_layernorm,
                quant_key=layer.self_attn.get_input_quant_key(),
            )
            prologue(w, positions, hidden_states, sd_q, gate, md, kv_cache, slots)
        dsa_in, po, pl = dsa_inputs(w, self.key, sd_q, kv_cache, md)
        core = torch.empty_like(gate)
        m1b_in = dict(
            o=po[0],
            lse=pl[0],
            seq_lens=md.seq_lens,
            qsl=md.query_start_loc,
            w_v=a.W_UV.transpose(1, 2),
            w_v_scale=None,
            gate=gate,
        )
        out = _k2(
            layer,
            runner,
            core,
            prefix_sum,
            block_residual,
            m1b_in=m1b_in,
            dsa=self.key,
            dsa_in=dsa_in,
        )
        return prefix_sum, block_residual, out
