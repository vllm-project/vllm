# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kimi-K3.1's DSA layers at DCP1 (``VLLM_KIMI_SD_DSA``): vLLM's query, KV and
indexer ops, then the sparse attention split-K -- ``split``: as ``dsa_attn``'s
own launch; ``fused`` (``auto``): in K2's launch, on its top CTAs -- and K2,
which merges its slices, runs W_UV and the gate (its fused M1b), then o_proj
and the MoE. ``VLLM_KIMI_SD_DSA_FOLD`` (fused): the attention reads the
indexer's logical top-k and the block table itself (no converter launch).

``prologue`` is ``MultiHeadLatentAttentionWrapper.forward`` and
``MLAAttention.forward`` up to the attention, op for op, writing the attention's
bf16 query (``concat_mla_q``'s rows) and the gate's sigmoid instead;
``convert_indices`` is ``forward_mqa``'s top-k remap. Pure decode steps of one
request only (the layer runs vLLM's ops otherwise).

``VLLM_KIMI_SD_DSA_PROLOGUE``: the AttnRes seam and ``prologue``'s projections,
norms and KV write run as one launch (D1a, ``dsa_pre``), then the indexer's
scoring and top-k (``indexer_tail``, vLLM's op).
"""

from typing import TYPE_CHECKING

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

OFF, SPLIT, FUSED, AUTO = "off", "split", "fused", "auto"
SPLITS = (16, 32)
# the shapes ``dsa_attn`` is written for: heads a rank, kv_lora_rank, rope,
# nope, v head dims, the indexer's top-k, the fp8_ds_mla record's bytes
DSA_SHAPES = (12, 512, 64, 128, 128, 2048, 656)


def mode() -> str:
    """``VLLM_KIMI_SD_DSA``: off, split, fused or auto."""
    import vllm.envs as envs

    return envs.VLLM_KIMI_SD_DSA


def fused() -> bool:
    """The attention in K2's launch: ``fused``, and ``auto``'s pick."""
    return mode() in (FUSED, AUTO)


def dsa_key(runner: MonoRunner, width: int):
    """The sparse attention's build at this width (one request: width rows):
    its own launch (``split``) or K2's stage (``fused``, with the fold)."""
    import vllm.envs as envs
    from vllm.models.kimi_k3.amd.specialized_decode.mono.attention import dsa_attn

    stage = fused()
    return dsa_attn.DsaAttnBuild(
        tokens=width,
        splits=envs.VLLM_KIMI_SD_DSA_SPLITS,
        chunk=dsa_attn.TOPK // envs.VLLM_KIMI_SD_DSA_SPLITS,
        fold=stage and envs.VLLM_KIMI_SD_DSA_FOLD,
        standalone=not stage,
        part_fp32=envs.VLLM_KIMI_SD_DSA_PART_FP32,
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


def _desc(t) -> str:
    if t is None:
        return "(none)"
    return f"{tuple(t.shape)} {t.dtype} strides {t.stride()}"


def dsa_reasons(runner: MonoRunner, layer) -> list[str]:
    """Why this DSA layer cannot run ``DsaM1Layer`` (empty: it can). The
    deployment's are checked at startup (``runtime.unsupported``), the steps'
    in ``plan`` and ``DsaM1Layer.launch``."""
    import vllm.envs as envs

    m = mode()
    if m == OFF:
        return ["VLLM_KIMI_SD_DSA=off"]
    w = getattr(layer.self_attn, "mla_attn", None)
    a = getattr(w, "mla_attn", None)
    if a is None or w.indexer is None or not w.is_sparse:
        return ["a layer without DSA's sparse MLA"]
    shapes = layer_shapes(layer)
    topk = getattr(a.impl, "topk_indices_buffer", None)
    # the fold reads the top-k rows in place: int32, TOPK wide and dense
    rows_ok = (
        topk is not None
        and topk.dtype == torch.int32
        and topk.dim() == 2
        and topk.size(-1) == DSA_SHAPES[5]
        and topk.stride() == (DSA_SHAPES[5], 1)
    )
    checks = (
        (
            fused() and envs.VLLM_KIMI_SD_DSA_FOLD and not rows_ok,
            f"a top-k buffer {_desc(topk)} not [tokens, {DSA_SHAPES[5]}] dense "
            "int32 (VLLM_KIMI_SD_DSA_FOLD=0 converts it)",
        ),
        (
            envs.VLLM_KIMI_SD_DSA_SPLITS not in SPLITS,
            f"VLLM_KIMI_SD_DSA_SPLITS={envs.VLLM_KIMI_SD_DSA_SPLITS} (16 or 32)",
        ),
        (runner.dcp != 1, f"DCP{runner.dcp} (DCP1 only)"),
        (a.kv_cache_dtype != "fp8_ds_mla", f"a {a.kv_cache_dtype} KV cache"),
        (
            shapes != DSA_SHAPES,
            f"DSA shapes {shapes} (heads, kv_lora_rank, rope, nope, v, top-k, "
            f"record bytes; {DSA_SHAPES} only)",
        ),
        # W_UV as the checkpoint's bf16 (``uv_bf16``), the query vLLM's bf16 bmm
        (
            a.is_aiter_triton_fp4_bmm_enabled
            or a.is_aiter_triton_fp8_bmm_enabled
            or getattr(a, "is_amx_bmm_enabled", False),
            "AITER's FP4 / FP8 MLA bmms (W_UV not bf16)",
        ),
        (a.q_pad_num_heads is not None, "a padded query (q_pad_num_heads)"),
        (w.q_lora_rank is None, "MLA without q-LoRA"),
        (w.g_proj is None, "MLA without the output gate (g_proj)"),
        (a.hisparse_cache is not None, "HiSparse"),
        (a.use_pcp, "prefill context parallelism"),
        (w.dcp_q_replicate, "DCP query replication"),
        (getattr(a.impl, "sinks", None) is not None, "attention sinks"),
    )
    return [why for bad, why in checks if bad]


def width_served(runner: MonoRunner, width: int) -> bool:
    """One request a step (``dsa_attn``'s rows), its slices on resident CTAs."""
    from vllm.models.kimi_k3.amd.specialized_decode.mono.attention import dsa_attn

    if width != runner.num_spec + 1:
        return False
    try:
        dsa_attn.check(dsa_key(runner, width))
    except AssertionError:
        return False
    return True


def prologue(w, positions, hidden, sd_q, sd_gate, md, kv_cache, slots) -> None:
    """``MultiHeadLatentAttentionWrapper.forward`` up to ``self.mla_attn`` and
    ``MLAAttention.forward`` (direct call) / ``forward_impl`` up to
    ``forward_mqa``'s attention, with the same ops: ``sd_q`` [S, 12, 576] gets
    the bf16 query ``concat_mla_q`` writes, ``sd_gate`` [S, 1536] the gate's
    sigmoid. Writes the KV cache and the indexer's top-k as they do."""
    a = w.mla_attn
    s = hidden.size(0)
    assert w.q_lora_rank is not None and not w.dcp_q_replicate
    assert a.hisparse_cache is None and not a.use_pcp
    assert md.num_prefills == 0 and md.num_decode_tokens == s
    assert md.num_actual_tokens == s
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
    if w.rotary_emb is not None:
        q[..., w.qk_nope_head_dim :], k_pe = w.rotary_emb(
            positions, q[..., w.qk_nope_head_dim :], k_pe
        )
    if not w.skip_topk:
        w.indexer(hidden, q_c, positions, w.indexer_rope_emb)
    a.impl.record_logical_topk_ready()
    a.prepare_kv_cache_update(md)
    a.update_kv_cache(
        kv_c_normed, k_pe, kv_cache, slots, md, a.kv_cache_dtype, a._k_scale
    )
    # forward_impl's W_UK bmm into the query's latent columns, the rope beside
    # it (concat_mla_q's copy)
    nope, rope = q.split([w.qk_nope_head_dim, w.qk_rope_head_dim], dim=-1)
    torch.bmm(
        nope.transpose(0, 1),
        a.W_UK_T,
        out=sd_q[:s, :, : a.kv_lora_rank].transpose(0, 1),
    )
    sd_q[:s, :, a.kv_lora_rank :].copy_(rope)
    torch.sigmoid(w.g_proj(hidden)[0], out=sd_gate[:s])


def convert_indices(a, md, kv_cache, s: int) -> torch.Tensor:
    """``forward_mqa``'s top-k remap at DCP1: the step's flat cache rows in
    ``md.paged_kv_indices`` (row t's at ``md.paged_kv_indptr[t]``), remapped
    once per indexer write as it does."""
    from vllm.v1.attention.backends.mla.rocm_aiter_mla_sparse import (
        fit_kpool_indices_to_aiter,
        triton_convert_req_index_to_global_index,
    )

    impl = a.impl
    _, block_stride_rows = _flat(kv_cache, md.block_size)
    if (
        impl.owns_indexer
        or md.remapped_epoch != impl.index_epoch
        or md.remapped_buf is not impl.topk_indices_buffer
    ):
        topk_indices = fit_kpool_indices_to_aiter(
            impl.topk_indices_buffer[:s], md.topk_tokens
        )
        triton_convert_req_index_to_global_index(
            md.req_id_per_token[:s],
            md.block_table,
            topk_indices,
            md.paged_kv_indptr,
            md.paged_kv_indices,
            BLOCK_SIZE=md.block_size,
            BLOCK_STRIDE_ROWS=block_stride_rows,
            NUM_TOPK_TOKENS=md.topk_tokens,
        )
        md.remapped_buf = impl.topk_indices_buffer
        md.remapped_epoch = impl.index_epoch
    return md.paged_kv_indices


def _flat(kv_cache, block_size: int):
    from vllm.v1.attention.backends.mla.sparse_utils import flat_kv_row_view

    return flat_kv_row_view(kv_cache, block_size)


def dsa_inputs(w, key, sd_q, kv_cache, md, idx):
    """``dsa_attn``'s inputs (``host_args``') and new partials, o
    [splits, S, 12, 512] and LSE [splits, S, 12] (M1b merges the splits).
    ``idx`` the converter's rows, or with the fold None: the indexer's top-k
    and the request's block table row (one request a step)."""
    from vllm.models.kimi_k3.amd.specialized_decode.mono.attention import dsa_attn

    a = w.mla_attn
    so, sl = dsa_attn.partial_shapes(key)
    dsa_in = dict(
        q=sd_q,
        kv=dsa_attn.kv_rows(kv_cache, md.block_size),
        iptr=md.paged_kv_indptr,
        scale=a.impl.scale,
    )
    if key.fold:
        # the engine sets the block size after the builds: a launch argument
        assert md.topk_tokens == dsa_attn.TOPK
        dsa_in.update(
            topk=a.impl.topk_indices_buffer,
            bt=md.block_table[0],
            bsr=_flat(kv_cache, md.block_size)[1],
            bs=md.block_size,
        )
    else:
        dsa_in.update(idx=idx)
    dt = torch.float32 if key.part_fp32 else torch.bfloat16
    po = sd_q.new_empty(so, dtype=dt)
    return dsa_in, po, torch.empty(sl, dtype=torch.float32, device=sd_q.device)


def prologue_on() -> bool:
    """``VLLM_KIMI_SD_DSA_PROLOGUE``: D1a in place of ``prologue``."""
    import vllm.envs as envs

    return envs.VLLM_KIMI_SD_DSA_PROLOGUE


def indexer_prep_on(w) -> bool:
    """D1a runs the MXFP4 indexer's prep (``VLLM_KIMI_SD_DSA_PROLOGUE_INDEXER``
    with ``VLLM_K3_FUSED_INDEXER_PREP``): a layer that runs its own 2048-wide top-k
    with 32 x 128 heads."""
    import vllm.envs as envs
    from vllm.models.kimi_k3.amd.specialized_decode.mono.attention import dsa_pre

    ind = w.indexer
    return (
        envs.VLLM_KIMI_SD_DSA_PROLOGUE_INDEXER
        and getattr(ind, "fused_prep", False)
        and not w.skip_topk
        and ind.index_num_heads == dsa_pre.IVN
        and ind.index_head_dim == dsa_pre.IKD
        and ind.dcp_scores is None
        and ind.topk_indices_buffer is not None
        and ind.topk_indices_buffer.shape[1] == dsa_pre.TOPK
    )


def pre_weights(w) -> dict:
    """D1a's weights: views of the loaded parameters; the indexer key norm's
    bf16 weight / bias (``KimiK3DSAIndexerKNorm``'s cached copies, or made)."""
    from vllm.models.kimi_k3.amd.specialized_decode.runtime import mla

    a, ind = w.mla_attn, w.indexer
    w_qkv, w_g = mla.qkv_gate_weights(w)
    kn = ind.k_index_norm
    nw, nb = getattr(kn, "weight_bf16", None), getattr(kn, "bias_bf16", None)
    if nw is None or nw.dtype != torch.bfloat16:
        nw, nb = kn.weight.to(torch.bfloat16), kn.bias.to(torch.bfloat16)
        # D1a bypasses the norm's forward, which fills the cache: fill it here
        # (``prologue_reasons`` runs this at init), else every captured step
        # replays the two casts. Never from a capture: its pool reuses them.
        if not torch.cuda.is_current_stream_capturing():
            kn.weight_bf16, kn.bias_bf16 = nw, nb
    return dict(
        w_qkv=w_qkv,
        w_g=w_g,
        w_ik=ind.wk_index.weight,
        w_iv=ind.wv_index.weight,
        q_nw=w.q_a_layernorm.weight,
        kv_nw=w.kv_a_layernorm.weight,
        ik_nw=nw,
        ik_nb=nb,
        w_qb=w.q_b_proj.weight,
        w_iq=ind.wq_index.weight,
        w_uk=a.W_UK_T,
    )


def prologue_reasons(layer, kv_cache=None) -> list[str]:
    """Why D1a cannot run this DSA layer's prologue (empty: it can)."""
    from vllm.models.kimi_k3.amd.dsa import KimiK3DSAIndexerKNorm
    from vllm.models.kimi_k3.amd.specialized_decode.mono.attention import dsa_pre

    w = layer.self_attn.mla_attn
    ind = w.indexer
    why = []
    if not getattr(ind, "use_fp4", False):
        why.append("an FP8 indexer cache (MXFP4 only)")
    if not isinstance(getattr(ind, "k_index_norm", None), KimiK3DSAIndexerKNorm):
        why.append("an indexer key norm other than the LayerNorm")
    if w.rotary_emb is not None or w.indexer_rope_emb is not None:
        why.append("RoPE (NoPE only)")
    if w.g_proj is None:
        why.append("no output gate")
    if why:
        return why
    t = pre_weights(w)
    t.pop("q_nw"), t.pop("kv_nw")
    if kv_cache is None:
        kv_cache = torch.empty(1, 1, dsa_pre.REC, dtype=torch.uint8)
    return dsa_pre.check_inputs(kv_cache=kv_cache, **t)


def indexer_tail(w, iq, ik, iv, prep=None) -> None:
    """``KimiK3DSAIndexer.forward`` past its projections (D1a's ``iq``, ``ik``,
    ``iv``): ``kimi_fp4_indexer`` (the key store, scoring, top-k). ``prep``:
    D1a's MXFP4 query, scales and head weights (it stored the keys and reset
    the top-k rows), leaving the op its scoring and top-k."""
    from vllm.distributed import get_dcp_group

    ind = w.indexer
    dcp = get_dcp_group()
    torch.ops.vllm.kimi_fp4_indexer(
        iq.view(-1, ind.index_num_heads, ind.index_head_dim),
        ik,
        iv,
        ind.k_cache.kv_cache,
        ind.topk_indices_buffer,
        ind.k_cache.prefix,
        ind.max_model_len,
        dcp.rank_in_group,
        dcp.world_size,
        ind.cp_interleave,
        ind.indexer_weight_scale * ind.index_softmax_scale,
        ind.dcp_scores,
        # D1a already applied the key LayerNorm (no k_norm). fused_prep only with
        # D1a's prep: without it, the op's unfused prep as before D1a ran it.
        prep is not None,
        None,
        None,
        0.0,
        *(prep if prep is not None else (None, None, None)),
    )


class DsaM1Layer(MlaOpsLayer):
    """A DSA layer: vLLM's query, KV and indexer ops (``prologue``), the top-k
    remap, ``dsa_attn``'s slices, then K2 (the slices merged, W_UV and the gate
    in its M1b, then o_proj and the MoE)."""

    def __init__(self, layer, runner: MonoRunner, width: int):
        super().__init__(layer, runner, width)
        self.wrapper = layer.self_attn.mla_attn
        self.key = dsa_key(runner, width)
        self._w_v: torch.Tensor | None = None
        self.pre = prologue_on()
        if self.pre:
            why = prologue_reasons(layer)
            if why:
                raise ValueError(
                    "Kimi-K3 specialized decode: VLLM_KIMI_SD_DSA_PROLOGUE=1 does "
                    "not support: " + "; ".join(why)
                )
        self.idx = self.pre and indexer_prep_on(self.wrapper)
        if self.pre:
            logger.info_once(
                "Kimi-K3 specialized decode: D1a runs the MXFP4 indexer prep: %s",
                self.idx,
            )

    def _pre_key(self, delta: bool):
        from vllm.models.kimi_k3.amd.specialized_decode.mono.attention import dsa_pre

        layer, w = self.layer, self.wrapper
        return dsa_pre.DsaPreBuild(
            tokens=self.width,
            nblocks=layer.prev_valid_blocks,
            delta=delta,
            eps=layer.self_attention_res_norm.variance_epsilon,
            out_eps=layer.input_layernorm.variance_epsilon,
            q_eps=w.q_a_layernorm.variance_epsilon,
            kv_eps=w.kv_a_layernorm.variance_epsilon,
            k_eps=w.indexer.k_index_norm.eps,
            idx=self.idx,
            iw_scale=(
                w.indexer.indexer_weight_scale * w.indexer.index_softmax_scale
                if self.idx
                else 1.0
            ),
        )

    def builds(self) -> list:
        k2 = _k2_key(self.layer, self.runner, self.width, False, True, self.key)
        out = [k2] if not self.key.standalone else [self.key, k2]
        # a layer after another always has a pending delta
        return [self._pre_key(True), *out] if self.pre else out

    def w_v(self) -> torch.Tensor:
        """W_UV as M1b's bf16 [12, 128 v, 512 k], k dense: a view of vLLM's
        [12, 512, 128] where its strides allow, else a copy (made once)."""
        if self._w_v is None:
            w_v = self.wrapper.mla_attn.W_UV.transpose(1, 2)
            if w_v.stride(-1) != 1 or w_v.data_ptr() % 16:
                w_v = w_v.contiguous()
            self._w_v = w_v
        return self._w_v

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
        if md.num_prefills or md.num_decode_tokens != s or md.num_actual_tokens != s:
            # not the pure one-request step the kernels serve: vLLM's ops
            return super().launch(
                step, positions, prefix_sum, block_residual, prefix_delta
            )
        if self.first:
            runner.step_begin()
        sd_q = prefix_sum.new_empty(s, dsa_attn.NH, dsa_attn.ENTRY)
        gate = prefix_sum.new_empty(s, dsa_attn.NH * a.v_head_dim)
        hidden_states = None
        if self.pre:
            self.launch_pre(prefix_sum, block_residual, prefix_delta, sd_q, gate,
                            md, kv_cache, slots)  # fmt: skip
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
        idx = None if self.key.fold else convert_indices(a, md, kv_cache, s)
        dsa_in, po, pl = dsa_inputs(w, self.key, sd_q, kv_cache, md, idx)
        if self.key.standalone:
            dsa_attn.dsa_attn_launch(self.key, part_o=po, part_lse=pl, **dsa_in)
        core = torch.empty_like(gate)
        m1b_in = dict(
            o=po[0],
            lse=pl[0],
            # read at DCP8 only (the push); one rank here
            seq_lens=md.query_start_loc,
            qsl=md.query_start_loc,
            w_v=self.w_v(),
            w_v_scale=None,
            gate=gate,
        )
        out = _k2(
            layer, runner, core, prefix_sum, block_residual, m1b_in=m1b_in,
            dsa=self.key, dsa_in=None if self.key.standalone else dsa_in,
        )  # fmt: skip
        if (
            verify_on()
            and hidden_states is not None
            and not torch.cuda.is_current_stream_capturing()
        ):
            self.verify(positions, hidden_states, sd_q, core, kv_cache, md, slots)
        return prefix_sum, block_residual, out

    def launch_pre(self, prefix, blocks, delta, sd_q, gate, md, kv_cache, slots):
        """D1a (``dsa_pre``): AttnRes, the projections, norms and the KV write,
        writing ``sd_q`` and ``gate`` as ``prologue`` does; then the indexer's
        scoring and top-k (vLLM's op) on its projections."""
        from vllm.forward_context import get_forward_context
        from vllm.models.kimi_k3.amd.specialized_decode.mono.attention import dsa_pre
        from vllm.models.kimi_k3.amd.specialized_decode.mono.attention.kda import (
            HIDDEN,
        )

        layer, w, runner = self.layer, self.wrapper, self.runner
        a = w.mla_attn
        s = prefix.size(0)
        assert a.hisparse_cache is None and not a.use_pcp and not w.dcp_q_replicate
        iq = prefix.new_empty(s, dsa_pre.IQN)
        ik = prefix.new_empty(s, dsa_pre.IKD)
        iv = prefix.new_empty(s, dsa_pre.IVN)
        idx, prep = {}, None
        if self.idx:
            ind = w.indexer
            ik_cache = ind.k_cache.kv_cache
            islots = get_forward_context().attn_metadata[ind.k_cache.prefix]
            islots = islots.slot_mapping[:s]
            why = dsa_pre.idx_reasons(ik_cache, islots, ind.topk_indices_buffer, s)
            assert not why, why
            u8 = torch.uint8
            prep = (
                torch.empty(s, dsa_pre.IVN, dsa_pre.IKD // 2, dtype=u8,
                            device=prefix.device),
                torch.empty(s, 1, 4, 16, 4, dtype=u8, device=prefix.device),
                torch.empty(s, dsa_pre.IVN, dtype=torch.float32,
                            device=prefix.device),
            )  # fmt: skip
            idx = dict(
                ik_cache=ik_cache,
                islots=islots,
                topk=ind.topk_indices_buffer,
                iq_pay=prep[0],
                iq_sc=prep[1],
                iw_out=prep[2],
            )
        dsa_pre.dsa_pre(
            self._pre_key(delta is not None),
            prefix=prefix,
            delta=delta,
            blocks=blocks,
            ares_nw=layer.self_attention_res_norm.weight,
            ares_qk=layer.self_attention_res_proj.weight.view(HIDDEN),
            in_nw=layer.input_layernorm.weight,
            **pre_weights(w),
            kv_cache=kv_cache,
            slots=slots[:s],
            gate_out=gate,
            q_out=sd_q,
            iq_out=iq,
            ik_out=ik,
            iv_out=iv,
            layer=layer.layer_idx,
            **runner.launch_args(),
            **idx,
        )
        if not w.skip_topk:
            indexer_tail(w, iq, ik, iv, prep)
        a.impl.record_logical_topk_ready()
        a.prepare_kv_cache_update(md)

    def verify(self, positions, hidden, sd_q, core, kv_cache, md, slots) -> None:
        """Debug (``VLLM_KIMI_SD_DSA_VERIFY``, eager steps): reruns vLLM's ops on
        this layer's input, twice (the second run vLLM's own run-to-run floor),
        and logs how the launch's query, KV write, top-k and o_proj input
        differ."""
        from vllm.models.kimi_k3.amd.specialized_decode.mono.attention import (
            dsa_attn,
        )

        w = self.wrapper
        impl = w.mla_attn.impl
        s = hidden.size(0)
        rows = dsa_attn.kv_rows(kv_cache, md.block_size)
        pitch = _flat(kv_cache.view(torch.uint8), md.block_size)[1]
        sl = slots[:s].long()
        at = sl // md.block_size * pitch + sl % md.block_size
        kv, topk = rows[at].clone(), impl.topk_indices_buffer[:s].clone()
        q = sd_q[:s].clone()

        def stock():
            out = w.forward(positions, hidden, skip_o_proj=True)
            q_ref = impl.q_concat_buffer[:s].reshape(q.shape).clone()
            return out, q_ref, rows[at].clone(), impl.topk_indices_buffer[:s].clone()

        def diff(x, y):
            x, y = x.float(), y.float()
            return f"rel_l2 {float((x - y).norm() / y.norm().clamp_min(1e-30)):.3e}"

        # vLLM's ops twice: the second run is their own run-to-run floor
        r1, r2 = stock(), stock()
        self.checks = getattr(self, "checks", 0) + 1
        if self.checks <= 4 or self.checks % 256 == 0:
            logger.info(
                "Kimi-K3 DSA verify layer %d step %d: launch vs vLLM: q %s %s, "
                "kv %s, top-k %s, core %s; vLLM vs vLLM: q %s %s, kv %s, "
                "top-k %s, core %s",
                self.layer.layer_idx,
                self.checks,
                torch.equal(q, r1[1]),
                diff(q, r1[1]),
                torch.equal(kv, r1[2]),
                torch.equal(topk, r1[3]),
                diff(core, r1[0]),
                torch.equal(r2[1], r1[1]),
                diff(r2[1], r1[1]),
                torch.equal(r2[2], r1[2]),
                torch.equal(r2[3], r1[3]),
                diff(r2[0], r1[0]),
            )


def verify_on() -> bool:
    import vllm.envs as envs

    return envs.VLLM_KIMI_SD_DSA_VERIFY
