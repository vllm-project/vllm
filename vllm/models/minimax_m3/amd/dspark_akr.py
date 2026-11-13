# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The amd_kernels ``draft`` route (gfx950, TP4, the draft not torch.compiled): the
MiniMax-M3.1 DSpark draft's attention, context K/V, MXFP8 linears and shared lm_head;
``dspark_akr_fused`` runs the rest of its forward, ``sample_akr`` its sampling."""

import time
from typing import Any

import torch

from vllm.logger import init_logger
from vllm.model_executor.layers.attention.attention import get_attention_context
from vllm.model_executor.layers.vocab_parallel_embedding import (
    UnquantizedEmbeddingMethod,
)
from vllm.models.minimax_m3.amd import akr_attn, akr_config

logger = init_logger(__name__)

_HEAD_DIM = 128
_V = 200064


def attach(model: Any) -> None:
    """From ``MiniMaxM3DSparkForCausalLM.__init__``, before the weights load."""
    inner = model.model
    if not akr_config.on("draft"):
        return
    if not inner.do_not_compile:
        logger.warning("amd_kernels draft stays off: the draft is torch.compiled")
        return
    from vllm.model_executor.kernels.linear.mxfp8 import rocm_akr

    for layer in inner.layers:
        a, mlp = layer.self_attn, layer.mlp
        a._akr_query = _query_ok(a)
        for lin in (a.qkv_proj, a.o_proj, mlp.gate_up_proj, mlp.down_proj):
            rocm_akr.mark_akr_linear(lin)
    rocm_akr.mark_akr_linear(inner._context_kv_proj)
    model.akr_setup = lambda spec, target: setup(spec, model, target)
    logger.info("amd_kernels draft: the DSpark draft runs amd_kernels")


def setup(spec, model: torch.nn.Module, target_model: torch.nn.Module) -> None:
    """The speculator's ``akr_setup`` hook (every TP rank, after ``load_model``, before
    any capture): the shared lm_head's layout, the fused forward, the sampling."""
    from vllm.models.minimax_m3.amd import dspark_akr_fused, sample_akr

    switch_shared_lm_head(model.lm_head, model.logits_processor, [model, target_model])
    fused = dspark_akr_fused.setup(model, spec.max_num_reqs * spec.num_query_per_req)
    if fused is not None:
        model.model._akr_fused = fused
        model.akr_fc_into = fused.fc_into
        model.akr_draft_capture = fused.capture
        spec._aux_staging = None  # the input fc runs eagerly before each replay
    sample_akr.setup(spec, model, target_model)


# ---------------------------------------------------------------- the query block
def _query_ok(attn: torch.nn.Module) -> bool:
    """The library's layer shape and RoPE, an fp8 KV cache under TRITON_ATTN (the
    layout ``[pages, KV heads, page, 256]``)."""
    from amd_kernels.attention.norm_rope_fp8_insert_d128_rot64 import rope_blocker
    from amd_kernels.attention.swa_decode_fp8kv_gqa16_d128 import layer_blocker

    inner, rope = attn.attn, attn.rotary_emb
    whys = (
        layer_blocker(
            attn.head_dim, attn.num_heads, attn.num_kv_heads, attn.sliding_window
        ),
        rope_blocker(rope.rotary_dim, rope.is_neox_style, rope.cos_sin_cache),
    )
    reasons = [why for why in whys if why is not None]
    if not str(getattr(inner, "kv_cache_dtype", "auto")).startswith("fp8"):
        reasons.append(f"kv cache {getattr(inner, 'kv_cache_dtype', None)}")
    if inner.attn_backend.get_name() != "TRITON_ATTN":
        reasons.append(f"backend {inner.attn_backend.get_name()}")
    if getattr(attn, "v_scale", None) is not None:
        reasons.append("v_scale")
    if reasons:
        logger.warning_once(
            "amd_kernels draft: the draft attention stays stock (%s)",
            ", ".join(reasons),
        )
    return not reasons


def attention_forward(
    attn: torch.nn.Module, qkv: torch.Tensor, positions: torch.Tensor
) -> torch.Tensor:
    """``MiniMaxM3DSparkAttention.forward`` after its qkv_proj: q/k-norm + RoPE + fp8
    K/V insert, the sliding-window attention over the fp8 cache, o_proj."""
    from amd_kernels.attention.norm_rope_fp8_insert_d128_rot64.qkv_norm_rope_insert import (  # noqa: E501
        qkv_norm_rope_insert,
    )

    attn_metadata, layer, kv_cache, slot_mapping = get_attention_context(
        attn.attn.layer_name
    )
    q = torch.empty((qkv.shape[0], attn.q_size), dtype=qkv.dtype, device=qkv.device)
    # no slots in the profile run
    insert = slot_mapping is not None and kv_cache is not None and kv_cache.numel() > 0
    qkv_norm_rope_insert(
        qkv,
        positions,
        attn.q_norm.weight,
        attn.k_norm.weight,
        attn.rotary_emb.cos_sin_cache,
        attn.q_norm.variance_epsilon,
        attn.num_heads,
        attn.num_kv_heads,
        q,
        kv_cache if insert else None,
        slot_mapping if insert else None,
        layer._k_scale,
        layer._v_scale,
    )
    return attn.o_proj(_attention(attn, q, attn_metadata, layer, kv_cache))[0]


def _attention(attn, q: torch.Tensor, attn_metadata, layer, kv_cache):
    """o_proj's input: the library's MXFP8 codes where the fused draft marked the layer
    (not in a breakable PIECEWISE capture), else bf16 ``[tokens, heads * 128]``. Rows of
    no request are not written; the profile run (no metadata) gives zeros; metadata past
    the kernel's case goes to the layer's TritonAttentionImpl."""
    from amd_kernels.attention.swa_decode_fp8kv_gqa16_d128 import MAX_QL
    from amd_kernels.attention.swa_decode_fp8kv_gqa16_d128.oproj_input import (
        swa_decode_fp8kv_oproj_input,
    )

    if attn_metadata is None:
        return torch.zeros_like(q)
    rows = q.shape[0]
    md = attn_metadata
    if not (isinstance(md.causal, bool) and 1 <= md.max_query_len <= MAX_QL):
        out = torch.empty_like(q)
        shape = (rows, q.shape[1] // _HEAD_DIM, _HEAD_DIM)
        layer.impl.forward(
            layer, q.view(shape), None, None, kv_cache, md, out.view(shape)
        )
        return out
    n, k = attn.o_proj.weight.shape
    mxfp8 = getattr(attn, "_akr_mxfp8_out", False) and akr_attn.oproj_mxfp8_mode_ok()
    out = swa_decode_fp8kv_oproj_input(
        q,
        kv_cache,
        md.block_table,
        md.seq_lens,
        md.query_start_loc,
        md.max_query_len,
        layer.impl.scale,
        layer.impl.sliding_window[0] + 1,  # type: ignore[attr-defined]
        md.causal,
        layer._k_scale,
        layer._v_scale,
        oproj_n=n if mxfp8 else None,
    )
    return out if isinstance(out, torch.Tensor) else akr_attn.oproj_input(out, rows, k)


# ---------------------------------------------------------------- the shared lm_head
def plain_processor(lp) -> bool:
    """A logits processor that only applies the lm_head (bf16, full vocabulary)."""
    return (
        lp.soft_cap is None
        and lp.scale == 1.0
        and lp.org_vocab_size == _V
        and not lp.logits_as_input
        and getattr(lp, "head_dtype", None) in (None, torch.bfloat16)
    )


class AkrShuffledLmHeadMethod(UnquantizedEmbeddingMethod):
    """The lm_head's GEMM on amd_kernels' ``bf16_gemm_n50016_k6144`` over the shard in
    ``shuffle_weight``'s layout (any row count)."""

    def apply(self, layer, x, bias=None):
        from amd_kernels.gemm.bf16_n50016_k6144 import bf16_gemm_n50016_k6144

        assert bias is None
        x2 = x.reshape(-1, x.shape[-1]).contiguous()
        out = bf16_gemm_n50016_k6144(x2, layer.weight, shuffled=True)
        return out.reshape(*x.shape[:-1], out.shape[-1])


def switch_shared_lm_head(lm_head, logits_processor, models: list) -> None:
    """Shuffle the lm_head's shard in place (a permutation of its storage) and give it
    the method above: the target's and the draft's logits run the library GEMM.
    Refused when another parameter holds the same tensor (a tied embedding)."""
    from amd_kernels.gemm.bf16_n50016_k6144 import shuffle_weight_, supports

    w = getattr(lm_head, "weight", None)
    if not (
        type(getattr(lm_head, "quant_method", None)) is UnquantizedEmbeddingMethod
        and isinstance(w, torch.Tensor)
        and supports(w, getattr(lm_head, "bias", None))
        and plain_processor(logits_processor)
    ):
        logger.warning("amd_kernels draft: the lm_head stays stock (shape or method)")
        return
    for model in models:
        for name, p in model.named_parameters(remove_duplicate=False):
            if p.data_ptr() == w.data_ptr() and p is not w:
                logger.warning(
                    "amd_kernels draft: the lm_head stays stock (%s holds it)", name
                )
                return
    t0 = time.perf_counter()
    shuffle_weight_(w.data)
    lm_head.quant_method = AkrShuffledLmHeadMethod()
    lm_head._akr_shuffled = True
    torch.accelerator.empty_cache()
    logger.info(
        "amd_kernels draft: the shared lm_head runs amd_kernels (shuffled, %.1f s)",
        time.perf_counter() - t0,
    )
