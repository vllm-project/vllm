# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek-V4.1 attention input on amd_kernels (decode rows).

Routes:
- `dense_gemm`: the fused wq_a + wkv GEMM, and the indexer wq_b on the
  MXFP8 input `norm_dual` writes, at the row counts the entry has winning
  rows for. They read the stock linears' preshuffled weights.
- `norm_dual`: q and kv RMSNorm in one launch, q as MXFP8 that both wq_b
  linears take as it is.
- `kv_score`: the compressor's kv_score GEMM and the indexer's weights_proj in
  one launch. It keeps one split-K scratch per device: never two at once.
- `compressor`: pool, norm, RoPE, main-cache insert and the indexer key store
  of a kv-source layer in one launch.
- `rope_insert`: q RoPE + fp8 cast and the KV RoPE + fp8 SWA insert, for a
  plain per-tensor fp8 KV cache.

Every entry point returns None (or False) without any work done where it does
not serve, and the caller runs the stock code.
"""

from typing import Any

import torch

from vllm.logger import init_logger
from vllm.model_executor.layers.fusion.quant_activation import QuantizedActivation
from vllm.models.deepseek_v41.amd import akr_config, akr_kernels

logger = init_logger(__name__)

_ROUTES = (
    "rope_insert",
    "norm_dual",
    "compressor",
    "kv_score",
    "dense_gemm",
    "wq_b",
    "indexer",
    "sparse_mla",
    "wo_a",
    "wo_b",
)


class LayerState:
    """`DeepseekV41ROCMAiterMLAAttention.akr`: per-layer state of the attention
    routes (this module's, `akr_indexer`'s and `akr_mla`'s)."""

    def __init__(self, layer) -> None:
        self.rope = (
            layer._plain_fp8_kv and not layer.use_pcp and akr_config.on("rope_insert")
        )
        self.norm_codes: bool | None = None  # both wq_b linears take MXFP8 input
        self.kv_score_w: torch.Tensor | None = None
        self.kv_score_n = 0
        self.wqa_wkv: Any = None
        self.wq_b: Any = None
        self.wo_b: Any = None  # akr_mla
        self.indexer: Any = None  # akr_indexer
        self.mla = False  # akr_mla
        self.mla_caches_ok: bool | None = None
        self.wo_a: Any = None  # akr_mla
        self.ragged_fresh = False
        self.want_codes = False
        self.codes: QuantizedActivation | None = None


class CompressorState:
    """`DeepseekCompressor.akr` of a kv-source layer (route `compressor`)."""

    def __init__(self, layer) -> None:
        self.layer = layer
        self.wk: torch.Tensor | None = None

    def compress_insert(self, compressor, kv_score, positions, attn_metadata) -> bool:
        return compress_insert(self, compressor, kv_score, positions, attn_metadata)


def create(layer) -> LayerState | None:
    """At the end of the attention layer's constructor."""
    if not any(akr_config.on(r) for r in _ROUTES):
        return None
    comp, ind = layer.compressor, layer.indexer
    if (
        akr_config.on("compressor")
        and comp is not None
        and ind is not None
        and ind.owns_k
        and ind.use_fp4_kv
        and getattr(ind.k_cache, "rocm_mxfp4", False)
        and layer._plain_fp8_kv
        and not layer.use_pcp
    ):
        comp.akr = CompressorState(layer)
    return LayerState(layer)


# ---------------------------------------------------------------- dense_gemm


def wqa_wkv(layer, hidden: torch.Tensor) -> torch.Tensor | None:
    """The fused wq_a + wkv GEMM on the bf16 layer input. None where a seam
    attached the input's MXFP8 copy: the stock GEMM on those codes is faster
    than the entry on bf16 (which quantizes) at every row count."""
    from vllm.models.deepseek_v4.amd.model import mxfp8_input

    w = layer.akr.wqa_wkv
    if (
        w is None
        or hidden.dim() != 2
        or hidden.dtype != torch.bfloat16
        or not akr_config.serves("dense_gemm", hidden.shape[0])
        or mxfp8_input(hidden) is not None
    ):
        return None
    return akr_kernels.mxfp8_dense(hidden, w)


def wq_b(layer, qr) -> torch.Tensor | None:
    """The attention wq_b on `norm_dual`'s MXFP8 rows, where the entry beats
    AITER's GEMM on the same codes."""
    w = layer.akr.wq_b
    if (
        w is None
        or not isinstance(qr, QuantizedActivation)
        or len(qr.orig_shape) != 2
        or not akr_config.serves("wq_b", qr.data.shape[0])
    ):
        return None
    return akr_kernels.mxfp8_dense(qr, w)


def indexer_wq_b(indexer, qr) -> torch.Tensor | None:
    """The indexer wq_b on `norm_dual`'s MXFP8 rows."""
    if not isinstance(qr, QuantizedActivation) or not akr_config.serves(
        "dense_gemm", qr.data.shape[0]
    ):
        return None
    return akr_kernels.mxfp8_dense(qr, indexer.akr)


# ---------------------------------------------------------------- norm_dual


def norm_dual(layer, qr_kv: torch.Tensor):
    """(MXFP8 qr, bf16 kv) or None."""
    st = layer.akr
    if st.norm_codes is None:
        linears = [layer.wq_b] + ([layer.indexer.wq_b] if layer.indexer else [])
        st.norm_codes = akr_config.on("norm_dual") and all(
            akr_config.takes_mxfp8_codes(lin) for lin in linears
        )
    if (
        not st.norm_codes
        or qr_kv.dim() != 2
        or qr_kv.dtype != torch.bfloat16
        or qr_kv.stride(1) != 1
        or not akr_config.serves("norm_dual", qr_kv.shape[0])
    ):
        return None
    qr, kv = qr_kv.split([layer.q_lora_rank, layer.head_dim], dim=-1)
    out = akr_kernels.rmsnorm_dual_mxfp8(
        qr,
        kv,
        layer.q_norm.weight.data,
        layer.kv_norm.weight.data,
        layer.eps,
        pad_scale=st.wq_b is not None and akr_config.serves("wq_b", qr_kv.shape[0]),
    )
    if out is None:
        return None
    codes, scale, kv_out = out
    return akr_config.quant_activation(codes, scale, qr_kv.dtype), kv_out


# ---------------------------------------------------------------- kv_score


def kv_score_serves(layer, hidden: torch.Tensor) -> bool:
    st = layer.akr
    return (
        st.kv_score_w is not None
        and hidden.dim() == 2
        and hidden.dtype == torch.bfloat16
        and hidden.is_contiguous()
        and akr_config.serves("kv_score", hidden.shape[0])
        and akr_kernels.bf16_dual_gemm_ok(hidden, st.kv_score_w, st.kv_score_n)
    )


def kv_score_weights(
    layer, hidden: torch.Tensor
) -> tuple[torch.Tensor | None, torch.Tensor]:
    """fp32 `[kv | score]` (None for a layer without a compressor) and the
    indexer's bf16 weights_proj output, in one launch."""
    st = layer.akr
    m = hidden.shape[0]
    kv_score = (
        torch.empty(m, st.kv_score_n, dtype=torch.float32, device=hidden.device)
        if st.kv_score_n
        else None
    )
    weights = torch.empty(
        m,
        layer.indexer.weights_proj.weight.shape[0],
        dtype=torch.bfloat16,
        device=hidden.device,
    )
    akr_kernels.bf16_dual_gemm(hidden, st.kv_score_w, kv_score, weights)
    return kv_score, weights


# ---------------------------------------------------------------- compressor


def compress_insert(st: CompressorState, compressor, kv_score, positions, md) -> bool:
    """Everything after the compressor GEMM of a kv-source layer in one launch;
    True if it ran."""
    layer = st.layer
    indexer = layer.indexer
    n = kv_score.shape[0]
    if st.wk is None or not akr_config.serves("compressor", n):
        return False
    main_md = md[compressor.k_cache_prefix]
    idx_md = md[indexer.k_cache.prefix]
    if compressor.compress_ratio == 2:
        state_md = md[compressor.state_cache.prefix]
        ring, ring_slots = compressor.state_cache.kv_cache, state_md.slot_mapping
        query_start_loc = state_md.query_start_loc
        token_to_req = state_md.token_to_req_indices
    else:
        ring, ring_slots = None, main_md.slot_mapping
        query_start_loc = token_to_req = None
    # padded rows past the slot mappings carry no group boundary
    n = min(n, main_md.slot_mapping.numel(), idx_md.slot_mapping.numel())
    n = min(n, ring_slots.numel())
    return akr_kernels.compressor_insert(
        kv_score[:n],
        positions[:n],
        ring,
        ring_slots[:n],
        query_start_loc,
        None if token_to_req is None else token_to_req[:n],
        compressor.norm.weight.data,
        compressor.rms_norm_eps,
        layer.kv_cache,
        main_md.slot_mapping[:n],
        layer.rotary_emb.cos_sin_cache,
        layer._flashinfer_fp8_kv_scale,
        st.wk,
        indexer.k_norm.weight.data,
        indexer.k_norm.variance_epsilon,
        indexer.k_cache.kv_cache,
        idx_md.slot_mapping[:n],
        layer.indexer_rotary_emb.cos_sin_cache,
        compressor.compress_ratio,
    )


# ---------------------------------------------------------------- rope_insert


def rope_insert(layer, q, kv, positions, attn_metadata) -> torch.Tensor | None:
    """fp8 q `[T, H, D]`, having inserted the fp8 KV rows; None for the stock
    kernel."""
    if not layer.akr.rope or not akr_config.serves("rope_insert", q.shape[0]):
        return None
    swa_md = attn_metadata.get(layer.swa_cache_layer.prefix)
    cache = layer.swa_cache_layer.kv_cache
    if (
        swa_md is None
        or cache.dtype != torch.float8_e4m3fn
        or q.dtype != torch.bfloat16
        or not q.is_contiguous()
        or not kv.is_contiguous()
        or swa_md.slot_mapping.numel() < q.shape[0]
    ):
        return None
    return akr_kernels.rope_fp8_kv_insert(
        q,
        kv,
        cache,
        swa_md.slot_mapping,
        positions,
        layer.rotary_emb.cos_sin_cache,
        layer._flashinfer_fp8_kv_scale,
        layer._flashinfer_fp8_q_scale_inv,
    )


# ---------------------------------------------------------------- load


def post_load(model) -> None:
    from vllm.models.deepseek_v41.amd.rocm import DeepseekV41ROCMAiterMLAAttention

    layers: list[Any] = [
        m
        for m in model.modules()
        if isinstance(m, DeepseekV41ROCMAiterMLAAttention) and m.akr is not None
    ]
    for layer in layers:
        st, ind, comp = layer.akr, layer.indexer, layer.compressor
        if akr_config.on("wq_b"):
            st.wq_b = akr_kernels.mxfp8_dense_weight(layer.wq_b)
        if akr_config.on("dense_gemm"):
            st.wqa_wkv = akr_kernels.mxfp8_dense_weight(layer.fused_wqa_wkv)
            if ind is not None:
                ind.akr = akr_kernels.mxfp8_dense_weight(ind.wq_b)
        if akr_config.on("kv_score") and ind is not None and not layer.use_pcp:
            wkv = None if comp is None else comp.fused_wkv_wgate.weight.data
            st.kv_score_w = akr_kernels.bf16_dual_gemm_weight(
                wkv, ind.weights_proj.weight.data
            )
            st.kv_score_n = 0 if wkv is None else wkv.shape[0]
        if comp is not None and comp.akr is not None:
            comp.akr.wk = akr_kernels.compressor_wk(ind.wk.weight.data)
    if layers and akr_config.on("norm_dual"):
        layer = layers[0]
        akr_kernels.rmsnorm_dual_warmup(
            layer.q_lora_rank, layer.head_dim, layer.q_norm.weight.device
        )
