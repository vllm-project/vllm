# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Routes `sparse_mla` and `wo_a` of DeepSeek-V4.1's attention on amd_kernels.

sparse_mla: the decode rows of the AITER sparse MLA call (plain per-tensor fp8
    KV, attention sink, ragged SWA list plus the ragged top-512 list), with the
    inverse RoPE and MXFP8 epilogue into the buffer wo_a reads.
wo_a: the grouped MXFP8 wo_a GEMM with an MXFP8 epilogue, so wo_b (stock, or
    the mHC post-mix chain) reads its codes without a quantize launch. It reads
    the AITER-preshuffled weight the stock linear keeps (no copy). One split-K
    workspace per device: default stream only.
wo_b: at the rows where the dense entry's decode kernel on wo_a's bf16 output
    (quantizer launch + GEMM) beats AITER's wo_b on the epilogue's codes, wo_a
    writes bf16 and the entry runs wo_b (the stock weight, shuffled scales).
"""

import torch

from vllm.logger import init_logger
from vllm.model_executor.layers.fusion.quant_activation import QuantizedActivation
from vllm.models.deepseek_v41.amd import akr_config, akr_kernels

logger = init_logger(__name__)


def _cache_ok(cache: torch.Tensor) -> bool:
    """An fp8 cache `[blocks, block, 512]` with a power-of-two block and
    512-byte rows, the geometry the kernel reads."""
    if cache.dim() != 3 or cache.dtype not in (torch.float8_e4m3fn, torch.uint8):
        return False
    _, block, dim = cache.shape
    return (
        block & (block - 1) == 0
        and cache.stride(2) == 1
        and cache.stride(1) == dim
        and cache.stride(0) % dim == 0
    )


def sparse_mla_decode(
    layer,
    q: torch.Tensor,
    positions: torch.Tensor,
    swa_cache: torch.Tensor,
    swa_indptr: torch.Tensor,
    swa_indices: torch.Tensor,
    extra_cache: torch.Tensor | None,
    extra_indptr: torch.Tensor | None,
    extra_indices: torch.Tensor | None,
    output_mxfp8: tuple[torch.Tensor, torch.Tensor],
) -> bool:
    """True if the kernel wrote this call's MXFP8 output."""
    st = layer.akr
    if not st.mla or not akr_config.serves("sparse_mla", q.shape[0]):
        return False
    if st.mla_caches_ok is None:
        # the caches are bound once; checked on the first (eager) call
        st.mla_caches_ok = _cache_ok(swa_cache) and (
            extra_cache is None or _cache_ok(extra_cache)
        )
    if not st.mla_caches_ok:
        return False
    return akr_kernels.sparse_mla_decode(
        q,
        swa_cache,
        swa_indptr,
        swa_indices,
        extra_cache,
        extra_indptr,
        extra_indices,
        layer.attn_sink,
        layer.scale,
        layer._flashinfer_fp8_q_scale,
        layer._flashinfer_fp8_kv_scale,
        positions,
        layer.rotary_emb.cos_sin_cache,
        *output_mxfp8,
    )


def wo_a_serves(layer, m: int) -> bool:
    w = layer.akr.wo_a
    return (
        w is not None
        and akr_config.serves("wo_a", m)
        and akr_kernels.mxfp8_grouped_ok(m, w)
    )


def wo_a(layer, attn_out: QuantizedActivation) -> QuantizedActivation | None:
    """wo_a's MXFP8 output for wo_b, or None, having launched nothing."""
    x = attn_out.data
    if (
        not wo_a_serves(layer, x.shape[0])
        or not x.is_contiguous()
        or not attn_out.scale.is_contiguous()
    ):
        return None
    codes, scale = akr_kernels.mxfp8_grouped_gemm(attn_out, layer.akr.wo_a)
    return akr_config.quant_activation(codes, scale, attn_out.orig_dtype)


def wo_a_wo_b(layer, attn_out: QuantizedActivation) -> torch.Tensor | None:
    """wo_b(wo_a(attn_out)) with wo_a's bf16 output through the dense entry
    (its quantizer launch + GEMM), where that beats wo_a's MXFP8 epilogue +
    AITER's wo_b on the codes; None, having launched nothing, elsewhere."""
    st = layer.akr
    m = attn_out.data.shape[0]
    if (
        st.wo_b is None
        or st.want_codes
        or layer.gemm_rs is not None
        or not akr_config.serves("wo_b", m)
        or not wo_a_serves(layer, m)
        or not attn_out.data.is_contiguous()
        or not attn_out.scale.is_contiguous()
    ):
        return None
    z = akr_kernels.mxfp8_grouped_gemm_bf16(attn_out, st.wo_a)
    return akr_kernels.mxfp8_dense(z, st.wo_b)


def post_load(model) -> None:
    from vllm.models.deepseek_v41.amd.rocm import DeepseekV41ROCMAiterMLAAttention

    layers = [
        m
        for m in model.modules()
        if isinstance(m, DeepseekV41ROCMAiterMLAAttention) and m.akr is not None
    ]
    states = []
    for layer in layers:
        st = layer.akr
        assert st is not None
        st.mla = (
            akr_config.on("sparse_mla")
            and layer._plain_fp8_kv
            and layer._use_aiter_sparse_mla
        )
        if akr_config.on("wo_a"):
            st.wo_a = akr_kernels.mxfp8_grouped_weight(
                layer.wo_a, layer.n_local_groups, layer.o_lora_rank
            )
        if (
            akr_config.on("wo_b")
            and st.wo_a is not None
            and getattr(layer.wo_b, "tp_size", 1) == 1
        ):
            st.wo_b = akr_kernels.mxfp8_dense_weight(layer.wo_b)
        states.append(st)
    logger.info(
        "amd_kernels attention: sparse_mla on %d, wo_a on %d of %d layers",
        sum(bool(st.mla) for st in states),
        sum(st.wo_a is not None for st in states),
        len(states),
    )
