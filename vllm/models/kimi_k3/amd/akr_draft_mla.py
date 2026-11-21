# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""`attention/mla_decode_common.fp8_dense_mla_decode` (switch `DMLA`): the DSpark
drafter's dense MLA decode over the fp8 latent cache in place of AITER's `mla_a8w8`
persistent decode (96 heads padded to 128, qlen 3 folded to 4).

Hooked by wrapping `AiterMLAImpl._forward_mqa` once (the backend file is not edited):
a call from a draft layer (`layer.is_draft_layer`) with the CPRR DCP route, causal, fp8
q [rows, heads, d_qk] that the entry serves (`fp8_dense_mla_decode_unsupported_reason`:
heads x d_qk, query length, requests, layouts) takes the kernel; the stock function
runs for everything else (the target's layers are DSA, never this impl). It returns
(o [rows, heads, kv_lora] bf16, lse [rows, heads] fp32 natural log) to the existing DCP
LSE merge. The drafter's `get_mla_metadata_v1` call is not skipped (it stays in the
backend's `_build_decode`; ~76 us/step left on the table, see the report)."""

import functools
from typing import Any

import torch

from vllm.logger import init_logger
from vllm.models.kimi_k3.amd import akr_config, akr_nonlayer

logger = init_logger(__name__)

_state: dict[str, Any] = {"installed": False}


def _warm(qlen: int, reqs: int, heads: int, row: int, cp_world: int, device):
    """Compile the kernel of `qlen` on a one-request dummy; the entry's reason (for a
    step of `reqs` requests) when it has none."""
    from amd_kernels.attention.mla_decode_common import (
        fp8_dense_mla_decode,
        fp8_dense_mla_decode_unsupported_reason,
    )

    def args(r: int, n: int = 16) -> tuple:
        f8, i32 = torch.float8_e4m3fn, torch.int32
        indptr = torch.full((r + 1,), n, dtype=i32, device=device)
        indptr[0] = 0
        one = torch.ones(1, dtype=torch.float32, device=device)
        return (
            torch.zeros(r * qlen, heads, row, dtype=f8, device=device),
            torch.zeros(n, row, dtype=f8, device=device),
            indptr,
            torch.arange(n, dtype=i32, device=device),
            indptr * cp_world,
            one,
            one,
            0.1,
            qlen,
            0,
            cp_world,
        )

    reason = fp8_dense_mla_decode_unsupported_reason(*args(reqs))
    if reason is not None:
        return reason
    fp8_dense_mla_decode(*args(1))
    torch.accelerator.synchronize(device)
    return None


def _try(impl, q, kv_cache, attn_metadata, layer):
    from vllm.v1.attention.backends.mla import rocm_aiter_mla as backend

    decode = attn_metadata.decode
    if (
        decode is None
        or decode.use_gluon_decode
        or decode.use_gluon_verify
        or decode.dcp_route is not backend._DCPDecodeRoute.CPRR
        or decode.g_kv_indptr is None
        or decode.max_qo_len is None
        or not attn_metadata.causal
        or decode.attn_out_dtype != torch.bfloat16
        or decode.paged_kv_indptr is None
        or decode.paged_kv_indices is None
    ):
        return None
    if type(q) is tuple:
        q = torch.cat(q, dim=-1)
    kv = backend._flat_mla_kv_view(kv_cache)
    if kv.dtype == torch.uint8:
        kv = kv.view(torch.float8_e4m3fn)
    from amd_kernels.attention.mla_decode_common import (
        fp8_dense_mla_decode,
        fp8_dense_mla_decode_unsupported_reason,
    )

    args = (
        q,
        kv,
        decode.paged_kv_indptr,
        decode.paged_kv_indices,
        decode.g_kv_indptr,
        layer._q_scale,
        layer._k_scale,
        impl.scale,
        int(decode.max_qo_len),
        decode.cp_rank,
        impl.dcp_world_size,
    )
    if akr_config.call_unsupported(
        "draft", fp8_dense_mla_decode_unsupported_reason(*args)
    ):
        return None
    return fp8_dense_mla_decode(*args)


def _wrap(stock):
    @functools.wraps(stock)
    def _forward_mqa(self, q, kv_c_and_k_pe_cache, attn_metadata, layer):
        if _state.get("on") and getattr(layer, "is_draft_layer", False):
            out = _try(self, q, kv_c_and_k_pe_cache, attn_metadata, layer)
            if out is not None:
                return out
        return stock(self, q, kv_c_and_k_pe_cache, attn_metadata, layer)

    return _forward_mqa


def setup(spec, model) -> None:
    if not akr_nonlayer.switch("DMLA"):
        return
    reason = None
    if spec.dcp_size != 8:
        reason = f"decode context parallel size {spec.dcp_size} (validated on DCP8)"
    elif spec.speculative_config.kv_cache_dtype not in ("fp8", "fp8_e4m3"):
        reason = f"draft kv cache dtype {spec.speculative_config.kv_cache_dtype}"
    if akr_nonlayer.blocked("DMLA", reason):
        return
    cfg = spec.draft_model_config.hf_config
    tp = akr_nonlayer.tp_group().world_size
    heads = cfg.num_attention_heads // tp * spec.dcp_size
    row = cfg.kv_lora_rank + cfg.qk_rope_head_dim
    reason = _warm(
        spec.num_query_per_req,
        spec.max_num_reqs,
        heads,
        row,
        spec.dcp_size,
        spec.device,
    )
    if akr_nonlayer.blocked("DMLA", reason):
        return
    from vllm.v1.attention.backends.mla import rocm_aiter_mla as backend

    if not _state["installed"]:
        backend.AiterMLAImpl._forward_mqa = _wrap(  # type: ignore[method-assign]
            backend.AiterMLAImpl._forward_mqa
        )
        _state["installed"] = True
    _state["on"] = True
    logger.info(
        "amd_kernels drafter: dense MLA decode kernel (qlen %d)", spec.num_query_per_req
    )
