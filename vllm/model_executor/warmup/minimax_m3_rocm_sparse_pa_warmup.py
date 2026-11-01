# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Connect loaded MiniMax-M3 ROCm sparse attention layers to the warmup of
their AITER sparse paged-attention decode kernel."""

import sys
import time
from typing import TYPE_CHECKING

import torch

from vllm.logger import init_logger

if TYPE_CHECKING:
    from vllm.v1.worker.gpu_worker import Worker

logger = init_logger(__name__)


def minimax_m3_rocm_sparse_pa_warmup(worker: "Worker") -> None:
    model_module = sys.modules.get("vllm.models.minimax_m3.amd.model")
    if model_module is None:
        return

    from vllm.models.minimax_m3.amd.ops.sparse_pa import (
        minimax_m3_sparse_attn_aiter_warmup,
    )
    from vllm.models.minimax_m3.amd.sparse_attention_msa import (
        MiniMaxM3SparseAiterPAImpl,
    )

    start = time.perf_counter()
    seen: set[tuple] = set()
    row_counts: tuple[int, ...] = ()
    for layer in worker.get_model().modules():
        if not isinstance(layer, model_module.MiniMaxM3SparseAttention):
            continue
        impl = layer.impl
        # Without the table buffers the indexer emits into, the attend gets
        # freshly allocated (aligned) tables; that path is not warmed here.
        if (
            not layer.use_aiter_sparse_pa
            or not isinstance(impl, MiniMaxM3SparseAiterPAImpl)
            or layer.sparse_bt_buffer is None
            or layer.sparse_ctx_buffer is None
            or layer.kv_cache.numel() == 0
        ):
            continue
        k_cache, v_cache = layer.get_aiter_sparse_pa_kv_cache()
        k_scale = getattr(layer, "_k_scale", None) if impl.use_fp8_kv else None
        v_scale = getattr(layer, "_v_scale", None) if impl.use_fp8_kv else None
        # One allocation of the table buffers serves every layer, so the layers
        # differ at most in their cache views.
        key = (
            impl.num_heads,
            impl.num_kv_heads,
            k_cache.dtype,
            tuple(k_cache.shape),
            tuple(v_cache.shape),
            None if k_scale is None else tuple(k_scale.shape),
            layer.sparse_bt_buffer.data_ptr(),
            layer.sparse_ctx_buffer.data_ptr(),
        )
        if key in seen:
            continue
        seen.add(key)
        row_counts = minimax_m3_sparse_attn_aiter_warmup(
            k_cache,
            v_cache,
            layer.sparse_bt_buffer,
            layer.sparse_ctx_buffer,
            impl.num_heads,
            impl.num_kv_heads,
            impl.head_size,
            impl.scale,
            worker.model_config.dtype,
            k_scale,
            v_scale,
        )
    if not seen:
        return
    torch.accelerator.synchronize()
    logger.info(
        "Warmed up MiniMax-M3 AITER sparse PA decode for %d cache layout(s) at "
        "%s rows, context lengths aligned and not, in %.2fs.",
        len(seen),
        row_counts,
        time.perf_counter() - start,
    )
