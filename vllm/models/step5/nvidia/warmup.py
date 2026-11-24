# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import TYPE_CHECKING

from ..common.warmup import run_step5_scored_prefill_warmup
from .sparse_attention import Step5DSAAttentionImpl

if TYPE_CHECKING:
    from vllm.v1.worker.gpu_worker import Worker

    from .model import Step5Attention


def step5_dsa_warmup(layers: list["Step5Attention"], worker: "Worker") -> None:
    """JIT the DSA scored-prefill kernels before the first long prompt."""
    impl = next(
        (
            layer.attn.impl
            for layer in layers
            if isinstance(layer.attn.impl, Step5DSAAttentionImpl)
        ),
        None,
    )
    if impl is None or impl.adapter is None:
        return
    if not run_step5_scored_prefill_warmup(worker, impl.sparse_topk):
        return
    impl.adapter.warmup_prefill_scoring(
        impl.max_num_regions,
        impl.sparse_topk,
        impl.indexer_heads if impl.indexer_heads * 2 == 8 else 0,
        worker.device,
    )
