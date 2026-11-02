# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import TYPE_CHECKING, Any

import torch

from vllm.logger import init_logger
from vllm.model_executor.warmup.attention_warmup import (
    mixed_batch_attention_warmup,
)
from vllm.models.minimax_m3.nvidia.indexer_msa import MiniMaxM3IndexerMSAImpl
from vllm.models.minimax_m3.nvidia.indexer_nvfp4 import MiniMaxM3IndexerNvfp4Impl
from vllm.models.minimax_m3.nvidia.model import MiniMaxM3SparseAttention
from vllm.models.minimax_m3.nvidia.msa_cutlass_sparse_decode import (
    NVFP4_BLOCK_SCALE_SHIFT,
    uses_q8kv4_decode,
)
from vllm.models.minimax_m3.nvidia.sparse_attention_msa import (
    MiniMaxM3SparseMSABaseImpl,
)
from vllm.platforms import current_platform
from vllm.tracing import instrument

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.v1.worker.gpu_worker import Worker

logger = init_logger(__name__)

# KV-cache dtypes fmha_sm100.warmup plans attention kernels for.
_MSA_KV_CACHE_DTYPES = ("auto", "bfloat16", "fp8", "fp8_e4m3", "nvfp4")


def _msa_prebuild_kwargs(
    vllm_config: "VllmConfig", module: MiniMaxM3SparseAttention
) -> dict[str, Any] | None:
    """``fmha_sm100.warmup`` arguments for the kernels this rank can request."""
    impl = module.impl
    indexer = module.indexer.impl
    msa = (
        impl
        if isinstance(impl, MiniMaxM3SparseMSABaseImpl)
        and module.kv_cache_dtype in _MSA_KV_CACHE_DTYPES
        else None
    )
    attention = msa is not None
    if isinstance(indexer, MiniMaxM3IndexerNvfp4Impl):
        index_cache_dtype = "nvfp4"
    elif isinstance(indexer, MiniMaxM3IndexerMSAImpl):
        index_cache_dtype = module.indexer_kv_dtype
    else:
        index_cache_dtype = None
    if not attention and index_cache_dtype is None:
        return None

    spec_config = vllm_config.speculative_config
    num_spec = spec_config.num_speculative_tokens if spec_config is not None else 0
    q8kv4_decode = attention and uses_q8kv4_decode(
        kv_cache_dtype=module.kv_cache_dtype,
        num_q_heads=module.num_heads,
        num_kv_heads=module.num_kv_heads,
    )
    return dict(
        kv_cache_dtype=module.kv_cache_dtype if attention else "auto",
        index_cache_dtype=index_cache_dtype,
        num_q_heads=module.num_heads,
        num_kv_heads=module.num_kv_heads,
        topk=impl.topk_blocks,
        decode_query_lens=range(1, num_spec + 2),
        num_index_heads=indexer.num_index_heads,
        page_size=impl.block_size,
        decode_backend="q8kv4" if q8kv4_decode else "kv_mode3",
        block_scale_shift=NVFP4_BLOCK_SCALE_SHIFT,
        sparse_decode=msa is not None and msa.use_cutlass_decode,
        sparse_prefill=attention,
        prefill_backend=msa.prefill_backend if msa is not None else "cute_dsl",
        device=torch.device("cuda", torch.accelerator.current_device_index()),
    )


@instrument(span_name="MiniMax M3 MSA warmup")
def minimax_m3_msa_warmup(worker: "Worker") -> None:
    sparse_module = next(
        (
            module
            for module in worker.get_model().modules()
            if isinstance(module, MiniMaxM3SparseAttention)
        ),
        None,
    )
    if sparse_module is None:
        return
    if not (
        current_platform.is_cuda() and current_platform.is_device_capability_family(100)
    ):
        return

    logger.info("Warming up MiniMax M3 MSA kernels.")

    # Build every native kernel the configuration can request at once instead
    # of compiling each one serially on first use.
    kwargs = _msa_prebuild_kwargs(worker.vllm_config, sparse_module)
    if kwargs is not None:
        from vllm.third_party.fmha_sm100 import warmup

        logger.info("MiniMax M3 MSA kernels: %s", warmup(**kwargs))

    # Cover sparse prefill (and its CuTe-DSL kernels) through the normal model
    # path.
    mixed_batch_attention_warmup(worker)
