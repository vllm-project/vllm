# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import TYPE_CHECKING

import numpy as np
import torch

from vllm.v1.worker.workspace import current_workspace_manager

if TYPE_CHECKING:
    from vllm.config import VllmConfig


FA3_MAX_DETERMINISTIC_SPLITS = 16
FA3_MAX_DYNAMIC_SPLIT_REQUESTS = 992


def get_split_counts(
    seq_lens: np.ndarray,
    query_lens: np.ndarray,
    is_prefilling: np.ndarray,
    stride: int,
) -> np.ndarray:
    """Return one split for prefill and padding, and length-keyed decode splits."""
    splits = np.ones(seq_lens.shape, dtype=np.int32)
    decode = (query_lens == 1) & ~is_prefilling
    splits[: decode.size][decode] = np.clip(
        (seq_lens[: decode.size][decode] + stride - 1) // stride,
        1,
        FA3_MAX_DETERMINISTIC_SPLITS,
    )
    return splits


def get_split_workspace(
    num_tokens: int, num_heads: int, head_size: int
) -> tuple[torch.Tensor, torch.Tensor]:
    shape = (FA3_MAX_DETERMINISTIC_SPLITS, num_heads, num_tokens)
    output, lse = current_workspace_manager().get_simultaneous(
        ((*shape, head_size), torch.float32), (shape, torch.float32)
    )
    return output, lse


def prepare_deterministic_split_workspace(config: "VllmConfig") -> None:
    """Validate the worker's loaded layers and allocate before the profile forward."""
    import vllm.envs as envs
    from vllm.config.vllm import get_layers_from_vllm_config
    from vllm.model_executor.layers.attention import Attention
    from vllm.platforms import current_platform
    from vllm.v1.attention.backend import AttentionType
    from vllm.v1.attention.backends.flash_attn import FlashAttentionImpl

    model = config.model_config
    parallel = config.parallel_config
    scheduler = config.scheduler_config
    if (
        not envs.VLLM_BATCH_INVARIANT
        or not current_platform.is_cuda()
        or not current_platform.is_device_capability((9, 0))
        or model.dtype != torch.bfloat16
        or config.cache_config.cache_dtype not in ("auto", "bfloat16")
        or not model.disable_cascade_attn
        or model.is_encoder_decoder
        or model.is_multimodal_model
        or config.speculative_config is not None
        or config.cache_config.kv_sharing_fast_prefill
        or config.kv_transfer_config is not None
        or parallel.enable_dbo
        or parallel.data_parallel_size != 1
        or parallel.pipeline_parallel_size != 1
        or parallel.decode_context_parallel_size != 1
        or parallel.prefill_context_parallel_size != 1
    ):
        raise ValueError(
            "Length-keyed FA3 requires batch-invariant Hopper BF16, decoder-only "
            "serving without cascade attention, speculative decoding, fast "
            "prefill, KV transfer, microbatching, DP, PP or context parallelism."
        )

    capture_size = config.compilation_config.max_cudagraph_capture_size or 0
    if (
        max(capture_size, scheduler.max_num_seqs) > FA3_MAX_DYNAMIC_SPLIT_REQUESTS
        or capture_size > scheduler.max_num_batched_tokens
    ):
        raise ValueError(
            "Length-keyed FA3 supports at most 992 padded requests and requires "
            "CUDA graph capture sizes within the scheduler token budget."
        )

    metadata_op = torch.ops._vllm_fa3_C.get_scheduler_metadata.default
    if not any(
        arg.name == "num_splits_per_request" for arg in metadata_op._schema.arguments
    ):
        raise RuntimeError("Length-keyed FA3 requires the updated FA3 extension.")

    num_heads = model.get_num_attention_heads(parallel)
    layers = get_layers_from_vllm_config(config, Attention)
    if not layers or any(
        not isinstance(impl := layer.impl, FlashAttentionImpl)
        or impl.vllm_flash_attn_version != 3
        or impl.attn_type != AttentionType.DECODER
        or impl.head_size != 128
        or impl.num_heads != num_heads
        or impl.sliding_window != (-1, -1)
        or impl.logits_soft_cap != 0
        or impl.sinks is not None
        for layer in layers.values()
    ):
        raise ValueError(
            "Length-keyed FA3 requires causal full attention, head size 128."
        )
    get_split_workspace(scheduler.max_num_batched_tokens, num_heads, 128)
