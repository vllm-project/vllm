# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Bound attention's query batch while retaining large projection/MoE batches."""

from collections.abc import Callable
from typing import Any

import torch

QUERY_CHUNK_SIZE = 2048
QUERY_TILE_SIZE = 256
MIN_CHUNKED_CONTEXT = 32768


def split_attention_queries(
    kernel: Callable[..., Any], arguments: dict[str, Any], chunk_size: int
) -> torch.Tensor:
    """Preserve each query's absolute causal position in the paged KV cache."""
    q = arguments["q"]
    rows = q.shape[0]
    output = arguments.get("out")
    if output is None:
        output = torch.empty_like(q)
    # Construct metadata on device so prefill can also be captured in a Graph.
    indices = torch.arange(2, dtype=torch.int32, device=q.device)
    full_cu = indices * chunk_size
    start = 0
    while start < rows:
        end = min(start + chunk_size, rows)
        if rows - end == 1:
            # FA2 treats q_len=1 specially; keep the final query in prefill.
            end = rows
        count = end - start
        remaining = rows - end
        kernel(
            **dict(
                arguments,
                q=q[start:end],
                out=output[start:end],
                cu_seqlens_q=full_cu if count == chunk_size else indices * count,
                max_seqlen_q=count,
                seqused_k=arguments["seqused_k"] - remaining,
                max_seqlen_k=arguments["max_seqlen_k"] - remaining,
            )
        )
        start = end
    return output


class MiniCPM5PrefillAttention:
    def __init__(self, kernel: Callable[..., Any], chunk_size: int):
        self.kernel = kernel
        self.chunk_size = chunk_size

    def eligible(self, arguments: dict[str, Any]) -> bool:
        q, k, v = (arguments[name] for name in ("q", "k", "v"))
        return (
            self.chunk_size > 1
            and self.chunk_size % QUERY_TILE_SIZE == 0
            and arguments["max_seqlen_k"] > MIN_CHUNKED_CONTEXT
            and arguments["max_seqlen_q"] > self.chunk_size
            and q.shape == (arguments["max_seqlen_q"], 16, 128)
            and q.device.type == "xpu"
            and q.dtype == k.dtype == v.dtype == torch.float16
            and k.dim() == 4
            and k.shape[2] == 1
            and arguments["cu_seqlens_q"].numel() == 2
            and arguments.get("seqused_k") is not None
            and arguments["seqused_k"].numel() == 1
            and arguments.get("block_table") is not None
            and arguments["block_table"].shape[0] == 1
            and arguments.get("causal") is True
            and not arguments.get("return_softmax_lse")
            and not arguments.get("return_attn_probs")
            and not arguments.get("dropout_p")
            and arguments.get("window_size") in (None, [-1, -1], (-1, -1))
            and not arguments.get("softcap")
            and all(
                arguments.get(name) is None
                for name in (
                    "cu_seqlens_k",
                    "alibi_slopes",
                    "q_v",
                    "q_descale",
                    "scheduler_metadata",
                    "s_aux",
                    "mask_mod",
                    "aux_tensors",
                    "dynamic_causal",
                )
            )
        )

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        if args or not self.eligible(kwargs):
            return self.kernel(*args, **kwargs)
        return split_attention_queries(self.kernel, kwargs, self.chunk_size)


def configure_prefill_attention(model, config) -> None:
    options = config.additional_config or {}
    chunk_size = options.get("minicpm5_prefill_query_chunk_size", QUERY_CHUNK_SIZE)
    if type(chunk_size) is not int or chunk_size not in (0, QUERY_CHUNK_SIZE):
        raise ValueError("minicpm5_prefill_query_chunk_size must be 0 or 2048")
    if (
        chunk_size == 0
        or config.model_config.dtype != torch.float16
        or config.parallel_config.tensor_parallel_size != 2
    ):
        return

    from vllm.v1.attention.backends.flash_attn import FlashAttentionImpl

    for layer in model.layers:
        impl = layer.self_attn.attn.impl
        if type(impl) is FlashAttentionImpl and impl.vllm_flash_attn_version == 2:
            impl.dense_attention_kernel = MiniCPM5PrefillAttention(
                impl.dense_attention_kernel, chunk_size
            )
