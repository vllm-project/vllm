# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""SM120 implementation variant for ``FLASHINFER_MLA_SPARSE``."""

from typing import TYPE_CHECKING, cast

import torch

from vllm.v1.attention.backend import (
    AttentionLayer,
    AttentionType,
    SparseMLAAttentionImpl,
)
from vllm.v1.attention.backends.mla.flashinfer_mla_sparse import (
    FlashInferMLASparseMetadata,
)
from vllm.v1.attention.backends.mla.sparse_utils import (
    triton_convert_req_index_to_global_index,
)
from vllm.v1.worker.workspace import current_workspace_manager

if TYPE_CHECKING:
    from vllm.model_executor.models.deepseek_v2 import Indexer

_DECODE_MAX_TOKENS = 64
_DECODE_SPLIT_TILE = 64


def _cdiv(x: int, y: int) -> int:
    return (int(x) + int(y) - 1) // int(y)


def _max_decode_workspace_tokens(max_num_batched_tokens: int) -> int:
    return min(int(max_num_batched_tokens), _DECODE_MAX_TOKENS)


def _get_decode_scratch(
    num_tokens: int,
    num_heads: int,
    d_v: int,
    topk: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    num_splits = _cdiv(topk, _DECODE_SPLIT_TILE)
    mid_out, mid_lse = current_workspace_manager().get_simultaneous(
        ((num_tokens, num_heads, num_splits, d_v), torch.bfloat16),
        ((num_tokens, num_heads, num_splits), torch.float32),
    )
    return mid_out, mid_lse


class FlashInferMLASparseSM120Impl(SparseMLAAttentionImpl[FlashInferMLASparseMetadata]):
    """SM120 FlashInfer sparse-MLA implementation."""

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int,
        alibi_slopes: list[float] | None,
        sliding_window: int | None,
        kv_cache_dtype: str,
        logits_soft_cap: float | None,
        attn_type: str,
        kv_sharing_target_layer_name: str | None,
        indexer: "Indexer | None" = None,
        **mla_args,
    ) -> None:
        if any([alibi_slopes, sliding_window, logits_soft_cap]):
            raise NotImplementedError(
                "FLASHINFER_MLA_SPARSE SM120 does not support alibi_slopes / "
                "sliding_window / logits_soft_cap"
            )
        if attn_type != AttentionType.DECODER:
            raise NotImplementedError(
                "FLASHINFER_MLA_SPARSE SM120 only supports decoder self-attention"
            )

        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = float(scale)
        self.num_kv_heads = num_kv_heads
        self.kv_cache_dtype = kv_cache_dtype
        if self.kv_cache_dtype != "fp8_ds_mla":
            raise NotImplementedError(
                "FLASHINFER_MLA_SPARSE SM120 requires the packed fp8_ds_mla "
                f"KV cache layout; got kv_cache_dtype={kv_cache_dtype!r}."
            )

        self.kv_lora_rank: int = mla_args["kv_lora_rank"]

        assert indexer is not None, (
            "FLASHINFER_MLA_SPARSE SM120 requires a sparse-MLA indexer "
            "(model with index_topk in its config)."
        )
        self.topk_indices_buffer: torch.Tensor | None = indexer.topk_indices_buffer
        from vllm.utils.flashinfer import has_flashinfer_sparse_mla_sm120

        if not has_flashinfer_sparse_mla_sm120():
            raise RuntimeError(
                "FLASHINFER_MLA_SPARSE SM120 requires FlashInfer's "
                "sparse-sm120 MLA wrapper."
            )

        from flashinfer.mla import BatchMLAPagedAttentionWrapper

        from vllm.config import get_current_vllm_config

        vllm_config = get_current_vllm_config()
        max_tokens = vllm_config.scheduler_config.max_num_batched_tokens
        wrapper_device = torch.device("cuda", torch.accelerator.current_device_index())
        kv_scale_format = "arbitrary_fp32" if self.head_size == 576 else "auto"
        self._wrapper = BatchMLAPagedAttentionWrapper(
            torch.empty(1, dtype=torch.int8, device=wrapper_device),
            backend="sparse-sm120",
            max_num_tokens=max_tokens,
            max_num_heads=num_heads,
            d_v=self.kv_lora_rank,
            kv_scale_format=kv_scale_format,
        )
        assert self.topk_indices_buffer is not None
        # Reserve shared decode scratch before the first real decode.
        _get_decode_scratch(
            _max_decode_workspace_tokens(max_tokens),
            num_heads,
            self.kv_lora_rank,
            self.topk_indices_buffer.shape[-1],
        )

        # The wrapper consumes BF16 Q and packed KV scales directly.
        self.supports_quant_query_input = False

    def forward_mqa(
        self,
        q: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        kv_c_and_k_pe_cache: torch.Tensor,
        attn_metadata: FlashInferMLASparseMetadata,
        layer: AttentionLayer,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        # The wrapper expects a single [T, H, kv_lora_rank + rope] tensor.
        if isinstance(q, tuple):
            q = torch.cat(q, dim=-1)

        num_actual_toks = q.shape[0]

        assert self.topk_indices_buffer is not None
        topk_indices = self.topk_indices_buffer[:num_actual_toks]

        topk_indices_physical = cast(
            torch.Tensor,
            triton_convert_req_index_to_global_index(
                attn_metadata.req_id_per_token[:num_actual_toks],
                attn_metadata.block_table,
                topk_indices,
                BLOCK_SIZE=attn_metadata.block_size,
                NUM_TOPK_TOKENS=topk_indices.shape[1],
            ),
        )

        output = q.new_empty(
            (num_actual_toks, self.num_heads, self.kv_lora_rank),
            dtype=q.dtype,
        )

        # Add the singleton kv-head axis expected by the FlashInfer wrapper.
        kv_cache_4d = kv_c_and_k_pe_cache.view(torch.uint8).unsqueeze(-2)

        mid_out = None
        mid_lse = None
        if num_actual_toks <= _DECODE_MAX_TOKENS:
            mid_out, mid_lse = _get_decode_scratch(
                num_actual_toks,
                self.num_heads,
                self.kv_lora_rank,
                topk_indices_physical.shape[-1],
            )

        self._wrapper.run_sparse_mla(
            q=q,
            kv_cache=kv_cache_4d,
            sparse_indices=topk_indices_physical,
            out=output,
            sm_scale=self.scale,
            mid_out=mid_out,
            mid_lse=mid_lse,
        )
        return output, None
