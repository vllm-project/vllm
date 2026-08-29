# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.model_executor.models.deepseek_v2 import DeepseekV32IndexerCache
from vllm.v1.attention.backend import AttentionLayer
from vllm.v1.attention.backends.mla.indexer import DeepseekV32IndexerBackend
from vllm.v1.attention.backends.mla.rocm_aiter_mla import AiterMLAHelper
from vllm.v1.attention.backends.mla.rocm_aiter_mla_sparse import (
    ROCMAiterMLASparseBackend,
    ROCMAiterMLASparseImpl,
    ROCMAiterMLASparseMetadata,
)
from vllm.v1.attention.ops.rocm_aiter_mla_sparse import (
    rocm_sparse_attn_prefill,
)


class HYV4ROCmSparseIndexerBackend(DeepseekV32IndexerBackend):
    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int]:
        return [16, 32]


class HYV4ROCmSparseIndexerCache(DeepseekV32IndexerCache):
    def get_attn_backend(self):
        return HYV4ROCmSparseIndexerBackend


class HYV4ROCmMLASparseImpl(ROCMAiterMLASparseImpl):
    supports_dense_mha_prefill = False

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
        topk_indices_buffer: torch.Tensor | None = None,
        indexer=None,
        **mla_args,
    ) -> None:
        sinks: torch.Tensor | None = mla_args.pop("sinks", None)
        super().__init__(
            num_heads,
            head_size,
            scale,
            num_kv_heads,
            alibi_slopes,
            sliding_window,
            kv_cache_dtype,
            logits_soft_cap,
            attn_type,
            kv_sharing_target_layer_name,
            topk_indices_buffer=topk_indices_buffer,
            indexer=indexer,
            **mla_args,
        )
        if sinks is None:
            raise ValueError("HYV4 ROCm sparse MLA requires attention sinks")
        if sinks.dtype != torch.float32 or sinks.shape != (num_heads,):
            raise ValueError(
                "HYV4 ROCm sparse MLA sinks must be float32 with shape "
                f"({num_heads},), got dtype={sinks.dtype}, shape={tuple(sinks.shape)}"
            )
        if kv_cache_dtype.startswith("fp8"):
            raise NotImplementedError(
                "HYV4 ROCm sink attention currently requires a BF16 KV cache"
            )
        if self.kv_lora_rank != 512 or mla_args["qk_rope_head_dim"] != 64:
            raise ValueError(
                "HYV4 ROCm sparse MLA requires a 512-d latent and 64-d RoPE"
            )
        self.sinks = sinks
        self.qk_nope_head_dim = self.kv_lora_rank
        self.qk_rope_head_dim: int = mla_args["qk_rope_head_dim"]

    def _forward_mla(
        self,
        layer: AttentionLayer,
        q: torch.Tensor,
        kv_c_and_k_pe_cache: torch.Tensor,
        attn_metadata: ROCMAiterMLASparseMetadata,
    ) -> torch.Tensor:
        if kv_c_and_k_pe_cache.dtype not in (torch.float16, torch.bfloat16):
            raise NotImplementedError(
                "HYV4 ROCm sink attention currently requires an unquantized KV cache"
            )
        output = torch.empty(
            (q.shape[0], q.shape[1], self.kv_lora_rank),
            dtype=attn_metadata.attn_out_dtype,
            device=q.device,
        )
        sinks = self.sinks
        if q.shape[1] > sinks.shape[0]:
            padded_sinks = sinks.new_full((q.shape[1],), float("-inf"))
            padded_sinks[: sinks.shape[0]] = sinks
            sinks = padded_sinks
        rocm_sparse_attn_prefill(
            q=q,
            kv=kv_c_and_k_pe_cache.view(-1, 1, q.shape[-1]),
            indices=attn_metadata.paged_kv_indices,
            topk_length=None,
            scale=self.scale,
            head_dim=q.shape[-1],
            nope_head_dim=self.qk_nope_head_dim,
            rope_head_dim=self.qk_rope_head_dim,
            attn_sink=sinks,
            output=output,
            value_dim=self.kv_lora_rank,
            allow_hyv4=True,
            ragged_indices=attn_metadata.paged_kv_indices,
            ragged_indptr=attn_metadata.paged_kv_indptr,
        )
        return AiterMLAHelper.get_mla_unpadded_o(self.num_heads, output)


class HYV4ROCmMLASparseBackend(ROCMAiterMLASparseBackend):
    @staticmethod
    def get_impl_cls() -> type[HYV4ROCmMLASparseImpl]:
        return HYV4ROCmMLASparseImpl

    @classmethod
    def supports_sink(cls) -> bool:
        return True
