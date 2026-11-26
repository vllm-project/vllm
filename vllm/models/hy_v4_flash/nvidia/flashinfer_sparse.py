# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Sink-aware FlashInfer sparse MLA for HY4 Flash on SM120."""

import torch

from vllm.model_executor.layers.attention.sparse_mla_attention import (
    SparseMLACommonImpl,
)
from vllm.utils.flashinfer import (
    flashinfer_trtllm_batch_decode_with_kv_cache_mla,
    has_flashinfer_sparse_mla_sm120,
)
from vllm.v1.attention.backend import AttentionType
from vllm.v1.attention.backends.mla.flashinfer_mla_sparse import (
    FlashInferMLASparseMetadataBuilder,
    FlashInferMLASparseSM120Backend,
    _get_workspace_buffer,
)
from vllm.v1.attention.backends.mla.flashinfer_mla_sparse_sm120 import (
    FlashInferMLASparseSM120Impl,
)


class HYV4FlashInferSparseMetadataBuilder(FlashInferMLASparseMetadataBuilder):
    def __init__(self, kv_cache_spec, layer_names, vllm_config, device):
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        layer = vllm_config.compilation_config.static_forward_context[layer_names[0]]
        assert layer.topk_buffer_width is not None
        self.topk_tokens = layer.topk_buffer_width


class HYV4FlashInferSparseBackend(FlashInferMLASparseSM120Backend):
    @classmethod
    def supports_sink(cls) -> bool:
        return True

    @classmethod
    def supports_pcp(cls) -> bool:
        return False

    @classmethod
    def supports_dcp(cls) -> bool:
        return False

    @staticmethod
    def get_builder_cls():
        return HYV4FlashInferSparseMetadataBuilder

    @staticmethod
    def get_impl_cls():
        return HYV4FlashInferSparseImpl


class HYV4FlashInferSWABackend(HYV4FlashInferSparseBackend):
    @classmethod
    def supports_sliding_window(cls) -> bool:
        return True


class HYV4FlashInferSparseImpl(FlashInferMLASparseSM120Impl):
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
        sinks: torch.Tensor,
        **mla_args,
    ) -> None:
        if alibi_slopes is not None or logits_soft_cap is not None:
            raise NotImplementedError("HY4 Flash sparse MLA requires unmodified logits")
        if attn_type != AttentionType.DECODER or kv_cache_dtype != "fp8_ds_mla":
            raise NotImplementedError(
                "HY4 Flash SM120 requires decoder attention with fp8_ds_mla KV"
            )
        if sinks.shape != (num_heads,) or sinks.dtype != torch.float32:
            raise ValueError("HY4 Flash requires one float32 sink per local head")
        if not has_flashinfer_sparse_mla_sm120():
            raise RuntimeError("HY4 Flash SM120 requires FlashInfer sparse MLA")
        SparseMLACommonImpl.__init__(
            self,
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
            **mla_args,
        )
        if self.kv_lora_rank != 512 or self.qk_rope_head_dim != 64:
            raise ValueError("HY4 Flash SM120 requires latent512 + rotary64 KV")
        if self.topk_indices_buffer is None:
            raise ValueError("HY4 Flash SM120 requires sparse token indices")
        self.sinks = sinks
        self.kv_scale_format = "pow2_fp32"
        self.supports_quant_query_input = False
        self._workspace_buffer: torch.Tensor | None = None

    def _run_mqa_kernel(
        self,
        q: torch.Tensor,
        kv_cache: torch.Tensor,
        topk_indices_physical: torch.Tensor,
    ) -> torch.Tensor:
        if self._workspace_buffer is None:
            self._workspace_buffer = _get_workspace_buffer(q.device)
        output = q.new_empty((q.shape[0], self.num_heads, self.kv_lora_rank))
        sparse_capacity = topk_indices_physical.shape[1]
        result = flashinfer_trtllm_batch_decode_with_kv_cache_mla(
            query=q.unsqueeze(1),
            kv_cache=kv_cache.view(torch.uint8).unsqueeze(1),
            workspace_buffer=self._workspace_buffer,
            qk_nope_head_dim=self.qk_nope_head_dim,
            kv_lora_rank=self.kv_lora_rank,
            qk_rope_head_dim=self.qk_rope_head_dim,
            block_tables=topk_indices_physical.unsqueeze(1),
            seq_lens=None,
            max_seq_len=sparse_capacity,
            out=output.unsqueeze(1),
            bmm1_scale=self.scale,
            bmm2_scale=1.0,
            sparse_mla_top_k=sparse_capacity,
            sinks=[self.sinks],
            kv_scale_format=self.kv_scale_format,
        )
        return result.squeeze(1)
