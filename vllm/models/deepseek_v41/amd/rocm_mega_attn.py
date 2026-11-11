# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


import torch

from vllm.config import VllmConfig
from vllm.models.deepseek_v41.amd.rocm import (
    DeepseekV41ROCMAiterMLAAttention,
    DeepseekV41ROCMAiterMLASparseBackend,
    DeepseekV41ROCMAiterSparseSWAMetadata,
)
from vllm.models.deepseek_v41.sparse_mla import DeepseekV4FlashMLAMetadata
from vllm.platforms import current_platform
from vllm.platforms.rocm import on_gfx950


class DeepseekV41ROCMAiterMLAMegaAttnBackend(DeepseekV41ROCMAiterMLASparseBackend):
    @staticmethod
    def get_name() -> str:
        return "ROCM_FLASHMLA_MEGA_ATTN_DSV41"


class DeepseekV41ROCMAiterMegaAttnAttention(DeepseekV41ROCMAiterMLAAttention):
    """ROCm mega-attention layer for DeepSeek V4.1.

    Q preparation still applies RoPE and uses the ordinary head-major layout.
    Prefill reads ROCm's paged 584-byte cache records directly and fuses
    sparse attention with inverse RoPE and MXFP8 output. Decode uses the
    ordinary ROCm sparse attention path.
    """

    backend_cls = DeepseekV41ROCMAiterMLAMegaAttnBackend

    @classmethod
    def is_available_for(cls, vllm_config: VllmConfig) -> bool:
        """Conservatively select the prefill kernel on MI355X."""
        if not current_platform.is_rocm() or not on_gfx950():
            return False
        if not hasattr(torch.ops._rocm_C, "fused_norm_rope_attn_rope_cast_prefill"):
            return False
        if vllm_config.model_config.dtype != torch.bfloat16:
            return False
        if vllm_config.cache_config.cache_dtype not in ("auto", "fp8", "fp8_ds_mla"):
            return False
        if vllm_config.kv_transfer_config is not None:
            return False
        parallel_config = vllm_config.parallel_config
        if (
            parallel_config.prefill_context_parallel_size != 1
            or parallel_config.decode_context_parallel_size != 1
        ):
            return False
        config = vllm_config.model_config.hf_config
        config = getattr(config, "text_config", config)
        if (
            getattr(config, "head_dim", None) != 512
            or getattr(config, "qk_rope_head_dim", None) != 64
            or (getattr(config, "sliding_window", 0) or 0) <= 0
        ):
            return False
        n_heads = getattr(config, "num_attention_heads", 0) or 0
        n_groups = getattr(config, "o_groups", 0) or 0
        tp_size = parallel_config.tensor_parallel_size
        if n_heads % tp_size or n_groups % tp_size:
            return False
        n_local_heads, n_local_groups = n_heads // tp_size, n_groups // tp_size
        return (
            0 < n_local_heads <= 65535
            and n_local_groups > 0
            and n_local_heads % n_local_groups == 0
        )

    def _forward_prefill(
        self,
        q: torch.Tensor,
        positions: torch.Tensor,
        compressed_k_cache: torch.Tensor | None,
        swa_k_cache: torch.Tensor,
        output: torch.Tensor | None,
        attn_metadata: DeepseekV4FlashMLAMetadata | None,
        swa_metadata: DeepseekV41ROCMAiterSparseSWAMetadata,
        mxfp8_out: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> None:
        """Resolve paged KV and fuse dequantization, attention and MXFP8 output."""
        num_decodes = swa_metadata.num_decodes
        num_prefills = swa_metadata.num_prefills
        first = swa_metadata.num_decode_tokens
        tokens = swa_metadata.num_prefill_tokens
        seq_lens = swa_metadata.prefill_seq_lens
        gather_lens = swa_metadata.prefill_gather_lens
        starts = swa_metadata.query_start_loc
        assert seq_lens is not None and gather_lens is not None and starts is not None
        assert self.topk_indices_buffer is not None
        assert mxfp8_out
        extra_table = None
        if self.compress_ratio:
            assert attn_metadata is not None and compressed_k_cache is not None
            extra_table = attn_metadata.block_table[
                num_decodes : num_decodes + num_prefills
            ]
        torch.ops._rocm_C.fused_norm_rope_attn_rope_cast_prefill(
            q[:tokens],
            swa_k_cache,
            swa_metadata.block_table[num_decodes : num_decodes + num_prefills],
            compressed_k_cache,
            extra_table,
            self.topk_indices_buffer[first : first + tokens],
            starts[num_decodes : num_decodes + num_prefills + 1],
            seq_lens[:num_prefills],
            gather_lens[:num_prefills],
            self.window_size,
            self.compress_ratio,
            self.attn_sink[: q.shape[1]],
            self.scale,
            positions[:tokens],
            self.rotary_emb.cos_sin_cache,
            current_platform.is_fp8_fnuz(),
            mxfp8_out[0][:tokens],
            mxfp8_out[1][:tokens],
        )
