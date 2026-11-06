# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


from typing import Any

import torch

from vllm.config import VllmConfig
from vllm.forward_context import get_forward_context
from vllm.models.deepseek_v41.amd.rocm import (
    DeepseekV41ROCMAiterMLAAttention,
    DeepseekV41ROCMAiterMLASparseBackend,
    DeepseekV41ROCMAiterSparseSWAMetadata,
)
from vllm.models.deepseek_v41.sparse_mla import DeepseekV4FlashMLAMetadata
from vllm.platforms import current_platform
from vllm.platforms.rocm import get_cdna_version


class DeepseekV41ROCMAiterMLAMegaAttnBackend(DeepseekV41ROCMAiterMLASparseBackend):
    @staticmethod
    def get_name() -> str:
        return "ROCM_FLASHMLA_MEGA_ATTN_DSV41"


class DeepseekV41ROCMAiterMegaAttnAttention(DeepseekV41ROCMAiterMLAAttention):
    """ROCm mega-attention layer for DeepSeek V4.1.

    Q keeps the ordinary head-major layout. Prefill reads ROCm's paged
    584-byte cache records directly and fuses sparse attention with inverse
    RoPE: MXFP8 output on gfx950, BF16 output on gfx942. Decode uses the
    ordinary ROCm sparse attention path.

    The Q preparation applies RoPE, except on gfx942 for batches larger than
    any CUDA graph: there it only inserts KV, the prefill kernel rotates its
    Q rows, and ``forward_mqa`` rotates the decode rows. Both run in the
    eager attention region, where the step's decode/prefill split is known.
    """

    backend_cls = DeepseekV41ROCMAiterMLAMegaAttnBackend
    _prefill_rotates_bf16 = True

    @classmethod
    def is_available_for(cls, vllm_config: VllmConfig) -> bool:
        """Conservatively select the prefill kernel on CDNA4 and CDNA3."""
        if not current_platform.is_rocm():
            return False
        cdna = get_cdna_version()
        if cdna == 4:
            op = "fused_norm_rope_attn_rope_cast_prefill"
        elif cdna == 3:
            op = "fused_norm_rope_attn_rope_prefill_bf16"
        else:
            return False
        if not hasattr(torch.ops._rocm_C, op):
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

    def __init__(self, *args, **kwargs):
        vllm_config = args[0] if args else kwargs["vllm_config"]
        super().__init__(*args, **kwargs)
        # Batches a CUDA graph can capture keep Q RoPE in the prep, since a
        # graph would replay one step's decode/prefill split. The prep and
        # forward_mqa decide on the same row count. Below 512 rows the extra
        # decode-row rotation costs about what the fused prep saves: on
        # MI325X, a step with 4 decode rows breaks even near 384.
        captured = vllm_config.compilation_config.max_cudagraph_capture_size or 0
        self._attn_rotates_q_from = (
            max(captured, 512) + 1 if get_cdna_version() == 3 else None
        )
        self._prefill_rotate_q = False

    def _attn_rotates_q(self, q: torch.Tensor) -> bool:
        threshold = self._attn_rotates_q_from
        return threshold is not None and q.shape[0] >= threshold

    def _fused_qnorm_rope_kv_insert(
        self,
        q: torch.Tensor,
        kv: torch.Tensor,
        positions: torch.Tensor,
        attn_metadata: Any,
    ) -> torch.Tensor:
        if not self._attn_rotates_q(q):
            return super()._fused_qnorm_rope_kv_insert(q, kv, positions, attn_metadata)
        if isinstance(attn_metadata, dict):
            swa_metadata = attn_metadata[self.swa_cache_layer.prefix]
            torch.ops._C.fused_deepseek_v4_kv_rope_insert(
                kv,
                self.swa_cache_layer.kv_cache,
                swa_metadata.slot_mapping,
                positions,
                self.rotary_emb.cos_sin_cache,
                swa_metadata.block_size,
                None,
                self.kv_mxfp8,
            )
        return q

    def forward_mqa(
        self,
        q: torch.Tensor,
        kv: torch.Tensor,
        positions: torch.Tensor,
        output: Any,
    ) -> None:
        attn_metadata = get_forward_context().attn_metadata
        self._prefill_rotate_q = self._attn_rotates_q(q)
        if self._prefill_rotate_q and isinstance(attn_metadata, dict):
            swa_metadata = attn_metadata[self.swa_cache_layer.prefix]
            n = swa_metadata.num_decode_tokens
            if n:
                cache = self.swa_cache_layer.kv_cache
                # An empty slot mapping skips the KV half: rotate Q only.
                q[:n] = torch.ops._C.fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert(
                    q[:n],
                    kv[:n],
                    cache.view(cache.shape[0], -1),
                    swa_metadata.slot_mapping[:0],
                    positions[:n],
                    self.rotary_emb.cos_sin_cache,
                    self.padded_heads,
                    self.eps,
                    swa_metadata.block_size,
                    False,
                    self.kv_mxfp8,
                    True,
                    False,
                )
        super().forward_mqa(q, kv, positions, output)

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
        """Resolve paged KV and fuse dequantization, attention and the output
        store: MXFP8 on gfx950, inverse-RoPE'd BF16 rows on gfx942."""
        num_decodes = swa_metadata.num_decodes
        num_prefills = swa_metadata.num_prefills
        first = swa_metadata.num_decode_tokens
        tokens = swa_metadata.num_prefill_tokens
        seq_lens = swa_metadata.prefill_seq_lens
        gather_lens = swa_metadata.prefill_gather_lens
        starts = swa_metadata.query_start_loc
        assert seq_lens is not None and gather_lens is not None and starts is not None
        assert self.topk_indices_buffer is not None
        extra_table = None
        if self.compress_ratio:
            assert attn_metadata is not None and compressed_k_cache is not None
            extra_table = attn_metadata.block_table[
                num_decodes : num_decodes + num_prefills
            ]
        args = (
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
        )
        if get_cdna_version() == 3:
            assert output is not None and mxfp8_out is None
            torch.ops._rocm_C.fused_norm_rope_attn_rope_prefill_bf16(
                *args, self._prefill_rotate_q, output[:tokens]
            )
        else:
            assert mxfp8_out is not None
            torch.ops._rocm_C.fused_norm_rope_attn_rope_cast_prefill(
                *args, mxfp8_out[0][:tokens], mxfp8_out[1][:tokens]
            )

    def _reserve_prefill_workspace(self, q: torch.Tensor) -> None:
        # The CDNA3 prefill reads the caches in place and stores into the
        # layer output, so it needs no gather workspace. The base class only
        # skips it for the AITER sparse MLA, which is gfx950-only.
        if get_cdna_version() != 3:
            super()._reserve_prefill_workspace(q)
