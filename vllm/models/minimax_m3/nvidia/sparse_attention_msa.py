# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MSA (SM100/Blackwell) block-sparse attention for MiniMax M3.

Prefill attends with ``fmha_sm100`` (``build_k2q_csr`` + ``sparse_atten_func``).
Decode uses Triton split-K by default, with an opt-in CUTLASS ``fmha_sm100``
path for regular decode and speculative verification.
"""

from dataclasses import dataclass

import torch

from vllm.config import VllmConfig
from vllm.config.attention import MiniMaxM3MSADecodeBackend
from vllm.forward_context import get_forward_context
from vllm.logger import init_logger
from vllm.models.minimax_m3.common.ops.sparse_attn import (
    SPARSE_BLOCK_SIZE,
    minimax_m3_sparse_attn_decode,
)
from vllm.models.minimax_m3.common.sparse_attention import (
    MiniMaxM3SparseBackend,
    MiniMaxM3SparseDecodeMetadata,
    MiniMaxM3SparseImpl,
    MiniMaxM3SparseMetadata,
    MiniMaxM3SparseMetadataBuilder,
)
from vllm.models.minimax_m3.nvidia.msa_cutlass_sparse_decode import (
    MSACutlassDecodeMetadata,
    MSACutlassDecodePlanCache,
    msa_cutlass_sparse_decode,
    prepare_decode_metadata,
    should_prepare_decode_metadata,
    supports_cutlass_sparse_decode,
)
from vllm.v1.attention.backend import (
    AttentionLayer,
    CommonAttentionMetadata,
)
from vllm.v1.kv_cache_interface import AttentionSpec

logger = init_logger(__name__)


class MiniMaxM3SparseMSABackend(MiniMaxM3SparseBackend):
    """MiniMax M3 backend with NVIDIA MSA-specific decode metadata."""

    @staticmethod
    def get_builder_cls() -> type["MiniMaxM3SparseMSAMetadataBuilder"]:
        return MiniMaxM3SparseMSAMetadataBuilder


class MiniMaxM3SparseCutlassBackend(MiniMaxM3SparseMSABackend):
    """Attention-backend alias selecting CUTLASS MSA sparse decode."""

    @staticmethod
    def get_name() -> str:
        return "CUTLASS_MSA"


class MiniMaxM3SparseTritonBackend(MiniMaxM3SparseMSABackend):
    """Attention-backend alias selecting Triton MSA sparse decode."""

    @staticmethod
    def get_name() -> str:
        return "TRITON_MSA"


@dataclass
class MiniMaxM3SparseMSADecodeMetadata(MiniMaxM3SparseDecodeMetadata):
    msa_cutlass: MSACutlassDecodeMetadata | None = None


class MiniMaxM3SparseMSAMetadataBuilder(MiniMaxM3SparseMetadataBuilder):
    """Prepare MSA plans only for decode shapes supported by ``fmha_sm100``."""

    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ) -> None:
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        config = vllm_config.model_config.hf_text_config
        tp_size = vllm_config.parallel_config.tensor_parallel_size
        self.num_q_heads = config.num_attention_heads // tp_size
        self.num_kv_heads = kv_cache_spec.num_kv_heads
        self.topk_blocks = config.sparse_attention_config["sparse_topk_blocks"]
        # AttentionSpec stores every FP8 mode as uint8, so retain the configured
        # format to distinguish E4M3 (supported) from E5M2 before planning.
        self.kv_cache_dtype = vllm_config.cache_config.cache_dtype
        self.decode_backend = vllm_config.attention_config.minimax_m3_msa_decode_backend
        self.msa_cutlass_plan_cache = MSACutlassDecodePlanCache()

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> MiniMaxM3SparseMetadata:
        metadata = super().build(
            common_prefix_len,
            common_attn_metadata,
            fast_build,
        )
        decode = metadata.decode
        if decode is None:
            return metadata

        msa_cutlass = None
        if should_prepare_decode_metadata(
            metadata.num_decodes,
            decode.decode_query_len,
            decode_backend=self.decode_backend,
            num_q_heads=self.num_q_heads,
            num_kv_heads=self.num_kv_heads,
            kv_cache_dtype=self.kv_cache_dtype,
            page_size=SPARSE_BLOCK_SIZE,
            topk_blocks=self.topk_blocks,
        ):
            seq_lens_cpu = common_attn_metadata.seq_lens_cpu_upper_bound
            assert seq_lens_cpu is not None
            msa_cutlass = prepare_decode_metadata(
                decode.block_table,
                decode.seq_lens,
                seq_lens_cpu[: metadata.num_decodes],
                decode.decode_query_len,
                num_q_heads=self.num_q_heads,
                num_kv_heads=self.num_kv_heads,
                page_size=SPARSE_BLOCK_SIZE,
                topk_blocks=self.topk_blocks,
                plan_cache=self.msa_cutlass_plan_cache,
            )
        metadata.decode = MiniMaxM3SparseMSADecodeMetadata(
            seq_lens=decode.seq_lens,
            block_table=decode.block_table,
            decode_query_len=decode.decode_query_len,
            msa_cutlass=msa_cutlass,
        )
        return metadata


class MiniMaxM3SparseMSAImpl(MiniMaxM3SparseImpl):
    """MSA block-sparse attention with guarded CUTLASS sparse decode."""

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int | None = None,
        kv_cache_dtype: str = "auto",
        *,
        topk_blocks: int,
        sparse_block_size: int,
        msa_decode_backend: MiniMaxM3MSADecodeBackend = "triton",
    ) -> None:
        super().__init__(
            num_heads,
            head_size,
            scale,
            num_kv_heads,
            kv_cache_dtype,
            topk_blocks=topk_blocks,
            sparse_block_size=sparse_block_size,
        )
        self.use_cutlass_decode = supports_cutlass_sparse_decode(
            decode_backend=msa_decode_backend,
            num_q_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads,
            kv_cache_dtype=self.kv_cache_dtype,
            page_size=self.block_size,
            topk_blocks=self.topk_blocks,
        )
        logger.info_once(
            "MiniMax M3 MSA sparse decode selected %s",
            "CUTLASS" if self.use_cutlass_decode else "Triton",
        )

    def should_use_msa_decode(self, layer_name: str) -> bool:
        if not self.use_cutlass_decode:
            return False
        attn_metadata = get_forward_context().attn_metadata
        if not isinstance(attn_metadata, dict):
            return False
        main_md = attn_metadata[layer_name]
        if not isinstance(main_md, MiniMaxM3SparseMetadata):
            return False
        decode = main_md.decode
        return (
            isinstance(decode, MiniMaxM3SparseMSADecodeMetadata)
            and decode.msa_cutlass is not None
        )

    def forward(
        self,
        layer: AttentionLayer,
        query: torch.Tensor,
        kv_cache: torch.Tensor,
        output: torch.Tensor,
        *,
        query_fp8: torch.Tensor | None = None,
    ) -> torch.Tensor:
        attn_metadata = get_forward_context().attn_metadata
        if not isinstance(attn_metadata, dict):
            return output  # profiling run; caches unbound
        main_md = attn_metadata[layer.layer_name]  # type: ignore[attr-defined]
        assert isinstance(main_md, MiniMaxM3SparseMetadata)

        nd = main_md.num_decode_tokens
        num_tokens = main_md.num_actual_tokens
        # Indexer top-k from the shared token-major buffer [total_q, H, MK]; the
        # kernels want [H, tokens, MK], so slice tokens on dim 0 then transpose.
        topk = layer.topk_indices_buffer  # type: ignore[attr-defined]
        assert topk is not None
        hd = self.head_size
        q = query[:num_tokens].view(-1, self.num_heads, hd)
        out = output[:num_tokens].view(-1, self.num_heads, hd)
        kv_cache = (
            kv_cache.view(self.kv_cache_fp8_dtype) if self.use_fp8_kv else kv_cache
        )
        k_scale = getattr(layer, "_k_scale", None) if self.use_fp8_kv else None
        v_scale = getattr(layer, "_v_scale", None) if self.use_fp8_kv else None

        # Decode [:nd]: CUTLASS for planned shapes, otherwise Triton.
        if main_md.num_decodes > 0:
            d = main_md.decode
            assert d is not None
            msa_metadata = (
                d.msa_cutlass
                if isinstance(d, MiniMaxM3SparseMSADecodeMetadata)
                else None
            )
            if self.use_cutlass_decode and msa_metadata is not None:
                assert query_fp8 is not None
                msa_cutlass_sparse_decode(
                    query_fp8[:nd].view(-1, self.num_heads, hd),
                    kv_cache,
                    topk[:nd],
                    out[:nd],
                    msa_metadata,
                    scale=self.scale,
                    q_scale_float=getattr(layer, "_q_scale_float", 1.0),
                    k_scale_float=getattr(layer, "_k_scale_float", 1.0),
                    v_scale_float=getattr(layer, "_v_scale_float", 1.0),
                )
            else:
                minimax_m3_sparse_attn_decode(
                    q[:nd],
                    kv_cache,
                    topk[:nd].transpose(0, 1),
                    d.block_table,
                    d.seq_lens,
                    self.num_kv_heads,
                    self.scale,
                    out[:nd],
                    d.decode_query_len,
                    k_scale=k_scale,
                    v_scale=v_scale,
                )

        # Prefill [nd:]: MSA sparse FMHA over the selected blocks.
        if main_md.num_prefills > 0:
            from vllm.third_party.fmha_sm100.sparse import (
                build_k2q_csr,
                sparse_atten_func,
            )

            p = main_md.prefill
            assert p is not None
            # [H, prefill, MK] transposed view; build_k2q_csr consumes the
            # strided view directly (topK stays innermost-contiguous).
            prefill_topk = topk[nd:num_tokens].transpose(0, 1)
            qp = q[nd:]
            k_cache, v_cache = kv_cache.split(self.head_size, dim=-1)
            k2q_row_ptr, k2q_q_indices, schedule = build_k2q_csr(
                prefill_topk,
                p.cu_seqlens_q,
                p.cu_seqlens_k,
                SPARSE_BLOCK_SIZE,
                total_k=0,
                max_seqlen_k=p.max_seq_len,
                max_seqlen_q=p.max_query_len,
                total_rows=p.total_kv_blocks,
                qhead_per_kv=qp.shape[1] // self.num_kv_heads,
                return_schedule=True,
            )
            sparse_atten_func(
                qp,
                k_cache,
                v_cache,
                k2q_row_ptr,
                k2q_q_indices,
                topK=self.topk_blocks,
                blk_kv=SPARSE_BLOCK_SIZE,
                causal=True,
                softmax_scale=self.scale,
                cu_seqlens_q=p.cu_seqlens_q,
                cu_seqlens_k=p.cu_seqlens_k,
                max_seqlen_q=p.max_query_len,
                max_seqlen_k=p.max_seq_len,
                page_table=p.block_table,
                seqused_k=p.seq_lens,
                schedule=schedule,
                out=out[nd:],
            )
        return output
