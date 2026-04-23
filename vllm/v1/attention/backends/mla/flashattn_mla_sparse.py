# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""FlashAttention MLA Sparse Attention Backend.

This backend uses FlashAttention 4's MLA weight-absorbed kernel with topk
sparsity (gather_kv_indices) for models like DeepSeek-V3.2 that use
index-based sparse attention.

The FA4 MLA absorbed attention computes:
    O = softmax(scale * (Q_pe @ K_pe.T + Q_nope @ KV_c.T)) @ KV_c

where Q_pe has head_dim=64 and Q_nope/KV_c have head_dim_v=512.

For sparse attention, gather_kv_indices specifies per-query-token which
KV cache positions to attend to. Page table indirection is baked into these
indices via triton_convert_req_index_to_global_index, since FA4's sparse
MLA kernel does not support paged attention directly.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar

import numpy as np
import torch

from vllm.config import VllmConfig
from vllm.config.cache import CacheDType
from vllm.model_executor.layers.attention.mla_attention import (
    get_mla_dims,
)
from vllm.platforms.interface import DeviceCapability
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionCGSupport,
    AttentionLayer,
    AttentionMetadata,
    AttentionMetadataBuilder,
    AttentionType,
    CommonAttentionMetadata,
    MultipleOf,
    SparseMLAAttentionImpl,
)
from vllm.v1.attention.backends.fa_utils import is_fa_version_supported
from vllm.v1.attention.backends.mla.sparse_utils import (
    triton_convert_req_index_to_global_index,
)
from vllm.v1.kv_cache_interface import AttentionSpec
from vllm.vllm_flash_attn.cute.interface import (
    _flash_attn_fwd as _fa4_fwd,
)

if TYPE_CHECKING:
    from vllm.model_executor.models.deepseek_v2 import Indexer


class FlashAttentionMLASparseBackend(AttentionBackend):
    supported_dtypes: ClassVar[list[torch.dtype]] = [torch.float16, torch.bfloat16]
    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = [
        "auto",
        "float16",
        "bfloat16",
    ]

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:
        return [MultipleOf(16)]

    @staticmethod
    def get_name() -> str:
        return "FLASH_ATTN_MLA_SPARSE"

    @staticmethod
    def get_impl_cls() -> type["FlashAttentionMLASparseImpl"]:
        return FlashAttentionMLASparseImpl

    @staticmethod
    def get_builder_cls() -> type["FlashAttentionMLASparseMetadataBuilder"]:
        return FlashAttentionMLASparseMetadataBuilder

    @classmethod
    def get_supported_head_sizes(cls) -> list[int]:
        # kv_lora_rank (512) + qk_rope_head_dim (64) = 576
        return [576]

    @classmethod
    def is_mla(cls) -> bool:
        return True

    @classmethod
    def is_sparse(cls) -> bool:
        return True

    @classmethod
    def supports_compute_capability(cls, capability: DeviceCapability) -> bool:
        # FA4 MLA absorbed kernel requires SM100 (Blackwell) or SM110.
        return capability.major == 10

    @classmethod
    def supports_combination(
        cls,
        head_size: int,
        dtype: torch.dtype,
        kv_cache_dtype: CacheDType | None,
        block_size: int | None,
        use_mla: bool,
        has_sink: bool,
        use_sparse: bool,
        device_capability: DeviceCapability,
    ) -> str | None:
        if not is_fa_version_supported(4):
            return "FlashAttention 4 is not available"

        from vllm.config import get_current_vllm_config

        vllm_config = get_current_vllm_config()
        if vllm_config.model_config is not None:
            hf_text_config = vllm_config.model_config.hf_text_config

            # FA4 MLA absorbed kernel requires head_dim=64, head_dim_v=512
            qk_rope_head_dim = getattr(hf_text_config, "qk_rope_head_dim", 0)
            kv_lora_rank = getattr(hf_text_config, "kv_lora_rank", 0)
            if qk_rope_head_dim != 64:
                return (
                    "FlashAttention MLA Sparse kernel requires "
                    f"qk_rope_head_dim=64, but got {qk_rope_head_dim}"
                )
            if kv_lora_rank != 512:
                return (
                    "FlashAttention MLA Sparse kernel requires "
                    f"kv_lora_rank=512, but got {kv_lora_rank}"
                )

            if not hasattr(hf_text_config, "index_topk"):
                return "FlashAttention MLA Sparse requires model with index_topk config"

            num_q_heads = vllm_config.model_config.get_num_attention_heads(
                vllm_config.parallel_config
            )
            num_kv_heads = vllm_config.model_config.get_num_kv_heads(
                vllm_config.parallel_config
            )
            if num_kv_heads > 0 and num_q_heads // num_kv_heads != 128:
                return (
                    "FA4 MLA sparse kernel requires MQA with 128 query heads "
                    f"per KV head, but got {num_q_heads // num_kv_heads}"
                )
        return None

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,  # assumed to be 1 for MLA
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> tuple[int, ...]:
        return (num_blocks, block_size, head_size)


@dataclass
class FlashAttentionMLASparseMetadata(AttentionMetadata):
    num_reqs: int
    max_query_len: int
    max_seq_len: int
    num_actual_tokens: int

    query_start_loc: torch.Tensor
    slot_mapping: torch.Tensor
    block_table: torch.Tensor
    req_id_per_token: torch.Tensor

    # Sequence lengths for all requests (context + query)
    seq_lens: torch.Tensor

    block_size: int = 64
    topk_tokens: int = 2048


class FlashAttentionMLASparseMetadataBuilder(
    AttentionMetadataBuilder[FlashAttentionMLASparseMetadata]
):
    _cudagraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.UNIFORM_BATCH

    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ) -> None:
        self.vllm_config = vllm_config
        self.layer_names = layer_names
        self.kv_cache_spec = kv_cache_spec
        self.model_config = vllm_config.model_config
        self.device = device

        self.mla_dims = get_mla_dims(self.model_config)
        self.topk_tokens = vllm_config.model_config.hf_config.index_topk

        self.req_id_per_token_buffer = torch.empty(
            (vllm_config.scheduler_config.max_num_batched_tokens,),
            dtype=torch.int32,
            device=device,
        )

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> FlashAttentionMLASparseMetadata:
        cm = common_attn_metadata
        num_tokens = cm.num_actual_tokens

        # Build req_id_per_token mapping
        starts = np.asarray(cm.query_start_loc_cpu, dtype=np.int32)
        seg_lengths = np.diff(starts)
        req_id_per_token = np.repeat(
            np.arange(seg_lengths.shape[0], dtype=np.int32), seg_lengths
        )

        # Zero-fill for cudagraphs
        self.req_id_per_token_buffer.fill_(0)
        self.req_id_per_token_buffer[: req_id_per_token.shape[0]].copy_(
            torch.from_numpy(req_id_per_token), non_blocking=True
        )
        req_id_per_token_tensor = self.req_id_per_token_buffer[:num_tokens]

        return FlashAttentionMLASparseMetadata(
            num_reqs=cm.num_reqs,
            max_query_len=cm.max_query_len,
            max_seq_len=cm.max_seq_len,
            num_actual_tokens=cm.num_actual_tokens,
            query_start_loc=cm.query_start_loc,
            slot_mapping=cm.slot_mapping,
            block_table=cm.block_table_tensor,
            req_id_per_token=req_id_per_token_tensor,
            seq_lens=cm.seq_lens,
            block_size=self.kv_cache_spec.block_size,
            topk_tokens=self.topk_tokens,
        )


class FlashAttentionMLASparseImpl(
    SparseMLAAttentionImpl[FlashAttentionMLASparseMetadata]
):
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
        # MLA Specific Arguments
        topk_indices_buffer: torch.Tensor | None = None,
        indexer: "Indexer | None" = None,
        **mla_args,
    ) -> None:
        unsupported_features = [alibi_slopes, sliding_window, logits_soft_cap]
        if any(unsupported_features):
            raise NotImplementedError(
                "FlashAttentionMLASparseImpl does not support one of the "
                "following: alibi_slopes, sliding_window, logits_soft_cap"
            )

        if attn_type != AttentionType.DECODER:
            raise NotImplementedError(
                "Encoder self-attention and encoder/decoder cross-attention "
                "are not implemented for FlashAttentionMLASparseImpl"
            )

        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = float(scale)
        self.num_kv_heads = num_kv_heads
        self.kv_cache_dtype = kv_cache_dtype

        # MLA-specific dimensions
        self.kv_lora_rank: int = mla_args["kv_lora_rank"]
        self.qk_nope_head_dim: int = mla_args["qk_nope_head_dim"]
        self.qk_rope_head_dim: int = mla_args["qk_rope_head_dim"]

        assert indexer is not None, "Indexer required for sparse MLA"
        self.topk_indices_buffer: torch.Tensor | None = indexer.topk_indices_buffer

        max_tokens = self.topk_indices_buffer.shape[0]
        device = self.topk_indices_buffer.device

        # Pre-allocate cu_seqlens_k (all zeros) and seqused_k buffers.
        # cu_seqlens_k=0 means all batches share K at offset 0.
        # seqused_k=total_k tells the kernel how large K actually is,
        # preventing seqlen-based masking from zeroing the output.
        self.cu_seqlens_k_buffer = torch.zeros(
            max_tokens + 1, dtype=torch.int32, device=device
        )
        self.seqused_k_buffer = torch.zeros(
            max_tokens, dtype=torch.int32, device=device
        )

    def forward_mqa(
        self,
        q: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        kv_c_and_k_pe_cache: torch.Tensor,
        attn_metadata: FlashAttentionMLASparseMetadata,
        layer: AttentionLayer,  # noqa: ARG002
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if isinstance(q, tuple):
            q_nope, q_pe = q
        else:
            q_nope, q_pe = torch.split(
                q, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1
            )

        num_actual_toks = q_pe.shape[0]

        kv_c_cache = kv_c_and_k_pe_cache[..., : self.kv_lora_rank]
        k_pe_cache = kv_c_and_k_pe_cache[..., self.kv_lora_rank :]

        assert self.topk_indices_buffer is not None
        topk_indices = self.topk_indices_buffer[:num_actual_toks]
        topk_tokens = topk_indices.shape[1]

        total_k = k_pe_cache.shape[0] * k_pe_cache.shape[1]

        gather_kv_indices = triton_convert_req_index_to_global_index(
            attn_metadata.req_id_per_token[:num_actual_toks],
            attn_metadata.block_table,
            topk_indices,
            BLOCK_SIZE=attn_metadata.block_size,
            NUM_TOPK_TOKENS=topk_tokens,
        )

        k_flat = k_pe_cache.reshape(total_k, 1, self.qk_rope_head_dim)
        v_flat = kv_c_cache.reshape(total_k, 1, self.kv_lora_rank)

        num_reqs = attn_metadata.num_reqs
        cu_seqlens_k = self.cu_seqlens_k_buffer[: num_reqs + 1]
        seqused_k = self.seqused_k_buffer[:num_reqs]
        seqused_k.fill_(total_k)

        # FA4 CuTE kernel launches on a TVM-FFI managed stream (CUDA
        # default stream 0x0) which differs from PyTorch's current stream.
        # Sync before to ensure inputs (gather_kv_indices from triton) are
        # visible, and after to ensure the output is visible to PyTorch.
        # TODO: eliminate these syncs by making the CuTE kernel use
        # PyTorch's current stream (tvm_ffi.use_torch_stream doesn't
        # propagate to the compiled kernel's TVMFFIEnvGetStream).
        torch.accelerator.synchronize()
        attn_out, _ = _fa4_fwd(
            q_pe,
            k_flat,
            v_flat,
            qv=q_nope,
            max_seqlen_q=max(attn_metadata.max_query_len, 1),
            cu_seqlens_q=attn_metadata.query_start_loc,
            max_seqlen_k=total_k,
            cu_seqlens_k=cu_seqlens_k,
            seqused_k=seqused_k,
            softmax_scale=self.scale,
            causal=False,
            gather_kv_indices=gather_kv_indices,
        )
        torch.accelerator.synchronize()

        return attn_out, None
