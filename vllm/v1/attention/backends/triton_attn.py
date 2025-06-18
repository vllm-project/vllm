# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Attention layer with PagedAttention and Triton prefix prefill."""
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar, Optional

import torch

from vllm import _custom_ops as ops
from vllm import envs
from vllm.attention.backends.abstract import (AttentionBackend, AttentionImpl,
                                              AttentionMetadata, AttentionType)
from vllm.attention.ops.chunked_prefill_paged_decode import (
    chunked_prefill_paged_decode)
from vllm.attention.ops.paged_attn import PagedAttention
from vllm.attention.ops.triton_unified_attention import unified_attention
from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.v1.attention.backends.flash_attn import FlashAttentionMetadata
from vllm.v1.attention.backends.utils import (
    AttentionMetadataBuilder, CommonAttentionMetadata,
    make_local_attention_virtual_batches)
from vllm.v1.kv_cache_interface import AttentionSpec
from vllm.v1.worker.block_table import BlockTable

if TYPE_CHECKING:
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner

logger = init_logger(__name__)


@dataclass
class TritonAttentionMetadata:
    # NOTE(sang): Definition of context_len, query_len, and seq_len.
    # |---------- N-1 iteration --------|
    # |---------------- N iteration ---------------------|
    # |- tokenA -|......................|-- newTokens ---|
    # |---------- context_len ----------|
    # |-------------------- seq_len ---------------------|
    #                                   |-- query_len ---|

    num_actual_tokens: int  # Number of tokens excluding padding.
    max_query_len: int
    query_start_loc: torch.Tensor
    max_seq_len: int
    seq_lens: torch.Tensor
    block_table: torch.Tensor
    slot_mapping: torch.Tensor

    # For cascade attention.
    use_cascade: bool
    common_prefix_len: int
    cu_prefix_query_lens: Optional[torch.Tensor]
    prefix_kv_lens: Optional[torch.Tensor]
    suffix_kv_lens: Optional[torch.Tensor]

    # Optional aot scheduling
    scheduler_metadata: Optional[torch.Tensor] = None
    prefix_scheduler_metadata: Optional[torch.Tensor] = None

    # for local attention
    @dataclass
    class LocalAttentionMetadata:
        local_query_start_loc: torch.Tensor
        local_seqused_k: torch.Tensor
        local_block_table: torch.Tensor
        local_max_query_len: int
        local_max_seq_len: int
        local_scheduler_metadata: Optional[torch.Tensor]

    local_attn_metadata: Optional[LocalAttentionMetadata] = None


class TritonAttentionMetadataBuilder(
        AttentionMetadataBuilder[TritonAttentionMetadata]):
    full_cudagraph_supported: ClassVar[bool] = True

    def __init__(self, runner: "GPUModelRunner", kv_cache_spec: AttentionSpec,
                 block_table: BlockTable):
        self.runner = runner
        self.block_size = kv_cache_spec.block_size
        self.kv_cache_spec = kv_cache_spec
        self.block_table = block_table

    def build_for_cudagraph_capture(
        self, common_attn_metadata: CommonAttentionMetadata
    ) -> TritonAttentionMetadata:
        attn_metadata = self.build(0, common_attn_metadata)
        # When doing full graph capture, setting seq_lens to
        # max_model_len will cause graph capture to be extremely
        # slow, so here we set it to 1.
        attn_metadata.seq_lens.fill_(1)
        return attn_metadata

    def build(
        self, common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata
    ) -> TritonAttentionMetadata:
        num_reqs = common_attn_metadata.num_reqs
        num_actual_tokens = common_attn_metadata.num_actual_tokens
        max_query_len = common_attn_metadata.max_query_len

        max_seq_len = int(self.runner.seq_lens_np[:num_reqs].max())
        query_start_loc = common_attn_metadata.query_start_loc
        seq_lens = common_attn_metadata.seq_lens
        block_table = self.block_table
        block_table_tensor = block_table.get_device_tensor()[:num_reqs]

        block_table.slot_mapping[:num_actual_tokens].copy_(
            block_table.slot_mapping_cpu[:num_actual_tokens],
            non_blocking=True)
        # Fill unused with -1. Needed for reshape_and_cache in full cuda graph
        # mode.
        block_table.slot_mapping[num_actual_tokens:].fill_(-1)

        slot_mapping = block_table.slot_mapping[:num_actual_tokens]

        # for local attention
        local_attn_metadata = None
        if self.runner.attention_chunk_size is not None:
            seqlens_q_local_np, virt_q_cu_seqlens_np, virt_k_seqlens_np, \
                virt_block_table_tensor = make_local_attention_virtual_batches(
                    self.runner.attention_chunk_size,
                    self.runner.query_start_loc_np[:num_reqs + 1],
                    self.runner.seq_lens_np[:num_reqs],
                    block_table_tensor,
                    self.block_size,
                )
            local_query_start_loc = torch.from_numpy(virt_q_cu_seqlens_np).to(
                self.runner.device, non_blocking=True)
            local_seqused_k = torch.from_numpy(virt_k_seqlens_np).to(
                self.runner.device, non_blocking=True)
            local_max_query_len = seqlens_q_local_np.max()
            local_max_seq_len = virt_k_seqlens_np.max()

            local_attn_metadata = TritonAttentionMetadata \
                        .LocalAttentionMetadata(
                local_query_start_loc=local_query_start_loc,
                local_seqused_k=local_seqused_k,
                local_block_table=virt_block_table_tensor,
                local_max_query_len=local_max_query_len,
                local_max_seq_len=local_max_seq_len,
                local_scheduler_metadata=None,
            )

        use_cascade = common_prefix_len > 0

        if use_cascade:
            cu_prefix_query_lens = torch.tensor([0, num_actual_tokens],
                                                dtype=torch.int32,
                                                device=self.runner.device)
            prefix_kv_lens = torch.tensor([common_prefix_len],
                                          dtype=torch.int32,
                                          device=self.runner.device)
            suffix_kv_lens = (self.runner.seq_lens_np[:num_reqs] -
                              common_prefix_len)
            suffix_kv_lens = torch.from_numpy(suffix_kv_lens).to(
                self.runner.device)
        else:
            cu_prefix_query_lens = None
            prefix_kv_lens = None
            suffix_kv_lens = None
            prefix_scheduler_metadata = None

        attn_metadata = TritonAttentionMetadata(
            num_actual_tokens=num_actual_tokens,
            max_query_len=max_query_len,
            query_start_loc=query_start_loc,
            max_seq_len=max_seq_len,
            seq_lens=seq_lens,
            block_table=block_table_tensor,
            slot_mapping=slot_mapping,
            use_cascade=use_cascade,
            common_prefix_len=common_prefix_len,
            cu_prefix_query_lens=cu_prefix_query_lens,
            prefix_kv_lens=prefix_kv_lens,
            suffix_kv_lens=suffix_kv_lens,
            local_attn_metadata=local_attn_metadata,
            prefix_scheduler_metadata=prefix_scheduler_metadata,
        )
        return attn_metadata

    def can_run_in_cudagraph(
            self, common_attn_metadata: CommonAttentionMetadata) -> bool:
        # Full CUDA Graph always supported
        return True


class TritonAttentionBackend(AttentionBackend):

    accept_output_buffer: bool = True

    @staticmethod
    def get_supported_head_sizes() -> list[int]:
        return [32, 64, 96, 128, 160, 192, 224, 256]

    @staticmethod
    def get_name() -> str:
        return "TRITON_ATTN_VLLM_V1"

    @staticmethod
    def get_impl_cls() -> type["TritonAttentionImpl"]:
        return TritonAttentionImpl

    @staticmethod
    def get_metadata_cls() -> type["AttentionMetadata"]:
        return TritonAttentionMetadata

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
    ) -> tuple[int, ...]:
        if block_size % 16 != 0:
            raise ValueError("Block size must be a multiple of 16.")
        return (2, num_blocks, block_size, num_kv_heads, head_size)

    @staticmethod
    def use_cascade_attention(*args, **kwargs) -> bool:
        return False

    @staticmethod
    def get_builder_cls() -> type["TritonAttentionMetadataBuilder"]:
        return TritonAttentionMetadataBuilder


class TritonAttentionImpl(AttentionImpl):

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int,
        alibi_slopes: Optional[list[float]],
        sliding_window: Optional[int],
        kv_cache_dtype: str,
        blocksparse_params: Optional[dict[str, Any]] = None,
        logits_soft_cap: Optional[float] = None,
        attn_type: AttentionType = AttentionType.DECODER,
        kv_sharing_target_layer_name: Optional[int] = None,
        use_irope: bool = False,
    ) -> None:
        if blocksparse_params is not None:
            raise ValueError(
                "TritonAttention does not support block-sparse attention.")
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = float(scale)
        self.num_kv_heads = num_kv_heads
        if alibi_slopes is not None:
            alibi_slopes = torch.tensor(alibi_slopes, dtype=torch.float32)
        self.alibi_slopes = alibi_slopes
        if sliding_window is None:
            self.sliding_window = (-1, -1)
        else:
            self.sliding_window = (sliding_window - 1, 0)
        self.kv_cache_dtype = kv_cache_dtype
        if logits_soft_cap is None:
            # In flash-attn, setting logits_soft_cap as 0 means no soft cap.
            logits_soft_cap = 0
        self.logits_soft_cap = logits_soft_cap
        self.kv_sharing_target_layer_name = kv_sharing_target_layer_name

        self.use_irope = use_irope

        self.num_queries_per_kv = self.num_heads // self.num_kv_heads

        support_head_sizes = TritonAttentionBackend.get_supported_head_sizes()
        if head_size not in support_head_sizes:
            raise ValueError(
                f"Head size {head_size} is not supported by TritonAttention. "
                f"Supported head sizes are: {support_head_sizes}.")

        if attn_type != AttentionType.DECODER:
            raise NotImplementedError("Encoder self-attention and "
                                      "encoder/decoder cross-attention "
                                      "are not implemented for "
                                      "TritonAttentionImpl")

        self.fp8_dtype = current_platform.fp8_dtype()
        self.force_prefill_decode_attn = \
            envs.VLLM_V1_USE_PREFILL_DECODE_ATTENTION

    def forward(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: FlashAttentionMetadata,
        output: Optional[torch.Tensor] = None,
        output_scale: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass with FlashAttention.

        Args:
            query: shape = [num_tokens, num_heads, head_size]
            key: shape = [num_tokens, num_kv_heads, head_size]
            value: shape = [num_tokens, num_kv_heads, head_size]
            kv_cache = [2, num_blocks, block_size, num_kv_heads, head_size]
            attn_metadata: Metadata for attention.
        Returns:
            shape = [num_tokens, num_heads * head_size]
        """
        assert output is not None, "Output tensor must be provided."

        if output_scale is not None:
            raise NotImplementedError(
                "fused output quantization is not yet supported"
                " for TritonAttentionImpl")

        if attn_metadata is None:
            # Profiling run.
            return output

        assert attn_metadata.use_cascade is False

        # IMPORTANT!
        # NOTE(woosuk): With piece-wise CUDA graphs, this method is executed in
        # eager-mode PyTorch. Thus, we need to be careful about any CPU overhead
        # in this method. For example, `view` and `slice` (or `[:n]`) operations
        # are surprisingly slow even in the case they do not invoke any GPU ops.
        # Minimize the PyTorch ops in this method as much as possible.
        # Whenever making a change in this method, please benchmark the
        # performance to make sure it does not introduce any overhead.

        use_prefill_decode_attn = self.force_prefill_decode_attn
        num_actual_tokens = attn_metadata.num_actual_tokens

        if use_prefill_decode_attn:
            key_cache, value_cache = PagedAttention.split_kv_cache(
                kv_cache, self.num_kv_heads, self.head_size)
        else:
            key_cache, value_cache = kv_cache.unbind(0)

        if self.kv_sharing_target_layer_name is None:
            # Reshape the input keys and values and store them in the cache.
            # Skip this if sharing KV cache with an earlier attention layer.
            if use_prefill_decode_attn:
                PagedAttention.write_to_paged_cache(
                    key,
                    value,
                    key_cache,
                    value_cache,
                    attn_metadata.slot_mapping,
                    self.kv_cache_dtype,
                    layer._k_scale,
                    layer._v_scale,
                )
            else:
                torch.ops._C_cache_ops.reshape_and_cache_flash(
                    key,
                    value,
                    key_cache,
                    value_cache,
                    attn_metadata.slot_mapping,
                    self.kv_cache_dtype,
                    layer._k_scale,
                    layer._v_scale,
                )

        if self.kv_cache_dtype.startswith("fp8"):
            key_cache = key_cache.view(self.fp8_dtype)
            value_cache = value_cache.view(self.fp8_dtype)
            num_tokens, num_heads, head_size = query.shape
            assert layer._q_scale == 1.0, \
                "A non 1.0 q_scale is not currently supported."
            if not current_platform.is_rocm():
                # Skip Q quantization on ROCm, since dequantizing back to
                # f32 in the attention kernel is not supported.
                query, _ = ops.scaled_fp8_quant(
                    query.reshape(
                        (num_tokens, num_heads * head_size)).contiguous(),
                    layer._q_scale)
                query = query.reshape((num_tokens, num_heads, head_size))

        use_local_attn = \
            (self.use_irope and attn_metadata.local_attn_metadata is not None)

        if use_local_attn:
            assert attn_metadata.local_attn_metadata is not None
            local_metadata = attn_metadata.local_attn_metadata
            cu_seqlens_q = local_metadata.local_query_start_loc
            seqused_k = local_metadata.local_seqused_k
            max_seqlen_q = local_metadata.local_max_query_len
            max_seqlen_k = local_metadata.local_max_seq_len
            block_table = local_metadata.local_block_table
        else:
            cu_seqlens_q = attn_metadata.query_start_loc
            seqused_k = attn_metadata.seq_lens
            max_seqlen_q = attn_metadata.max_query_len
            max_seqlen_k = attn_metadata.max_seq_len
            block_table = attn_metadata.block_table

        if use_prefill_decode_attn:
            # Compute attention and update output up to `num_actual_tokens`.
            chunked_prefill_paged_decode(query=query[:num_actual_tokens],
                                         key=key[:num_actual_tokens],
                                         value=value[:num_actual_tokens],
                                         output=output[:num_actual_tokens],
                                         kv_cache_dtype=self.kv_cache_dtype,
                                         key_cache=key_cache,
                                         value_cache=value_cache,
                                         block_table=block_table,
                                         query_start_loc=cu_seqlens_q,
                                         seq_lens=seqused_k,
                                         max_seq_len=max_seqlen_k,
                                         max_query_len=max_seqlen_q,
                                         k_scale=layer._k_scale,
                                         v_scale=layer._v_scale,
                                         alibi_slopes=self.alibi_slopes,
                                         sliding_window=self.sliding_window[0],
                                         sm_scale=self.scale)

        else:
            descale_shape = (cu_seqlens_q.shape[0] - 1, key.shape[1])

            unified_attention(
                q=query[:num_actual_tokens],
                k=key_cache,
                v=value_cache,
                out=output[:num_actual_tokens],
                cu_seqlens_q=cu_seqlens_q,
                max_seqlen_q=max_seqlen_q,
                seqused_k=seqused_k,
                max_seqlen_k=max_seqlen_k,
                softmax_scale=self.scale,
                causal=True,
                alibi_slopes=self.alibi_slopes,
                window_size=self.sliding_window,
                block_table=block_table,
                softcap=self.logits_soft_cap,
                q_descale=None,  # Not supported
                k_descale=layer._k_scale.expand(descale_shape),
                v_descale=layer._v_scale.expand(descale_shape),
            )

        return output
