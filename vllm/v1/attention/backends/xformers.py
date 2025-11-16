# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Attention layer with XFormersAttention."""

from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar, Optional

import torch

from vllm.attention.backends.abstract import (AttentionBackend, AttentionImpl,
                                              AttentionMetadata, AttentionType)
from vllm.attention.ops.triton_unified_attention import unified_attention
from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.v1.attention.backends.utils import (
    AttentionMetadataBuilder, CommonAttentionMetadata,
    reorder_batch_to_split_decodes_and_prefills, split_decodes_and_prefills)
from vllm.v1.kv_cache_interface import AttentionSpec
try:
    from xformers import ops as xops
    from xformers.ops.fmha.attn_bias import (
        AttentionBias, PagedBlockDiagonalCausalWithOffsetPaddedKeysMask, LowerTriangularMask)

    XFORMERS_AVAILABLE = True
except ImportError:
    XFORMERS_AVAILABLE = False

if TYPE_CHECKING:
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.worker.gpu_input_batch import InputBatch

from vllm import _custom_ops as ops

logger = init_logger(__name__)


ATTN_MASK = None


class XFormersAttentionBackend(AttentionBackend):

    accept_output_buffer: bool = True

    @classmethod
    def get_supported_dtypes(cls) -> list[torch.dtype]:
        return [torch.float16, torch.bfloat16]

    @classmethod
    def get_supported_head_sizes(cls) -> list[int]:
        return [
            32,
            40,
            48,
            56,
            64,
            72,
            80,
            88,
            96,
            104,
            112,
            120,
            128,
            136,
            144,
            152,
            160,
            168,
            176,
            184,
            192,
            200,
            208,
            216,
            224,
            232,
            240,
            248,
            256,
        ]

    @classmethod
    def validate_head_size(cls, head_size: int) -> None:
        supported_head_sizes = cls.get_supported_head_sizes()
        if head_size not in supported_head_sizes:
            attn_type = cls.__name__.removesuffix("Backend")
            raise ValueError(
                f"Head size {head_size} is not supported by {attn_type}. "
                f"Supported head sizes are: {supported_head_sizes}. "
                "Set VLLM_ATTENTION_BACKEND=FLEX_ATTENTION to use "
                "FlexAttention backend which supports all head sizes.")

    @staticmethod
    def get_name() -> str:
        return "XFORMERS_VLLM_V1"

    @staticmethod
    def get_impl_cls() -> type["XFormersAttentionImpl"]:
        return XFormersAttentionImpl

    @staticmethod
    def get_metadata_cls() -> type["AttentionMetadata"]:
        return XFormersAttentionMetadata

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
    def get_builder_cls() -> type["XFormersAttentionMetadataBuilder"]:
        return XFormersAttentionMetadataBuilder

    @staticmethod
    def use_cascade_attention(*args, **kwargs) -> bool:
        return False


@dataclass
class XFormersAttentionMetadata:
    num_actual_tokens: int  # Number of tokens excluding padding.
    max_query_len: int
    query_start_loc: torch.Tensor
    max_seq_len: int
    seq_lens: torch.Tensor
    block_table: torch.Tensor
    slot_mapping: torch.Tensor

    num_prefill_tokens: int = 0
    num_decode_tokens: int = 0
    num_prefills: int = 0
    num_decodes: int = 0

    # Biases for different attention types.
    attn_bias: Optional["AttentionBias"] = None

    # Training mode flag
    is_training: bool = False

    # Training attention mask - dict with 'masks' (list of tensors) and 'seq_lens' (list of ints)
    training_attention_mask: Optional[dict] = None

    # Self-attention prefill/decode metadata cache
    _cached_prefill_metadata: Optional["XFormersAttentionMetadata"] = None
    _cached_decode_metadata: Optional["XFormersAttentionMetadata"] = None

    @property
    def prefill_metadata(self) -> Optional["XFormersAttentionMetadata"]:
        if self.num_prefills == 0:
            return None

        if self._cached_prefill_metadata is not None:
            # Recover cached prefill-phase attention
            # metadata structure
            return self._cached_prefill_metadata

        q_start_loc = self.query_start_loc[self.num_decodes:]
        q_seqlens = torch.diff(q_start_loc)
        kv_seqlens = self.seq_lens[self.num_decodes:]
        # Construct & cache prefill-phase attention metadata structure
        self._cached_prefill_metadata = XFormersAttentionMetadata(
            num_actual_tokens=self.num_prefill_tokens,
            max_query_len=int(q_seqlens.max().item()),
            query_start_loc=q_start_loc - q_start_loc[0],
            max_seq_len=int(kv_seqlens.max().item()),
            seq_lens=kv_seqlens,
            block_table=self.block_table[self.num_decodes:],
            slot_mapping=self.slot_mapping[self.num_decode_tokens:],
        )
        return self._cached_prefill_metadata

    @property
    def decode_metadata(self) -> Optional["XFormersAttentionMetadata"]:
        if self.num_decode_tokens == 0:
            return None

        if self._cached_decode_metadata is not None:
            # Recover cached decode-phase attention
            # metadata structure
            return self._cached_decode_metadata

        q_start_loc = self.query_start_loc
        q_seqlens = torch.diff(q_start_loc)
        decode_kv_seqlens = self.seq_lens[:self.num_decodes]
        # Construct & cache decode-phase attention metadata structure
        self._cached_decode_metadata = XFormersAttentionMetadata(
            num_actual_tokens=self.num_decode_tokens,
            max_query_len=int(q_seqlens[:self.num_decodes].max().item()),
            query_start_loc=q_start_loc[:self.num_decodes + 1],
            max_seq_len=int(decode_kv_seqlens.max().item()),
            seq_lens=decode_kv_seqlens,
            block_table=self.block_table[:self.num_decodes],
            slot_mapping=self.slot_mapping[:self.num_decode_tokens],
            attn_bias=self.attn_bias,
        )
        return self._cached_decode_metadata


class XFormersAttentionMetadataBuilder(
        AttentionMetadataBuilder[XFormersAttentionMetadata]):

    reorder_batch_threshold: ClassVar[int] = 1

    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ):
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)

        assert XFORMERS_AVAILABLE
        self.block_size = kv_cache_spec.block_size
        self._num_decodes = 0
        self._num_decode_tokens = 0

    def reorder_batch(self, input_batch: "InputBatch",
                      scheduler_output: "SchedulerOutput") -> bool:
        return reorder_batch_to_split_decodes_and_prefills(
            input_batch,
            scheduler_output,
            decode_threshold=self.reorder_batch_threshold)

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
        is_training: bool = False,
        training_attention_mask: Optional[dict] = None,
    ) -> XFormersAttentionMetadata:
        num_decodes, num_prefills, num_decode_tokens, num_prefill_tokens = (
            split_decodes_and_prefills(
                common_attn_metadata,
                decode_threshold=self.reorder_batch_threshold))

        num_actual_tokens = common_attn_metadata.num_actual_tokens
        q_start_loc = common_attn_metadata.query_start_loc
        q_seqlens = torch.diff(q_start_loc)
        max_query_len = common_attn_metadata.max_query_len
        kv_seqlens = common_attn_metadata.seq_lens
        max_seq_len = common_attn_metadata.max_seq_len
        block_table = common_attn_metadata.block_table_tensor
        slot_mapping = common_attn_metadata.slot_mapping

        bias = None
        if num_decodes > 0 and not is_training:
            # Construct the decoder bias.
            # Skip for training since we won't use KV cache
            decode_q_seqlens = q_seqlens[:num_decodes]
            decode_kv_seqlens = kv_seqlens[:num_decodes]
            bias = (
                PagedBlockDiagonalCausalWithOffsetPaddedKeysMask.from_seqlens(
                    q_seqlen=decode_q_seqlens.tolist(),
                    kv_seqlen=decode_kv_seqlens.tolist(),
                    page_size=self.block_size,
                    block_tables=block_table[:num_decodes],
                    device=block_table.device,
                ))

        return XFormersAttentionMetadata(
            num_actual_tokens=num_actual_tokens,
            num_prefill_tokens=num_prefill_tokens,
            num_decode_tokens=num_decode_tokens,
            num_prefills=num_prefills,
            num_decodes=num_decodes,
            max_query_len=max_query_len,
            query_start_loc=q_start_loc,
            max_seq_len=max_seq_len,
            seq_lens=kv_seqlens,
            block_table=block_table,
            slot_mapping=slot_mapping,
            attn_bias=bias,
            is_training=is_training,
            training_attention_mask=training_attention_mask,
        )


class XFormersAttentionImpl(AttentionImpl):

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int,
        alibi_slopes: Optional[list[float]],
        sliding_window: Optional[int],
        kv_cache_dtype: str,
        logits_soft_cap: Optional[float] = None,
        attn_type: AttentionType = AttentionType.DECODER,
        kv_sharing_target_layer_name: Optional[str] = None,
    ) -> None:
        if kv_sharing_target_layer_name is not None:
            raise NotImplementedError("KV sharing is not supported in V0.")
        if alibi_slopes is not None:
            raise NotImplementedError(
                "XFormers does not support alibi slopes yet.")
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = float(scale)
        self.num_kv_heads = num_kv_heads
        self.num_queries_per_kv = self.num_heads // self.num_kv_heads
        self.kv_cache_dtype = kv_cache_dtype
        self.kv_sharing_target_layer_name = kv_sharing_target_layer_name
        if alibi_slopes is not None:
            alibi_slopes = torch.tensor(alibi_slopes, dtype=torch.float32)
        self.alibi_slopes = alibi_slopes
        if sliding_window is None:
            self.sliding_window = (-1, -1)
        else:
            self.sliding_window = (sliding_window - 1, 0)
        if logits_soft_cap is None:
            # Setting logits_soft_cap to 0 means no soft cap.
            logits_soft_cap = 0
        self.logits_soft_cap = logits_soft_cap

        XFormersAttentionBackend.validate_head_size(head_size)

        if attn_type != AttentionType.DECODER:
            raise NotImplementedError("Encoder self-attention and "
                                      "encoder/decoder cross-attention "
                                      "are not implemented for "
                                      "XFormersAttentionImpl.")

    def forward(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: XFormersAttentionMetadata,
        output: Optional[torch.Tensor] = None,
        output_scale: Optional[torch.Tensor] = None,
        output_block_scale: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass with XFormers.

        Args:
            query: shape = [num_tokens, num_heads, head_size]
            key: shape = [num_tokens, num_kv_heads, head_size]
            value: shape = [num_tokens, num_kv_heads, head_size]
            kv_cache: shape =
                [2, num_blocks, block_size, num_kv_heads, head_size]
            attn_metadata: Metadata for attention.
        Returns:
            shape = [num_tokens, num_heads * head_size]
        """
        assert output is not None, "Output tensor must be provided."

        if output_scale is not None or output_block_scale is not None:
            raise NotImplementedError(
                "fused output quantization is not yet supported"
                " for XFormersAttentionImpl")

        if attn_metadata is None:
            # Profiling run.
            return output

        # Dispatch to training-specific forward if in training mode
        if attn_metadata.is_training:
            return self.forward_training(
                layer=layer,
                query=query,
                key=key,
                value=value,
                kv_cache=kv_cache,
                attn_metadata=attn_metadata,
                output=output,
            )

        # Cache the input KVs.
        key_cache, value_cache = kv_cache.unbind(0)
        if self.kv_sharing_target_layer_name is None:
            # Reshape the input keys and values and store them in the cache.
            # Skip this if sharing KV cache with an earlier attention layer.
            # NOTE(woosuk): Here, key and value are padded while slot_mapping is
            # not padded. However, we don't need to do key[:num_actual_tokens]
            # and value[:num_actual_tokens] because the reshape_and_cache_flash
            # op uses the slot_mapping's shape to determine the number of
            # actual tokens.
            ops.reshape_and_cache_flash(
                key,
                value,
                key_cache,
                value_cache,
                attn_metadata.slot_mapping,
                self.kv_cache_dtype,
                layer._k_scale,
                layer._v_scale,
            )

        num_actual_tokens = attn_metadata.num_actual_tokens
        num_decode_tokens = attn_metadata.num_decode_tokens
        if prefill_meta := attn_metadata.prefill_metadata:
            descale_shape = (prefill_meta.query_start_loc.shape[0] - 1,
                             key.shape[1])
            unified_attention(
                q=query[num_decode_tokens:num_actual_tokens],
                k=key_cache,
                v=value_cache,
                out=output[num_decode_tokens:num_actual_tokens],
                cu_seqlens_q=prefill_meta.query_start_loc,
                max_seqlen_q=prefill_meta.max_query_len,
                seqused_k=prefill_meta.seq_lens,
                max_seqlen_k=prefill_meta.max_seq_len,
                softmax_scale=self.scale,
                causal=True,
                alibi_slopes=self.alibi_slopes,
                window_size=self.sliding_window,
                block_table=prefill_meta.block_table,
                softcap=self.logits_soft_cap,
                q_descale=None,  # Not supported
                k_descale=layer._k_scale.expand(descale_shape),
                v_descale=layer._v_scale.expand(descale_shape),
            )

        if decode_meta := attn_metadata.decode_metadata:
            # Query for decode. KV is not needed because it is already cached.
            decode_query = query[:num_decode_tokens]
            # Reshape query to [1, B_T, G, H, D].
            q = decode_query.view(1, -1, self.num_kv_heads,
                                  self.num_queries_per_kv, self.head_size)
            # Reshape the k and v caches to [1, Bkv_T, G, H, D]
            cache_k = key_cache.view(1, -1, self.num_kv_heads, 1,
                                     self.head_size).expand(
                                         1,
                                         -1,
                                         self.num_kv_heads,
                                         self.num_queries_per_kv,
                                         self.head_size,
                                     )
            cache_v = value_cache.view(1, -1, self.num_kv_heads, 1,
                                       self.head_size).expand(
                                           1,
                                           -1,
                                           self.num_kv_heads,
                                           self.num_queries_per_kv,
                                           self.head_size,
                                       )

            attn_bias = decode_meta.attn_bias
            output[:
                   num_decode_tokens] = xops.memory_efficient_attention_forward(
                       q,
                       cache_k,
                       cache_v,
                       attn_bias=attn_bias,
                       p=0.0,
                       scale=self.scale,
                   ).view(decode_query.shape)

        # Reshape the output tensor.
        return output

    def forward_training_(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: XFormersAttentionMetadata,
        output: torch.Tensor,
    ) -> torch.Tensor:
        """Training-specific forward pass that bypasses KV cache.
        
        This method directly calls xformers memory_efficient_attention_forward
        on the Q, K, V tensors without using the KV cache, allowing gradients
        to flow through for training.

        Args:
            query: shape = [num_tokens, num_heads, head_size]
            key: shape = [num_tokens, num_kv_heads, head_size]
            value: shape = [num_tokens, num_kv_heads, head_size]
            kv_cache: shape = [2, num_blocks, block_size, num_kv_heads, head_size]
                      (not used in training, but kept for API compatibility)
            attn_metadata: Metadata for attention.
            output: shape = [num_tokens, num_heads, head_size]
        Returns:
            shape = [num_tokens, num_heads * head_size]
        """
        num_tokens = query.shape[0]

        # For training, we process all tokens together
        # Reshape tensors for xformers BMHK format: [batch, seqlen, num_heads, head_size]
        # xformers backward doesn't support BMGHK format, so we expand K/V to num_heads
        
        # Query: [num_tokens, num_heads, head_size] -> [1, num_tokens, num_heads, head_size]
        q = query.unsqueeze(0)
        
        # Key/Value: [num_tokens, num_kv_heads, head_size] -> [1, num_tokens, num_heads, head_size]
        if self.num_kv_heads != self.num_heads:
            # GQA: Expand K and V by repeating each kv_head num_queries_per_kv times
            k = key.unsqueeze(0).unsqueeze(3).expand(
                1, num_tokens, self.num_kv_heads, self.num_queries_per_kv, self.head_size
            ).reshape(1, num_tokens, self.num_heads, self.head_size)
            v = value.unsqueeze(0).unsqueeze(3).expand(
                1, num_tokens, self.num_kv_heads, self.num_queries_per_kv, self.head_size
            ).reshape(1, num_tokens, self.num_heads, self.head_size)
        else:
            # MHA: just add batch dimension
            k = key.unsqueeze(0)
            v = value.unsqueeze(0)

        # Create causal mask for training
        # For training with batched sequences, use BlockDiagonalCausalMask to prevent cross-sample attention
        if hasattr(attn_metadata, 'query_start_loc') and hasattr(attn_metadata, 'seq_lens'):
            query_start_loc = attn_metadata.query_start_loc
            if query_start_loc is not None and len(query_start_loc) > 1:
                # Multiple samples in batch - create block diagonal mask
                seqlens = torch.diff(query_start_loc).tolist()
                from xformers.ops.fmha.attn_bias import BlockDiagonalCausalMask
                attn_bias = BlockDiagonalCausalMask.from_seqlens(seqlens)
            else:
                # Single sample - use simple causal mask
                attn_bias = LowerTriangularMask()
        else:
            # Fallback to simple causal mask if metadata unavailable
            attn_bias = LowerTriangularMask()

        # Use xformers memory_efficient_attention (NOT _forward) for training
        # This function supports both forward and backward passes
        attn_output = xops.memory_efficient_attention(
            q,
            k,
            v,
            attn_bias=attn_bias,  # Use proper mask (block diagonal or causal)
            p=0.0,  # No dropout
            scale=self.scale,
        )

        # Reshape output from [1, num_tokens, num_heads, head_size]
        # back to [num_tokens, num_heads, head_size]
        output[:] = attn_output.squeeze(0)

        return output

    def forward_training__(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: XFormersAttentionMetadata,
        output: torch.Tensor,
    ) -> torch.Tensor:
        """Training-specific forward pass using PyTorch SDPA (equivalent to Transformers).
        
        This method uses torch.nn.functional.scaled_dot_product_attention instead of
        xformers, making it equivalent to the Transformers SDPA implementation.

        Args:
            query: shape = [num_tokens, num_heads, head_size]
            key: shape = [num_tokens, num_kv_heads, head_size]
            value: shape = [num_tokens, num_kv_heads, head_size]
            kv_cache: shape = [2, num_blocks, block_size, num_kv_heads, head_size]
                      (not used in training, but kept for API compatibility)
            attn_metadata: Metadata for attention.
            output: shape = [num_tokens, num_heads, head_size]
        Returns:
            shape = [num_tokens, num_heads * head_size]
        """
        global ATTN_MASK

        num_tokens = query.shape[0]

        # Reshape tensors for SDPA BNHD format: [batch, num_heads, seqlen, head_size]
        # Query: [num_tokens, num_heads, head_size] -> [1, num_heads, num_tokens, head_size]
        q = query.unsqueeze(0).transpose(1, 2)
        
        # Key/Value: [num_tokens, num_kv_heads, head_size] -> [1, num_kv_heads, num_tokens, head_size]
        # Then expand to num_heads for GQA
        if self.num_kv_heads != self.num_heads:
            # GQA: Expand K and V by repeating each kv_head num_queries_per_kv times
            k = key.unsqueeze(0).transpose(1, 2)  # [1, num_kv_heads, num_tokens, head_size]
            v = value.unsqueeze(0).transpose(1, 2)
            
            # Expand: [1, num_kv_heads, num_tokens, head_size] -> [1, num_heads, num_tokens, head_size]
            k = k.unsqueeze(2).expand(
                1, self.num_kv_heads, self.num_queries_per_kv, num_tokens, self.head_size
            ).reshape(1, self.num_heads, num_tokens, self.head_size)
            v = v.unsqueeze(2).expand(
                1, self.num_kv_heads, self.num_queries_per_kv, num_tokens, self.head_size
            ).reshape(1, self.num_heads, num_tokens, self.head_size)
        else:
            # MHA: just add batch dimension and transpose
            k = key.unsqueeze(0).transpose(1, 2)
            v = value.unsqueeze(0).transpose(1, 2)

        # print(q.shape, k.shape, v.shape)

        # Flatten tensors to match Transformers format for comparison
        # SDPA: [batch, num_heads, seq_len, head_dim] -> [batch*seq_len, num_heads, head_dim]
        batch, num_heads, seq_len, head_dim = q.shape
        
        # Reshape: [batch, num_heads, seq_len, head_dim] -> [batch, seq_len, num_heads, head_dim]
        query_reordered = q.transpose(1, 2).contiguous()
        key_reordered = k.transpose(1, 2).contiguous()
        value_reordered = v.transpose(1, 2).contiguous()
        
        # Flatten batch and seq_len: [batch, seq_len, num_heads, head_dim] -> [batch*seq_len, num_heads, head_dim]
        query_flat = query_reordered.reshape(-1, num_heads, head_dim)
        key_flat = key_reordered.reshape(-1, num_heads, head_dim)
        value_flat = value_reordered.reshape(-1, num_heads, head_dim)

        # # save query, key, value to separate csv files
        # import pandas as pd
        # df = pd.DataFrame({
        #     "query": query_flat.flatten().tolist(),
        # })
        # df.to_csv(f"vllm_query.csv", index=False)
        # df = pd.DataFrame({
        #     "key": key_flat.flatten().tolist(),
        # })
        # df.to_csv(f"vllm_key.csv", index=False)
        # df = pd.DataFrame({
        #     "value": value_flat.flatten().tolist(),
        # })
        # df.to_csv(f"vllm_value.csv", index=False)
        # ss
        # print(f"Flattened shapes: q={query_flat.shape}, k={key_flat.shape}, v={value_flat.shape}")

        # Create attention mask for causal attention
        # For training with batched sequences, we need a block diagonal causal mask
        # attention_mask = None
        is_causal = False
        
        if hasattr(attn_metadata, 'query_start_loc') and hasattr(attn_metadata, 'seq_lens'):
            query_start_loc = attn_metadata.query_start_loc
            if query_start_loc is not None and len(query_start_loc) > 1:
                # Multiple samples in batch - create block diagonal mask
                # For SDPA, we need to create a proper attention mask
                # Shape: [batch, num_heads, seq_len, seq_len] or [batch, 1, seq_len, seq_len]
                seqlens = torch.diff(query_start_loc).tolist()
                
                # Create a block diagonal causal mask
                # For simplicity with SDPA, we'll use is_causal=True which handles causal masking
                # but we need to ensure no cross-sample attention
                # This is a limitation - SDPA doesn't have native block diagonal support
                # For now, we'll just use is_causal=True (which works for single sample)
                pass

        dropout = 0.0
        scaling = self.scale

        if ATTN_MASK is None:
            
            # Load attention_mask from PEFT: shape [4, 1, 512, 512], dtype bool
            # Convert to vLLM format: [1, 1, 2048, 2048] (block diagonal)
            import pandas as pd
            df = pd.read_csv("transformers_attn_mask.csv")
            batch_vllm, num_heads, seq_len_vllm, head_dim = q.shape  # [1, 32, 2048, 64]
            
            # PEFT has [4, 1, 512, 512], we need [1, 1, 2048, 2048]
            peft_batch = 4
            peft_seq_len = 512
            peft_mask = torch.tensor(df["attn_mask"].values, dtype=torch.bool, device=q.device).reshape(peft_batch, 1, peft_seq_len, peft_seq_len)
            
            # Create block diagonal mask: [1, 1, 2048, 2048]
            ATTN_MASK = torch.zeros(1, 1, seq_len_vllm, seq_len_vllm, dtype=torch.bool, device=q.device)
            for i in range(peft_batch):
                start_idx = i * peft_seq_len
                end_idx = (i + 1) * peft_seq_len
                ATTN_MASK[0, 0, start_idx:end_idx, start_idx:end_idx] = peft_mask[i, 0]
            
            print(f"vLLM loaded ATTN_MASK shape: {ATTN_MASK.shape}, dtype: {ATTN_MASK.dtype}")

        attn_mask = ATTN_MASK

        # # Construct block diagonal attention mask from per-request masks
        # attn_mask = None
        # if attn_metadata.training_attention_mask is not None:
        #     mask_data = attn_metadata.training_attention_mask
        #     if isinstance(mask_data, dict) and 'masks' in mask_data:
        #         # Extract individual masks and sequence lengths
        #         masks_per_req = mask_data['masks']  # List of [seq_len_i, seq_len_i] tensors
        #         seq_lens = mask_data['seq_lens']    # List of seq_len_i integers

        #         # Total sequence length (sum of all requests)
        #         total_seq_len = sum(seq_lens)
        #         batch_vllm, num_heads_vllm, _, head_dim = q.shape  # [1, num_heads, total_seq_len, head_dim]
                
        #         # Create block diagonal mask: [1, 1, total_seq_len, total_seq_len]
        #         attn_mask = torch.zeros(1, 1, total_seq_len, total_seq_len, 
        #                                dtype=torch.bool, device=q.device)
                
        #         # Fill in blocks for each request
        #         current_offset = 0
        #         for req_mask, seq_len in zip(masks_per_req, seq_lens):
        #             # req_mask shape: [seq_len, seq_len]
        #             end_offset = current_offset + seq_len
        #             attn_mask[0, 0, current_offset:end_offset, current_offset:end_offset] = req_mask
        #             current_offset = end_offset
                
        #         # # Save mask to CSV for comparison
        #         # import pandas as pd
        #         # df = pd.DataFrame({
        #         #     "attn_mask": attn_mask.flatten().cpu().tolist(),
        #         # })
        #         # df.to_csv("vllm_xformers_attn_mask.csv", index=False)
        #         # print(f"vLLM xformers saved attn_mask shape: {attn_mask.shape}, dtype: {attn_mask.dtype}")
        #         # ss


        # Use PyTorch's scaled_dot_product_attention (same as Transformers)
        attn_output = torch.nn.functional.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=dropout,
            scale=scaling,
            is_causal=is_causal,
        )

        # print(dropout, scaling, is_causal)

        # import pandas as pd
        # df = pd.DataFrame({
        #     "q": query_flat.flatten().tolist(),
        # })
        # df.to_csv(f"vllm_sdpa_q.csv", index=False)
        # df = pd.DataFrame({
        #     "k": key_flat.flatten().tolist(),
        # })
        # df.to_csv(f"vllm_sdpa_k.csv", index=False)
        # df = pd.DataFrame({
        #     "v": value_flat.flatten().tolist(),
        # })
        # df.to_csv(f"vllm_sdpa_v.csv", index=False)

        # import pandas as pd
        # df = pd.DataFrame({
        #     "attn_output": attn_output.flatten().tolist(),
        # })
        # df.to_csv(f"vllm_attn_output.csv", index=False)
        # ss

        # Reshape output from [1, num_heads, num_tokens, head_size]
        # to [num_tokens, num_heads, head_size]
        # First transpose to [1, num_tokens, num_heads, head_size]
        attn_output = attn_output.transpose(1, 2).contiguous()
        # Then squeeze batch dimension
        output[:] = attn_output.squeeze(0)

        return output

    def forward_training___(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: XFormersAttentionMetadata,
        output: torch.Tensor,
    ) -> torch.Tensor:
        """Training-specific forward pass using PyTorch SDPA (equivalent to Transformers).
        
        Args:
            query: shape = [num_tokens, num_heads, head_size]
            key: shape = [num_tokens, num_kv_heads, head_size]
            value: shape = [num_tokens, num_kv_heads, head_size]
        Returns:
            shape = [num_tokens, num_heads * head_size]
        """
        global ATTN_MASK
        
        # Extract batch structure from metadata
        query_start_loc = attn_metadata.query_start_loc
        if query_start_loc is None or len(query_start_loc) <= 1:
            raise ValueError("Training requires proper batch metadata with query_start_loc")
        
        # Get sequence lengths for each sample in the batch
        seqlens = torch.diff(query_start_loc).tolist()
        batch_size = len(seqlens)
        max_seq_len = max(seqlens)
        
        # Verify all sequences have the same length (simplification for now)
        if len(set(seqlens)) != 1:
            raise NotImplementedError("Variable sequence lengths not yet supported")
        seq_len = seqlens[0]
        
        # Reshape Q from [total_tokens, num_heads, head_dim] to [batch, num_heads, seq_len, head_dim]
        # total_tokens = batch * seq_len
        q = query.view(batch_size, seq_len, self.num_heads, self.head_size).transpose(1, 2).contiguous()
        
        # Reshape K/V from [total_tokens, num_kv_heads, head_dim] to [batch, num_kv_heads, seq_len, head_dim]
        k = key.view(batch_size, seq_len, self.num_kv_heads, self.head_size).transpose(1, 2).contiguous()
        v = value.view(batch_size, seq_len, self.num_kv_heads, self.head_size).transpose(1, 2).contiguous()
        
        # Expand K/V to match Q using repeat_kv logic (same as Transformers)
        if self.num_kv_heads != self.num_heads:
            # GQA: Expand K and V by repeating each kv_head num_queries_per_kv times
            # From [batch, num_kv_heads, seq_len, head_dim] to [batch, num_heads, seq_len, head_dim]
            k = k[:, :, None, :, :].expand(
                batch_size, self.num_kv_heads, self.num_queries_per_kv, seq_len, self.head_size
            ).reshape(batch_size, self.num_heads, seq_len, self.head_size).contiguous()
            
            v = v[:, :, None, :, :].expand(
                batch_size, self.num_kv_heads, self.num_queries_per_kv, seq_len, self.head_size
            ).reshape(batch_size, self.num_heads, seq_len, self.head_size).contiguous()

        # Handle attention mask
        attn_mask = None
        is_causal = False
        dropout = 0.0
        scaling = self.scale
        
        if ATTN_MASK is None and batch_size > 1:
            # Load or construct attention mask if needed
            # For block diagonal causal mask matching multiple sequences
            import pandas as pd
            df = pd.read_csv("transformers_attn_mask.csv")
            
            # Reconstruct block diagonal mask
            total_seq_len = batch_size * seq_len
            peft_mask = torch.tensor(df["attn_mask"].values, dtype=torch.bool, device=q.device).reshape(
                batch_size, 1, seq_len, seq_len
            )
            
            # Create block diagonal mask: [batch, 1, seq_len, seq_len] or [1, 1, total_seq_len, total_seq_len]
            # SDPA can handle per-sample masks with shape [batch, 1, seq_len, seq_len]
            ATTN_MASK = peft_mask
            print(f"vLLM loaded ATTN_MASK shape: {ATTN_MASK.shape}, dtype: {ATTN_MASK.dtype}")
        
        attn_mask = ATTN_MASK
        
        # DEBUG: Save Q, K, V before SDPA for comparison
        import pandas as pd
        pd.DataFrame({"q": q.detach().cpu().float().flatten().numpy()}).to_csv("vllm_q.csv", index=False)
        pd.DataFrame({"k": k.detach().cpu().float().flatten().numpy()}).to_csv("vllm_k.csv", index=False)
        pd.DataFrame({"v": v.detach().cpu().float().flatten().numpy()}).to_csv("vllm_v.csv", index=False)
        if attn_mask is not None:
            pd.DataFrame({"mask": attn_mask.detach().cpu().flatten().numpy()}).to_csv("vllm_mask.csv", index=False)
        print(f"[vLLM] Saved Q: {q.shape}, K: {k.shape}, V: {v.shape}, is_causal: {is_causal}, dropout: {dropout}, scale: {scaling}")
        
        # Use PyTorch's scaled_dot_product_attention (same as Transformers)
        attn_output = torch.nn.functional.scaled_dot_product_attention(
            q,  # [batch, num_heads, seq_len, head_dim]
            k,  # [batch, num_heads, seq_len, head_dim]
            v,  # [batch, num_heads, seq_len, head_dim]
            attn_mask=attn_mask,
            dropout_p=dropout,
            scale=scaling,
            is_causal=is_causal,
        )
        
        # Reshape output from [batch, num_heads, seq_len, head_dim]
        # back to [total_tokens, num_heads, head_dim]
        attn_output_transposed = attn_output.transpose(1, 2).contiguous()
        
        # DEBUG: Save attention output for comparison (before final reshape)
        pd.DataFrame({"output": attn_output_transposed.detach().cpu().float().flatten().numpy()}).to_csv("vllm_output.csv", index=False)
        print(f"[vLLM] Saved attn_output: {attn_output_transposed.shape}")
        ss

        attn_output_flat = attn_output_transposed.view(-1, self.num_heads, self.head_size)
        output[:] = attn_output_flat
        
        return output

    def forward_training(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: XFormersAttentionMetadata,
        output: torch.Tensor,
    ) -> torch.Tensor:
        """Training-specific forward pass using PyTorch SDPA (equivalent to Transformers).

        This method matches the exact reshaping and computation done in Transformers' SDPA.

        Args:
            query: shape = [num_tokens, num_heads, head_size]
            key: shape = [num_tokens, num_kv_heads, head_size]
            value: shape = [num_tokens, num_kv_heads, head_size]
        Returns:
            shape = [num_tokens, num_heads * head_size]
        """
        global ATTN_MASK

        # Extract batch structure from metadata
        query_start_loc = attn_metadata.query_start_loc
        if query_start_loc is None or len(query_start_loc) <= 1:
            raise ValueError("Training requires proper batch metadata with query_start_loc")

        # Get sequence lengths for each sample in the batch
        seqlens = torch.diff(query_start_loc).tolist()
        batch_size = len(seqlens)

        # Verify all sequences have the same length (simplification for now)
        if len(set(seqlens)) != 1:
            raise NotImplementedError("Variable sequence lengths not yet supported")
        seq_len = seqlens[0]

        # Reshape Q from [total_tokens, num_heads, head_dim] to [batch, num_heads, seq_len, head_dim]
        # This matches Transformers format: [batch, num_heads, seq_len, head_dim]
        q = query.view(batch_size, seq_len, self.num_heads, self.head_size).transpose(1, 2).contiguous()

        # Reshape K/V from [total_tokens, num_kv_heads, head_dim] to [batch, num_kv_heads, seq_len, head_dim]
        k = key.view(batch_size, seq_len, self.num_kv_heads, self.head_size).transpose(1, 2).contiguous()
        v = value.view(batch_size, seq_len, self.num_kv_heads, self.head_size).transpose(1, 2).contiguous()

        # # save query_states, key_states, value_states to a csv file
        # import pandas as pd
        # df = pd.DataFrame({
        #     "q": q.flatten().tolist(),
        # })
        # df.to_csv("vllm_q_reshaped.csv", index=False)
        # df = pd.DataFrame({
        #     "k": k.flatten().tolist(),
        # })
        # df.to_csv("vllm_k_reshaped.csv", index=False)
        # df = pd.DataFrame({
        #     "v": v.flatten().tolist(),
        # })
        # df.to_csv("vllm_v_reshaped.csv", index=False)
        # ss

        # Apply repeat_kv expansion (same as Transformers sdpa_attention.py lines 62-63)
        # This matches the logic in repeat_kv function
        if self.num_kv_heads != self.num_heads:
            # GQA: Expand K and V by repeating each kv_head num_queries_per_kv times
            # From [batch, num_kv_heads, seq_len, head_dim] to [batch, num_heads, seq_len, head_dim]
            # This is equivalent to repeat_kv in Transformers
            k = k[:, :, None, :, :].expand(
                batch_size, self.num_kv_heads, self.num_queries_per_kv, seq_len, self.head_size
            ).reshape(batch_size, self.num_heads, seq_len, self.head_size).contiguous()

            v = v[:, :, None, :, :].expand(
                batch_size, self.num_kv_heads, self.num_queries_per_kv, seq_len, self.head_size
            ).reshape(batch_size, self.num_heads, seq_len, self.head_size).contiguous()

        # Now q, k, v all have shape [batch, num_heads, seq_len, head_dim]
        # This matches Transformers format exactly: [4, 32, 512, 64]

        # Handle attention mask - reshape to match PEFT format [batch, 1, seq_len, seq_len]
        dropout = 0.0
        scaling = self.scale
        is_causal = False
        attn_mask = None

        if attn_metadata.training_attention_mask is not None:
            mask_data = attn_metadata.training_attention_mask
            if isinstance(mask_data, dict) and 'masks' in mask_data:
                # Extract individual masks and sequence lengths
                masks_per_req = mask_data['masks']  # List of [seq_len_i, seq_len_i] tensors
                seq_lens_list = mask_data['seq_lens']    # List of seq_len_i integers

                # Verify we have the right number of masks
                if len(masks_per_req) != batch_size:
                    raise ValueError(f"Number of masks ({len(masks_per_req)}) doesn't match batch_size ({batch_size})")

                # Stack masks to create batched format: [batch, 1, seq_len, seq_len]
                # Each mask is [seq_len_i, seq_len_i], we add head dim and batch them
                batched_masks = []
                for i, mask in enumerate(masks_per_req):
                    # Verify mask shape matches expected seq_len
                    if mask.shape[0] != seq_len or mask.shape[1] != seq_len:
                        raise ValueError(
                            f"Mask {i} shape {mask.shape} doesn't match expected [{seq_len}, {seq_len}]. "
                            f"seq_lens_list: {seq_lens_list}, batch_size: {batch_size}"
                        )
                    # Add head dimension: [seq_len, seq_len] -> [1, seq_len, seq_len]
                    mask_with_head = mask.unsqueeze(0)
                    batched_masks.append(mask_with_head)

                # Stack along batch dimension: [batch, 1, seq_len, seq_len]
                attn_mask = torch.stack(batched_masks, dim=0)

                # Verify shape matches PEFT format
                if attn_mask.shape != (batch_size, 1, seq_len, seq_len):
                    raise ValueError(
                        f"Attention mask shape {attn_mask.shape} doesn't match expected [{batch_size}, 1, {seq_len}, {seq_len}]. "
                        f"Individual mask shapes: {[m.shape for m in masks_per_req]}, seq_lens_list: {seq_lens_list}"
                    )

        # # Debug: save attention mask
        # import pandas as pd
        # df = pd.DataFrame({
        #     "attn_mask": attn_mask.flatten().tolist(),
        # })
        # df.to_csv("vllm_attn_mask.csv", index=False)
        # ss

        # import pandas as pd
        # # save q, k, v to separate csv files
        # df = pd.DataFrame({
        #     "q": q.flatten().tolist(),
        # })
        # df.to_csv("vllm_q_in_attn.csv", index=False)
        # df = pd.DataFrame({
        #     "k": k.flatten().tolist(),
        # })
        # df.to_csv("vllm_k_in_attn.csv", index=False)
        # df = pd.DataFrame({
        #     "v": v.flatten().tolist(),
        # })
        # df.to_csv("vllm_v_in_attn.csv", index=False)
        # ss

        # Call SDPA with same parameters as Transformers (sdpa_attention.py line 118)
        attn_output = torch.nn.functional.scaled_dot_product_attention(
            q,  # [batch, num_heads, seq_len, head_dim] - same as Transformers
            k,  # [batch, num_heads, seq_len, head_dim] - same as Transformers
            v,  # [batch, num_heads, seq_len, head_dim] - same as Transformers
            attn_mask=attn_mask,
            dropout_p=dropout,
            scale=scaling,
            is_causal=is_causal,
        )
        # attn_output shape: [batch, num_heads, seq_len, head_dim] = [4, 32, 512, 64]

        # Reshape output to match Transformers' output format
        # Transformers does: attn_output.transpose(1, 2).contiguous() (line 128)
        # This gives: [batch, seq_len, num_heads, head_dim] = [4, 512, 32, 64]
        attn_output = attn_output.transpose(1, 2).contiguous()

        # import pandas as pd
        # df = pd.DataFrame({
        #     "attn_output": attn_output.flatten().tolist(),
        # })
        # df.to_csv("vllm_attn_output.csv", index=False)

        # Flatten back to vLLM format: [total_tokens, num_heads, head_dim]
        # [4, 512, 32, 64] -> [2048, 32, 64]
        attn_output_flat = attn_output.view(-1, self.num_heads, self.head_size)

        # import pandas as pd
        # df = pd.DataFrame({
        #     "attn_output": attn_output_flat.flatten().tolist(),
        # })
        # df.to_csv("vllm_attn_output_flat.csv", index=False)

        output[:] = attn_output_flat

        return attn_output_flat

        # return output
