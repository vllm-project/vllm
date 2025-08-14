# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Attention layer with AiterFlashAttention."""
from dataclasses import dataclass
from typing import ClassVar, Optional

import torch

from vllm.attention.backends.abstract import (AttentionBackend, AttentionImpl,
                                              AttentionMetadata, AttentionType)
from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.v1.attention.backends.utils import (AttentionMetadataBuilder,
                                              CommonAttentionMetadata)
from vllm.v1.kv_cache_interface import AttentionSpec

_PARTITION_SIZE_ROCM = 256

if current_platform.is_rocm():
    import aiter

    from vllm.triton_utils import tl, triton
    from vllm.utils import direct_register_custom_op

    @triton.jit
    def _vllm_layout_trans_kernel(
        k_buffer_ptr,
        v_buffer_ptr,
        k_values_ptr,
        v_values_ptr,
        b_query_lens_loc,
        b_seq_lens_loc,
        block_table,
        block_table_stride_0,
        k_scale,
        v_scale,
        output_dtype: tl.constexpr,
        E_DIM: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        batch_idx = tl.program_id(0)
        block_idx = tl.program_id(1)

        batch_query_indexes = tl.load(b_query_lens_loc + batch_idx +
                                      tl.arange(0, 2))
        batch_query_start, batch_query_end = tl.split(batch_query_indexes)
        query_len = batch_query_end - batch_query_start

        if query_len <= 1:
            return

        batch_token_indexes = tl.load(b_seq_lens_loc + batch_idx +
                                      tl.arange(0, 2))
        batch_token_start, batch_token_end = tl.split(batch_token_indexes)
        seq_len = batch_token_end - batch_token_start

        if block_idx * BLOCK_SIZE < seq_len:
            block_mask = (block_idx * BLOCK_SIZE +
                          tl.arange(0, BLOCK_SIZE)[:, None]) < seq_len

            kv_idx = tl.load(block_table + batch_idx * block_table_stride_0 +
                             block_idx).to(tl.int64)

            kv_buffer_off = kv_idx * BLOCK_SIZE * E_DIM + tl.arange(
                0, BLOCK_SIZE)[:, None] * E_DIM + tl.arange(0, E_DIM)[None, :]
            k_vals = tl.load(k_buffer_ptr + kv_buffer_off,
                             mask=block_mask,
                             other=0.0)
            if k_vals.dtype.is_fp8():
                k_vals = (k_vals.to(tl.float32) *
                          tl.load(k_scale)).to(output_dtype)
            else:
                k_vals = k_vals.to(output_dtype)

            v_vals = tl.load(v_buffer_ptr + kv_buffer_off,
                             mask=block_mask,
                             other=0.0)
            if v_vals.dtype.is_fp8():
                v_vals = (v_vals.to(tl.float32) *
                          tl.load(v_scale)).to(output_dtype)
            else:
                v_vals = v_vals.to(output_dtype)
            kv_values_off = batch_token_start * E_DIM + \
                block_idx * BLOCK_SIZE * E_DIM + \
                tl.arange(0, BLOCK_SIZE)[:, None] * E_DIM + \
                tl.arange(0, E_DIM)[None, :]
            tl.store(k_values_ptr + kv_values_off, k_vals, mask=block_mask)
            tl.store(v_values_ptr + kv_values_off, v_vals, mask=block_mask)

    def vllm_layout_trans(b_query_lens_loc, b_seq_lens_loc, block_table,
                          k_cache, v_cache, max_seq_len, k_scale, v_scale,
                          output_dtype, total_tokens):
        H_KV = v_cache.shape[2]
        D = v_cache.shape[3]
        BLOCK_SIZE = v_cache.shape[1]

        k_values = torch.empty(
            (total_tokens, H_KV, D),
            dtype=output_dtype,
            device=k_cache.device,
        )
        v_values = torch.empty(
            (total_tokens, H_KV, D),
            dtype=output_dtype,
            device=v_cache.device,
        )

        grid = (block_table.shape[0],
                (max_seq_len + BLOCK_SIZE - 1) // BLOCK_SIZE)

        if output_dtype == torch.float16:
            output_dtype = tl.float16
        elif output_dtype == torch.bfloat16:
            output_dtype = tl.bfloat16
        else:
            raise ValueError(f"Unsupported output dtype: {output_dtype}")

        _vllm_layout_trans_kernel[grid](k_cache,
                                        v_cache,
                                        k_values,
                                        v_values,
                                        b_query_lens_loc,
                                        b_seq_lens_loc,
                                        block_table,
                                        block_table.stride(0),
                                        k_scale,
                                        v_scale,
                                        output_dtype=output_dtype,
                                        E_DIM=H_KV * D,
                                        BLOCK_SIZE=BLOCK_SIZE)

        return k_values, v_values

    def flash_attn_varlen_func_impl(
        q: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        out: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        cu_seqlens_k: torch.Tensor,
        max_seqlen_q: int,
        max_seqlen_k: int,
        softmax_scale: float,
        window_size: Optional[list[int]],  # -1 means infinite context window
        alibi_slopes: Optional[list[float]],
        block_table: torch.Tensor,
        k_scale: torch.Tensor,
        v_scale: torch.Tensor,
        total_tokens: int = 0,
    ) -> torch.Tensor:
        if total_tokens == 0:
            total_tokens = int(cu_seqlens_k[-1].item())
        k, v = vllm_layout_trans(cu_seqlens_q, cu_seqlens_k, block_table,
                                 k_cache, v_cache, max_seqlen_k, k_scale,
                                 v_scale, q.dtype, total_tokens)

        output = aiter.flash_attn_varlen_func(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=cu_seqlens_q,
            max_seqlen_q=max_seqlen_q,
            min_seqlen_q=1,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_k=max_seqlen_k,
            softmax_scale=softmax_scale,
            causal=True,
            alibi_slopes=alibi_slopes,
            window_size=window_size,
            out=out,
        )
        return output

    def flash_attn_varlen_func_fake(
        q: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        out: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        cu_seqlens_k: torch.Tensor,
        max_seqlen_q: int,
        max_seqlen_k: int,
        softmax_scale: float,
        window_size: Optional[list[int]],  # -1 means infinite context window
        alibi_slopes: Optional[list[float]],
        block_table: torch.Tensor,
        k_scale: torch.Tensor,
        v_scale: torch.Tensor,
        total_tokens: int = 0,
    ) -> torch.Tensor:
        return torch.empty(q.shape[0],
                           q.shape[1],
                           v_cache.shape[-2],
                           dtype=q.dtype,
                           device=q.device)

    direct_register_custom_op("flash_attn_varlen_func",
                              flash_attn_varlen_func_impl, ["out"],
                              flash_attn_varlen_func_fake,
                              dispatch_key=current_platform.dispatch_key)

logger = init_logger(__name__)


@dataclass
class AiterFlashAttentionMetadata:
    # NOTE(sang): Definition of context_len, query_len, and seq_len.
    # |---------- N-1 iteration --------|
    # |---------------- N iteration ---------------------|
    # |- tokenA -|......................|-- newTokens ---|
    # |---------- context_len ----------|
    # |-------------------- seq_len ---------------------|
    #                                   |-- query_len ---|

    num_actual_tokens: int  # Number of tokens excluding padding.
    num_actual_kv_tokens: int
    max_query_len: int
    query_start_loc: torch.Tensor
    max_seq_len: int
    seq_lens: torch.Tensor
    slot_mapping: torch.Tensor
    block_table: torch.Tensor
    cu_seq_lens: Optional[torch.Tensor]

    # For cascade attention.
    use_cascade: bool
    common_prefix_len: int
    total_tokens: int


class AiterFlashAttentionMetadataBuilder(
        AttentionMetadataBuilder[AiterFlashAttentionMetadata]):
    full_cudagraph_supported: ClassVar[bool] = True

    def __init__(self, kv_cache_spec: AttentionSpec, layer_names: list[str],
                 vllm_config: VllmConfig, device: torch.device):
        self.vllm_config = vllm_config
        self.model_config = vllm_config.model_config
        self.parallel_config = vllm_config.parallel_config
        self.cache_config = vllm_config.cache_config
        self.device = device

        self.num_heads_q = self.model_config.get_num_attention_heads(
            self.parallel_config)
        self.num_heads_kv = self.model_config.get_num_kv_heads(
            self.parallel_config)
        self.headdim = self.model_config.get_head_size()
        self.block_size = kv_cache_spec.block_size
        self.kv_cache_spec = kv_cache_spec
        # Sliding window size to be used with the AOT scheduler will be
        # populated on first build() call.
        self.aot_sliding_window: Optional[tuple[int, int]] = None
        self.total_tokens: int = 0

    def build_for_cudagraph_capture(
            self, common_attn_metadata: CommonAttentionMetadata):
        self.total_tokens = self.model_config.max_model_len \
            * self.vllm_config.scheduler_config.max_num_partial_prefills
        res = self.build(common_prefix_len=0,
                         common_attn_metadata=common_attn_metadata)
        self.total_tokens = 0
        return res

    def build(self,
              common_prefix_len: int,
              common_attn_metadata: CommonAttentionMetadata,
              fast_build: bool = False) -> 'AiterFlashAttentionMetadata':

        num_actual_tokens = common_attn_metadata.num_actual_tokens
        max_query_len = common_attn_metadata.max_query_len
        max_seq_len = int(common_attn_metadata.seq_lens_cpu.max())
        query_start_loc = common_attn_metadata.query_start_loc
        seq_lens = common_attn_metadata.seq_lens
        block_table_tensor = common_attn_metadata.block_table_tensor
        slot_mapping = common_attn_metadata.slot_mapping
        if max_query_len > 1:
            # We pre-compute cumulative seq len needed for prefill attention
            # here to avoid recomputing it for every layer
            cu_seq_lens = torch.zeros(seq_lens.shape[0] + 1,
                                      dtype=torch.int32,
                                      device=seq_lens.device)
            torch.cumsum(seq_lens,
                         dim=0,
                         dtype=cu_seq_lens.dtype,
                         out=cu_seq_lens[1:])
            num_actual_kv_tokens = int(cu_seq_lens[-1].item())
        else:
            cu_seq_lens = None
            num_actual_kv_tokens = 0

        def schedule(batch_size, cu_query_lens, max_query_len, seqlens,
                     max_seq_len, causal):
            return None

        use_cascade = common_prefix_len > 0

        attn_metadata = AiterFlashAttentionMetadata(
            num_actual_tokens=num_actual_tokens,
            num_actual_kv_tokens=num_actual_kv_tokens,
            max_query_len=max_query_len,
            query_start_loc=query_start_loc,
            max_seq_len=max_seq_len,
            seq_lens=seq_lens,
            block_table=block_table_tensor,
            slot_mapping=slot_mapping,
            cu_seq_lens=cu_seq_lens,
            use_cascade=use_cascade,
            common_prefix_len=common_prefix_len,
            total_tokens=self.total_tokens,
        )
        return attn_metadata

    def can_run_in_cudagraph(
            self, common_attn_metadata: CommonAttentionMetadata) -> bool:
        # Full CUDA Graph always supported (FA2 support checked separately)
        return True

    def use_cascade_attention(self, *args, **kwargs) -> bool:
        return False


class AiterFlashAttentionBackend(AttentionBackend):

    accept_output_buffer: bool = True

    @classmethod
    def get_supported_dtypes(cls) -> list[torch.dtype]:
        return [torch.float16, torch.bfloat16]

    @classmethod
    def get_supported_head_sizes(cls) -> list[int]:
        return [64, 128, 256]

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
        return "FLASH_ATTN_VLLM_V1"

    @staticmethod
    def get_impl_cls() -> type["AiterFlashAttentionImpl"]:
        return AiterFlashAttentionImpl

    @staticmethod
    def get_metadata_cls() -> type["AttentionMetadata"]:
        return AiterFlashAttentionMetadata

    @staticmethod
    def get_builder_cls() -> type["AiterFlashAttentionMetadataBuilder"]:
        return AiterFlashAttentionMetadataBuilder

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


class AiterFlashAttentionImpl(AttentionImpl):

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
        kv_sharing_target_layer_name: Optional[int] = None,
    ) -> None:
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = float(scale)
        self.num_kv_heads = num_kv_heads
        if alibi_slopes is not None:
            alibi_slopes = torch.tensor(alibi_slopes, dtype=torch.float32)
        self.alibi_slopes = alibi_slopes
        if sliding_window is None:
            self.sliding_window = [-1, -1]
        else:
            self.sliding_window = [sliding_window - 1, 0]
        self.kv_cache_dtype = kv_cache_dtype
        if logits_soft_cap is None:
            # In flash-attn, setting logits_soft_cap as 0 means no soft cap.
            logits_soft_cap = 0.
        self.logits_soft_cap = logits_soft_cap
        self.kv_sharing_target_layer_name = kv_sharing_target_layer_name

        assert self.num_heads % self.num_kv_heads == 0
        self.num_queries_per_kv = self.num_heads // self.num_kv_heads

        AiterFlashAttentionBackend.validate_head_size(head_size)

        if attn_type != AttentionType.DECODER:
            raise NotImplementedError("Encoder self-attention and "
                                      "encoder/decoder cross-attention "
                                      "are not implemented for "
                                      "FlashAttentionImpl")

    def forward(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: AiterFlashAttentionMetadata,
        output: Optional[torch.Tensor] = None,
        output_scale: Optional[torch.Tensor] = None,
        output_block_scale: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass with AiterFlashAttention.

        Args:
            query: shape = [num_tokens, num_heads, head_size]
            key: shape = [num_tokens, num_kv_heads, head_size]
            value: shape = [num_tokens, num_kv_heads, head_size]
            kv_cache = [2, num_blocks, block_size, num_kv_heads, head_size]
            attn_metadata: Metadata for attention.
        Returns:
            shape = [num_tokens, num_heads * head_size]
        NOTE: FP8 quantization, flash-attn expect the size of
              {q,k,v}_descale to be (num_sequences, num_kv_heads).
              We use torch's .expand() to avoid duplicating values
        """
        assert output is not None, "Output tensor must be provided."

        if output_scale is not None or output_block_scale is not None:
            raise NotImplementedError(
                "fused output quantization is not yet supported"
                " for FlashAttentionImpl")

        if attn_metadata is None:
            # Profiling run.
            return output

        # IMPORTANT!
        # NOTE(woosuk): With piece-wise CUDA graphs, this method is executed in
        # eager-mode PyTorch. Thus, we need to be careful about any CPU overhead
        # in this method. For example, `view` and `slice` (or `[:n]`) operations
        # are surprisingly slow even in the case they do not invoke any GPU ops.
        # Minimize the PyTorch ops in this method as much as possible.
        # Whenever making a change in this method, please benchmark the
        # performance to make sure it does not introduce any overhead.

        num_actual_tokens = attn_metadata.num_actual_tokens
        key_cache, value_cache = kv_cache.unbind(0)
        if self.kv_sharing_target_layer_name is None:
            # Reshape the input keys and values and store them in the cache.
            # Skip this if sharing KV cache with an earlier attention layer.
            # NOTE(woosuk): Here, key and value are padded while slot_mapping is
            # not padded. However, we don't need to do key[:num_actual_tokens]
            # and value[:num_actual_tokens] because the reshape_and_cache_flash
            # op uses the slot_mapping's shape to determine the number of
            # actual tokens.
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
            key_cache = key_cache.view(torch.float8_e4m3fnuz)
            value_cache = value_cache.view(torch.float8_e4m3fnuz)

        if not attn_metadata.use_cascade:
            cu_seqlens_q = attn_metadata.query_start_loc
            seqused_k = attn_metadata.seq_lens
            max_seqlen_q = attn_metadata.max_query_len
            max_seqlen_k = attn_metadata.max_seq_len
            block_table = attn_metadata.block_table

            if max_seqlen_q > 1:
                torch.ops.vllm.flash_attn_varlen_func(
                    query[:num_actual_tokens],
                    key_cache,
                    value_cache,
                    out=output[:num_actual_tokens],
                    cu_seqlens_q=cu_seqlens_q,
                    max_seqlen_q=max_seqlen_q,
                    max_seqlen_k=max_seqlen_k,
                    softmax_scale=self.scale,
                    alibi_slopes=self.alibi_slopes,
                    window_size=self.sliding_window,
                    block_table=block_table,
                    cu_seqlens_k=attn_metadata.cu_seq_lens,
                    k_scale=layer._k_scale,
                    v_scale=layer._v_scale,
                    total_tokens=attn_metadata.num_actual_kv_tokens,
                )

            _, num_heads, head_size = query.shape
            nbytes_per_qo_elem = torch.finfo(query.dtype).bits // 8
            num_seqs = seqused_k.shape[0]
            max_num_partitions = (max_seqlen_k + _PARTITION_SIZE_ROCM -
                                  1) // _PARTITION_SIZE_ROCM

            workspace_buffer = torch.empty(
                (num_seqs * num_heads * max_num_partitions * head_size) *
                nbytes_per_qo_elem + 2 *
                (num_seqs * num_heads * max_num_partitions) * 4,
                dtype=torch.uint8,
                device=output.device,
            )

            torch.ops.aiter.paged_attention_v1(
                output[:num_actual_tokens],
                workspace_buffer,
                query[:num_actual_tokens],
                key_cache,
                value_cache,
                self.scale,
                block_table,
                cu_seqlens_q,
                seqused_k,
                max_seqlen_k,
                self.alibi_slopes,
                self.kv_cache_dtype,
                "NHD",
                self.logits_soft_cap,
                layer._k_scale,
                layer._v_scale,
                None,
                _PARTITION_SIZE_ROCM,
            )
            return output
        else:
            raise NotImplementedError(
                "Cascade attention is not implemented for ROCM AITER")
