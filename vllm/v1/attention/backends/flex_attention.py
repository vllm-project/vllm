# SPDX-License-Identifier: Apache-2.0
"""Attention layer with FlashAttention."""
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Type

import torch

from vllm.attention.backends.abstract import (AttentionBackend, AttentionImpl,
                                              AttentionMetadata, AttentionType)
from vllm.logger import init_logger
from vllm.platforms import current_platform

if current_platform.is_cuda():
    from vllm.vllm_flash_attn import flash_attn_varlen_func

logger = init_logger(__name__)

from torch.nn.attention.flex_attention import (BlockMask, _mask_mod_signature,
                                               _score_mod_signature,
                                               flex_attention)


class FlexAttentionBackend(AttentionBackend):

    accept_output_buffer: bool = True

    @staticmethod
    def get_supported_head_sizes() -> List[int]:
        return [32, 64, 96, 128, 160, 192, 224, 256]

    @staticmethod
    def get_name() -> str:
        return "FLEX_ATTENTION_VLLM_V1"

    @staticmethod
    def get_impl_cls() -> Type["FlexAttentionImpl"]:
        return FlexAttentionImpl

    @staticmethod
    def get_metadata_cls() -> Type["AttentionMetadata"]:
        return FlexAttentionMetadata

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
    ) -> Tuple[int, ...]:
        if block_size % 16 != 0:
            raise ValueError("Block size must be a multiple of 16.")
        return (2, num_blocks, block_size, num_kv_heads, head_size)

    @staticmethod
    def get_builder_cls() -> type["FlexAttentionMetadataBuilder"]:
        return FlexAttentionMetadataBuilder

    @staticmethod
    def use_cascade_attention(*args, **kwargs) -> bool:
        return False


@dataclass
class FlexAttentionMetadata:
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

    use_cascade: bool
    common_prefix_len: int
    cu_prefix_query_lens: Optional[torch.Tensor]
    prefix_kv_lens: Optional[torch.Tensor]
    suffix_kv_lens: Optional[torch.Tensor]

    # For logging.
    num_input_tokens: int = 0  # Number of tokens including padding.

    # Flex Metadata
    block_mask: Optional[BlockMask] = None
    score_mod: Optional[_score_mod_signature] = None
    mask_mod: Optional[_mask_mod_signature] = None

    def __post_init__(self):
        assert self.use_cascade is False, "Not implemented yet."
        assert self.common_prefix_len == 0, "Not implemented yet."
        assert self.cu_prefix_query_lens is None, "Not implemented yet."
        assert self.prefix_kv_lens is None, "Not implemented yet."
        assert self.suffix_kv_lens is None, "Not implemented yet."


class FlexAttentionMetadataBuilder:

    def __init__(self, runner: "GPUModelRunner"):
        self.runner = runner

    def reorder_batch(self, input_batch: "InputBatch",
                      scheduler_output: "SchedulerOutput") -> bool:
        return False

    def build(self, num_reqs: int, num_actual_tokens: int, max_query_len: int,
              common_prefix_len: int):

        max_seq_len = self.runner.seq_lens_np[:num_reqs].max()
        query_start_loc = self.runner.query_start_loc_cpu[:num_reqs + 1].to(
            self.runner.device, non_blocking=True)
        seq_lens = self.runner.seq_lens_cpu[:num_reqs].to(self.runner.device,
                                                          non_blocking=True)
        block_table = (
            self.runner.input_batch.block_table.get_device_tensor()[:num_reqs])
        slot_mapping = self.runner.slot_mapping_cpu[:num_actual_tokens].to(
            self.runner.device, non_blocking=True).long()

        use_cascade = common_prefix_len > 0
        if use_cascade:
            # TODO: Optimize.
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

        return FlexAttentionMetadata(
            num_actual_tokens=num_actual_tokens,
            max_query_len=max_query_len,
            query_start_loc=query_start_loc,
            max_seq_len=max_seq_len,
            seq_lens=seq_lens,
            block_table=block_table,
            slot_mapping=slot_mapping,
            use_cascade=use_cascade,
            common_prefix_len=common_prefix_len,
            cu_prefix_query_lens=cu_prefix_query_lens,
            prefix_kv_lens=prefix_kv_lens,
            suffix_kv_lens=suffix_kv_lens,
        )


class FlexAttentionImpl(AttentionImpl):
    sliding_window: Optional[int]
    alibi_slopes: List[float]
    logits_soft_cap: Optional[float]

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int,
        alibi_slopes: Optional[List[float]],
        sliding_window: Optional[int],
        kv_cache_dtype: str,
        blocksparse_params: Optional[Dict[str, Any]] = None,
        logits_soft_cap: Optional[float] = None,
        attn_type: AttentionType = AttentionType.DECODER,
    ) -> None:
        if blocksparse_params is not None:
            breakpoint()  # we should support this :think
            raise ValueError(
                "FlashAttention does not support block-sparse attention.")
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = float(scale)
        self.num_kv_heads = num_kv_heads

    @staticmethod
    def view_as_4d(tensor: torch.Tensor) -> torch.Tensor:
        """View a 3d tensor as 4D."""
        if tensor.ndim == 4:
            return tensor
        assert tensor.ndim == 3
        return tensor[None, :, :, :]

    def forward(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: FlexAttentionMetadata,
        output: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass with FLexAttention.

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
        enable_gqa = self.num_kv_heads != self.num_heads
        # batch_size, seq_len, hidden_size = query.shape
        query, key, value = map(
            lambda x: self.view_as_4d(x).permute(0, 2, 1, 3),
            (query, key, value))

        # if kv_cache.numel() > 0:
        #     slot_mapping = attn_metadata.slot_mapping
        #     key_cache, value_cache = kv_cache
        #     write_to_kv_cache(key, value, key_cache, value_cache, slot_mapping)

        if attn_metadata is None:
            # Profiling run.
            return output
        # import fbvscode
        # fbvscode.set_trace()
        return flex_attention(
            query,
            key,
            value,
            attn_metadata.score_mod,
            attn_metadata.block_mask,
            self.scale,
            enable_gqa=enable_gqa,
        )

        # IMPORTANT!
        # NOTE(woosuk): With piece-wise CUDA graphs, this method is executed in
        # eager-mode PyTorch. Thus, we need to be careful about any CPU overhead
        # in this method. For example, `view` and `slice` (or `[:n]`) operations
        # are surprisingly slow even in the case they do not invoke any GPU ops.
        # Minimize the PyTorch ops in this method as much as possible.
        # Whenever making a change in this method, please benchmark the
        # performance to make sure it does not introduce any overhead.

        num_actual_tokens = attn_metadata.num_actual_tokens
        # Reshape the input keys and values and store them in the cache.
        # NOTE(woosuk): Here, key and value are padded while slot_mapping is
        # not padded. However, we don't need to do key[:num_actual_tokens] and
        # value[:num_actual_tokens] because the reshape_and_cache_flash op uses
        # the slot_mapping's shape to determine the number of actual tokens.
        key_cache, value_cache = kv_cache.unbind(0)
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

        # Compute attention and update output up to `num_actual_tokens`.
        if not attn_metadata.use_cascade:
            # Regular attention (common case).
            flash_attn_varlen_func(
                q=query[:num_actual_tokens],
                k=key_cache,
                v=value_cache,
                out=output[:num_actual_tokens],
                cu_seqlens_q=attn_metadata.query_start_loc,
                max_seqlen_q=attn_metadata.max_query_len,
                seqused_k=attn_metadata.seq_lens,
                max_seqlen_k=attn_metadata.max_seq_len,
                softmax_scale=self.scale,
                causal=True,
                alibi_slopes=self.alibi_slopes,
                window_size=self.sliding_window,
                block_table=attn_metadata.block_table,
                softcap=self.logits_soft_cap,
                fa_version=self.vllm_flash_attn_version,
            )
            return output

        return output


def write_to_kv_cache(
    key: torch.Tensor,
    value: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
) -> None:
    key = key.flatten(0, 2)
    value = value.flatten(0, 2)
    key_cache = key_cache.flatten(0, 2)
    value_cache = value_cache.flatten(0, 2)
    key_cache.index_copy_(0, slot_mapping, key)
    value_cache.index_copy_(0, slot_mapping, value)


# def use_cascade_attention(
#     common_prefix_len: int,
#     query_lens: np.ndarray,
#     num_query_heads: int,
#     num_kv_heads: int,
#     use_alibi: bool,
#     use_sliding_window: bool,
#     num_sms: int,
# ) -> bool:
#     """Decide whether to use cascade attention.

#     This function 1) checks whether cascade attention is supported with the
#     given configuration, and 2) heuristically decides whether using cascade
#     attention can improve performance.
#     """
#     # Too short common prefix. Probably not worth using cascade attention.
#     # We use an arbitrary threshold of 256 tokens. TODO: Tune this threshold.
#     # NOTE(woosuk): This is the common case. We should return False as soon as
#     # possible to avoid any unnecessary computation.
#     if common_prefix_len < 256:
#         return False
#     # Cascade attention is currently not supported with these variants.
#     if use_alibi or use_sliding_window:
#         return False
#     # Too few queries. Probably not worth using cascade attention.
#     # We use an arbitrary threshold of 8 queries. TODO: Tune this threshold.
#     num_reqs = len(query_lens)
#     if num_reqs < 8:
#         return False

#     # Heuristics to decide whether using cascade attention is beneficial.
#     # 1. When FlashDecoding is not used for normal attention, cascade attention
#     #    is likely to be faster since it saves memory bandwidth.
#     num_queries_per_kv = num_query_heads // num_kv_heads
#     # The criteria for using FlashDecoding can be found in the following link:
#     # https://github.com/vllm-project/flash-attention/blob/96266b1111111f3d11aabefaf3bacbab6a89d03c/csrc/flash_attn/flash_api.cpp#L535
#     use_flash_decoding = (num_queries_per_kv > 1 and not use_sliding_window
#                           and not use_alibi and np.all(query_lens == 1))
#     if not use_flash_decoding:
#         # Use cascade attention.
#         return True

#     # 2. When FlashDecoding is used for normal attention, it is not clear
#     #    whether cascade attention is beneficial, because FlashDecoding can
#     #    launch more CTAs than cascade attention.
#     #    We use a simple performance model to compare the two methods.
#     #    NOTE(woosuk): The performance model is very rough and may not be
#     #    accurate.
#     num_tokens = num_reqs
#     # NOTE(woosuk): These are default tile sizes. flash-attn might use
#     # different tile sizes (e.g., 64 or 256) depending on the configuration.
#     q_tile_size = 128
#     kv_tile_size = 128
#     num_prefix_tiles = cdiv(common_prefix_len, kv_tile_size)

#     cascade_ctas = num_query_heads * cdiv(num_tokens, q_tile_size)
#     cascade_waves = cdiv(cascade_ctas, num_sms)
#     cascade_time = cascade_waves * num_prefix_tiles

#     flash_decoding_ctas = (num_reqs * num_kv_heads *
#                            cdiv(num_queries_per_kv, q_tile_size))
#     flash_decoding_ctas *= num_prefix_tiles
#     flash_decoding_time = cdiv(flash_decoding_ctas, num_sms)

#     # Use cascade attention if it is faster than FlashDecoding.
#     return cascade_time < flash_decoding_time

# def cascade_attention(
#     output: torch.Tensor,
#     query: torch.Tensor,
#     key_cache: torch.Tensor,
#     value_cache: torch.Tensor,
#     cu_query_lens: torch.Tensor,
#     max_query_len: int,
#     cu_prefix_query_lens: torch.Tensor,
#     prefix_kv_lens: torch.Tensor,
#     suffix_kv_lens: torch.Tensor,
#     max_kv_len: int,
#     softmax_scale: float,
#     alibi_slopes: Optional[torch.Tensor],
#     sliding_window: Tuple[int, int],
#     logits_soft_cap: float,
#     block_table: torch.Tensor,
#     common_prefix_len: int,
#     fa_version: int,
# ) -> torch.Tensor:
#     assert alibi_slopes is None, ("Cascade attention does not support ALiBi.")
#     # TODO: Support sliding window.
#     assert sliding_window == (-1, -1), (
#         "Cascade attention does not support sliding window.")

#     num_tokens = query.shape[0]
#     block_size = key_cache.shape[-3]
#     assert common_prefix_len % block_size == 0
#     num_common_kv_blocks = common_prefix_len // block_size
#     assert num_common_kv_blocks > 0

#     # Process shared prefix.
#     prefix_output, prefix_lse = flash_attn_varlen_func(
#         q=query,
#         k=key_cache,
#         v=value_cache,
#         cu_seqlens_q=cu_prefix_query_lens,
#         seqused_k=prefix_kv_lens,
#         max_seqlen_q=num_tokens,
#         max_seqlen_k=common_prefix_len,
#         softmax_scale=softmax_scale,
#         causal=False,
#         window_size=sliding_window,
#         block_table=block_table[:1],
#         softcap=logits_soft_cap,
#         return_softmax_lse=True,
#         fa_version=fa_version,
#     )

#     # Process suffix per query.
#     suffix_output, suffix_lse = flash_attn_varlen_func(
#         q=query,
#         k=key_cache,
#         v=value_cache,
#         cu_seqlens_q=cu_query_lens,
#         seqused_k=suffix_kv_lens,
#         max_seqlen_q=max_query_len,
#         max_seqlen_k=max_kv_len - common_prefix_len,
#         softmax_scale=softmax_scale,
#         causal=True,
#         window_size=sliding_window,
#         block_table=block_table[:, num_common_kv_blocks:],
#         softcap=logits_soft_cap,
#         return_softmax_lse=True,
#         fa_version=fa_version,
#     )

#     # Merge prefix and suffix outputs, and store the result in output.
#     merge_attn_states(output, prefix_output, prefix_lse, suffix_output,
#                       suffix_lse)
