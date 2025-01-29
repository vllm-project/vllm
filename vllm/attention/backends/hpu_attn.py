###############################################################################
# Copyright (C) 2024 Habana Labs, Ltd. an Intel Company
###############################################################################

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Type

import torch
import vllm_hpu_extension.kernels as kernels
import vllm_hpu_extension.ops as ops
from vllm_hpu_extension.flags import enabled_flags
from vllm_hpu_extension.utils import (Matmul, ModuleFusedSDPA, Softmax,
                                      VLLMKVCache)

from vllm.attention.backends.abstract import (AttentionBackend, AttentionImpl,
                                              AttentionLayer,
                                              AttentionMetadata, AttentionType)
from vllm.attention.backends.utils import CommonAttentionState
from vllm.attention.ops.hpu_paged_attn import (HPUPagedAttention,
                                               HPUPagedAttentionMetadata)
from vllm.logger import init_logger

logger = init_logger(__name__)


class HPUAttentionBackend(AttentionBackend):

    @staticmethod
    def get_name() -> str:
        return "HPU_ATTN"

    @staticmethod
    def get_impl_cls() -> Type["HPUAttentionImpl"]:
        return HPUAttentionImpl

    @staticmethod
    def get_metadata_cls() -> Type["AttentionMetadata"]:
        return HPUAttentionMetadata

    @staticmethod
    def get_state_cls() -> Type["CommonAttentionState"]:
        return CommonAttentionState

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
    ) -> Tuple[int, ...]:
        return HPUPagedAttention.get_kv_cache_shape(num_blocks, block_size,
                                                    num_kv_heads, head_size)

    @staticmethod
    def swap_blocks(
        src_kv_cache: torch.Tensor,
        dst_kv_cache: torch.Tensor,
        src_to_dsts: torch.Tensor,
    ) -> None:
        HPUPagedAttention.swap_blocks(src_kv_cache, dst_kv_cache, src_to_dsts)

    @staticmethod
    def copy_blocks(
        kv_caches: List[torch.Tensor],
        src_to_dsts: torch.Tensor,
    ) -> None:
        HPUPagedAttention.copy_blocks(kv_caches, src_to_dsts)


@dataclass
class HPUAttentionMetadata(HPUPagedAttentionMetadata, AttentionMetadata):
    """Metadata for HPUAttentionbackend."""
    # Currently, input sequences can only contain all prompts
    # or all decoding. True if all sequences are prompts.
    is_prompt: bool
    attn_bias: Optional[torch.Tensor]
    seq_lens_tensor: Optional[torch.Tensor]
    context_lens_tensor: Optional[torch.Tensor]
    seq_lens: Optional[List[int]] = None
    encoder_seq_lens: Optional[List[int]] = None
    encoder_seq_lens_tensor: Optional[torch.Tensor] = None
    cross_block_indices: Optional[torch.Tensor] = None
    cross_block_offsets: Optional[torch.Tensor] = None
    cross_block_list: Optional[torch.Tensor] = None
    cross_slot_mapping: Optional[torch.Tensor] = None
    cross_block_mapping: Optional[torch.Tensor] = None
    cross_block_groups: Optional[torch.Tensor] = None
    cross_block_scales: Optional[torch.Tensor] = None
    cross_block_usage: Optional[torch.Tensor] = None
    cross_attn_bias: Optional[torch.Tensor] = None


class HPUAttentionImpl(AttentionImpl, torch.nn.Module):
    """
    If the input tensors contain prompt tokens, the layout is as follows:
    |<--------------- num_prefill_tokens ----------------->|
    |<--prefill_0-->|<--prefill_1-->|...|<--prefill_N-1--->|

    Otherwise, the layout is as follows:
    |<----------------- num_decode_tokens ------------------>|
    |<--decode_0-->|..........|<--decode_M-1-->|<--padding-->|

    Generation tokens can contain padding when cuda-graph is used.
    Currently, prompt tokens don't contain any padding.

    The prompts might have different lengths, while the generation tokens
    always have length 1.
    """

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
        max_seq_len: int = 4096,
        attn_type: str = AttentionType.DECODER,
    ) -> None:
        super(AttentionImpl, self).__init__()
        self.kv_cache_dtype = kv_cache_dtype
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = float(scale)
        self.matmul_qk = Matmul()
        self.softmax = Softmax()
        self.matmul_av = Matmul()
        self.batch2block_matmul = Matmul()
        self.block2batch_matmul = Matmul()
        self.k_cache = VLLMKVCache()
        self.v_cache = VLLMKVCache()
        HPUFusedSDPA = kernels.fsdpa()
        self.fused_scaled_dot_product_attention = None if HPUFusedSDPA is None \
            else ModuleFusedSDPA(HPUFusedSDPA)
        self.num_kv_heads = num_heads if num_kv_heads is None else num_kv_heads
        self.sliding_window = sliding_window
        self.alibi_slopes = alibi_slopes
        if alibi_slopes is not None:
            alibi_slopes_tensor = torch.tensor(alibi_slopes,
                                               dtype=torch.bfloat16)
            self.alibi_slopes = alibi_slopes_tensor
        assert self.num_heads % self.num_kv_heads == 0
        self.num_queries_per_kv = self.num_heads // self.num_kv_heads

        self.prefill_use_fusedsdpa = "fsdpa" in enabled_flags()
        if self.prefill_use_fusedsdpa:
            assert alibi_slopes is None, \
                'Prefill with FusedSDPA not supported with alibi slopes!'

        suppored_head_sizes = HPUPagedAttention.get_supported_head_sizes()
        if head_size not in suppored_head_sizes:
            raise ValueError(
                f"Head size {head_size} is not supported by PagedAttention. "
                f"Supported head sizes are: {suppored_head_sizes}.")

        self.attn_type = attn_type
        if (self.attn_type != AttentionType.DECODER
                and self.attn_type != AttentionType.ENCODER_DECODER):
            raise NotImplementedError("Encoder self-attention "
                                      "is not implemented for "
                                      "HPUAttentionImpl")

    def forward(
        self,
        layer: AttentionLayer,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: HPUAttentionMetadata,
        output: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass with xFormers and PagedAttention.

        Args:
            query: shape = [num_tokens, num_heads * head_size]
            key: shape = [num_tokens, num_kv_heads * head_size]
            value: shape = [num_tokens, num_kv_heads * head_size]
            kv_cache = [2, num_blocks, block_size * num_kv_heads * head_size]
            attn_metadata: Metadata for attention.
        Returns:
            shape = [num_tokens, num_heads * head_size]
        """
        assert layer._k_scale_float == 1.0 and layer._v_scale_float == 1.0
        if self.attn_type == AttentionType.ENCODER_DECODER:
            return self.forward_encoder_decoder(
                query=query,
                key=key,
                value=value,
                kv_cache=kv_cache,
                attn_metadata=attn_metadata,
                k_scale=layer._k_scale_float,
                v_scale=layer._k_scale_float,
            )

        batch_size, seq_len, hidden_size = query.shape
        _, seq_len_kv, _ = key.shape

        query = query.view(-1, self.num_heads, self.head_size)
        key = key.view(-1, self.num_kv_heads, self.head_size)
        value = value.view(-1, self.num_kv_heads, self.head_size)
        block_indices = attn_metadata.block_indices
        block_offsets = attn_metadata.block_offsets
        if attn_metadata.is_prompt:
            key = key.unflatten(0, (block_indices.size(0), -1))
            value = value.unflatten(0, (block_indices.size(0), -1))
        if kv_cache is not None and isinstance(kv_cache, tuple):
            key_cache, value_cache = HPUPagedAttention.split_kv_cache(
                kv_cache, self.num_kv_heads, self.head_size)

            # Reshape the input keys and values and store them in the cache.
            # If kv_cache is not provided, the new key and value tensors are
            # not cached. This happens during the initial memory profiling run.
            key_cache = self.k_cache(key, key_cache, block_indices,
                                     block_offsets)
            value_cache = self.v_cache(value, value_cache, block_indices,
                                       block_offsets)

        if attn_metadata.is_prompt:
            # Prompt run.
            query_shape = (batch_size, seq_len, self.num_heads, self.head_size)
            kv_shape = (batch_size, seq_len_kv, self.num_kv_heads,
                        self.head_size)
            if attn_metadata is None or attn_metadata.block_list is None:
                if not self.prefill_use_fusedsdpa:
                    # TODO: move this outside of model
                    assert attn_metadata.attn_bias is not None, \
                            'attn_bias must be set before calling model.forward'
                    attn_bias = attn_metadata.attn_bias
                    if self.alibi_slopes is not None:
                        position_bias = _make_alibi_bias(
                            self.alibi_slopes, self.num_kv_heads,
                            attn_bias.dtype, attn_bias.shape[-1])
                        attn_bias = attn_bias.tile(
                            (1, self.num_kv_heads, 1, 1))
                        attn_bias.add_(position_bias)
                else:
                    attn_bias = None

                out = ops.prompt_attention(
                    query.view(query_shape),
                    key.view(kv_shape),
                    value.view(kv_shape),
                    attn_bias=attn_bias,
                    p=0.0,
                    scale=self.scale,
                    matmul_qk_op=self.matmul_qk,
                    softmax_op=self.softmax,
                    matmul_av_op=self.matmul_av,
                    valid_seq_lengths=attn_metadata.seq_lens_tensor,
                    fsdpa_op=self.fused_scaled_dot_product_attention,
                )
            else:
                # TODO: enable FusedSDPA
                out = HPUPagedAttention.forward_prefix(
                    query=query.view(query_shape),
                    key=key.view(kv_shape),
                    value=value.view(kv_shape),
                    key_cache=key_cache,
                    value_cache=value_cache,
                    block_list=attn_metadata.block_list,
                    attn_bias=attn_metadata.attn_bias,
                    scale=self.scale,
                    matmul_qk_op=self.matmul_qk,
                    matmul_av_op=self.matmul_av,
                    softmax_op=self.softmax,
                    keys_fetch_func=self.k_cache.fetch_from_cache,
                    values_fetch_func=self.v_cache.fetch_from_cache)
            output = out.reshape(batch_size, seq_len, hidden_size)
        else:
            # Decoding run.
            output = HPUPagedAttention.forward_decode(
                query=query,
                key_cache=key_cache,
                value_cache=value_cache,
                block_list=attn_metadata.block_list,
                block_mapping=attn_metadata.block_mapping,
                block_bias=attn_metadata.attn_bias,
                block_scales=attn_metadata.block_scales,
                block_groups=attn_metadata.block_groups,
                scale=self.scale,
                matmul_qk_op=self.matmul_qk,
                matmul_av_op=self.matmul_av,
                batch2block_matmul_op=self.batch2block_matmul,
                block2batch_matmul_op=self.block2batch_matmul,
                keys_fetch_func=self.k_cache.fetch_from_cache,
                values_fetch_func=self.v_cache.fetch_from_cache)
        # Reshape the output tensor.
        return output.view(batch_size, seq_len, hidden_size)

    def forward_encoder_decoder(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: HPUAttentionMetadata,
        k_scale: float = 1.0,
        v_scale: float = 1.0,
    ) -> torch.Tensor:
        """Forward pass with xFormers and PagedAttention.

        Args:
            query: shape = [num_tokens, num_heads * head_size]
            key: shape = [num_tokens, num_kv_heads * head_size]
            value: shape = [num_tokens, num_kv_heads * head_size]
            kv_cache = [2, num_blocks, block_size * num_kv_heads * head_size]
            attn_metadata: Metadata for attention.
        Returns:
            shape = [num_tokens, num_heads * head_size]
        """
        batch_size, hidden_size = query.shape

        if attn_metadata.is_prompt:
            batch_size = attn_metadata.num_prefills
            batched_tokens, _ = query.shape
            batched_kv_tokens, _, _ = key.shape
            assert batch_size > 0, (
                "In prefill stage the num_prefills should be > 0")
            assert batched_tokens % batch_size == 0
            assert batched_kv_tokens % batch_size == 0
            seq_len = batched_tokens // batch_size

        query = query.view(-1, self.num_heads, self.head_size)
        if key is not None:
            assert value is not None
            key = key.view(-1, self.num_kv_heads, self.head_size)
            value = value.view(-1, self.num_kv_heads, self.head_size)
        else:
            assert value is None

        block_indices = attn_metadata.cross_block_indices
        block_offsets = attn_metadata.cross_block_offsets
        if kv_cache is not None and isinstance(kv_cache, tuple):
            key_cache, value_cache = HPUPagedAttention.split_kv_cache(
                kv_cache, self.num_kv_heads, self.head_size)

            # Reshape the input keys and values and store them in the cache.
            # If kv_cache is not provided, the new key and value tensors are
            # not cached. This happens during the initial memory profiling run.
            if (key is not None) and (value is not None):
                # During cross-attention decode, key & value will be None,
                # we don't need to cache them.
                key_cache = self.k_cache(key, key_cache, block_indices,
                                         block_offsets)
                value_cache = self.v_cache(value, value_cache, block_indices,
                                           block_offsets)

        if attn_metadata.is_prompt:
            # Prompt run.
            batch_size = attn_metadata.num_prefills

            query_shape = (batch_size, -1, self.num_heads, self.head_size)
            kv_shape = (batch_size, -1, self.num_kv_heads, self.head_size)
            # Just a workaround, to make ops.prompt_attention go into the
            # torch ops assembly path.
            # TODO: add new prompt_attention op in vllm_hpu_extension
            # which calls FusedSDPA with causal = False.
            attn_bias = torch.zeros((batch_size, 1, 1, 1),
                                    device=query.device,
                                    dtype=torch.bool)
            out = ops.prompt_attention(
                query.view(query_shape),
                key.view(kv_shape),
                value.view(kv_shape),
                attn_bias=attn_bias,
                p=0.0,
                scale=self.scale,
                matmul_qk_op=self.matmul_qk,
                softmax_op=self.softmax,
                matmul_av_op=self.matmul_av,
            )
            output = out.reshape(batch_size, seq_len, hidden_size)
        else:
            # Enc/dec cross-attention KVs match encoder sequence length;
            # cross-attention utilizes special "cross" block tables
            block_list = attn_metadata.cross_block_list
            block_mapping = attn_metadata.cross_block_mapping
            block_scales = attn_metadata.cross_block_scales
            block_groups = attn_metadata.cross_block_groups
            attn_bias = attn_metadata.cross_attn_bias
            # Decoding run.
            output = HPUPagedAttention.forward_decode(
                query=query,
                key_cache=key_cache,
                value_cache=value_cache,
                block_list=block_list,
                block_mapping=block_mapping,
                block_bias=attn_bias,
                block_scales=block_scales,
                block_groups=block_groups,
                scale=self.scale,
                matmul_qk_op=self.matmul_qk,
                matmul_av_op=self.matmul_av,
                batch2block_matmul_op=self.batch2block_matmul,
                block2batch_matmul_op=self.block2batch_matmul,
                keys_fetch_func=self.k_cache.fetch_from_cache,
                values_fetch_func=self.v_cache.fetch_from_cache)
        # Reshape the output tensor.
        return output.view(batch_size, -1, hidden_size)


def _make_alibi_bias(
    alibi_slopes: torch.Tensor,
    num_kv_heads: int,
    dtype: torch.dtype,
    seq_len: int,
) -> torch.Tensor:
    bias = torch.arange(seq_len, dtype=dtype)
    # NOTE(zhuohan): HF uses
    #     `bias = bias[None, :].repeat(seq_len, 1)`
    # here. We find that both biases give the same results, but
    # the bias below more accurately follows the original ALiBi
    # paper.
    # Calculate a matrix where each element represents ith element- jth
    # element.
    bias = bias[None, :] - bias[:, None]

    padded_len = (seq_len + 7) // 8 * 8
    num_heads = alibi_slopes.shape[0]
    bias = torch.empty(
        1,  # batch size
        num_heads,
        seq_len,
        padded_len,
        device=alibi_slopes.device,
        dtype=dtype,
    )[:, :, :, :seq_len].copy_(bias)
    bias.mul_(alibi_slopes[:, None, None])
    if num_heads != num_kv_heads:
        bias = bias.unflatten(1, (num_kv_heads, num_heads // num_kv_heads))
    return bias
