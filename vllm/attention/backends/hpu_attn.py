# SPDX-License-Identifier: Apache-2.0

###############################################################################
# Copyright (C) 2024-2025 Habana Labs, Ltd. an Intel Company
###############################################################################

import os
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
from vllm.distributed import get_tensor_model_parallel_rank
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
        logits_soft_cap: Optional[float] = None,
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
        self.prompt_position_bias = None
        self.tp_rank = get_tensor_model_parallel_rank()
        self.prev_attn = None
        self.alibi_slopes = None
        if alibi_slopes is not None:
            slope_tensor_dtype = {
                True: torch.float32,
                False: torch.bfloat16,
            }["fp32_alibi_biases" in enabled_flags()]
            alibi_slopes_tensor = torch.tensor(alibi_slopes,
                                               dtype=slope_tensor_dtype)
            self.alibi_slopes = alibi_slopes_tensor

        assert self.num_heads % self.num_kv_heads == 0
        self.num_queries_per_kv = self.num_heads // self.num_kv_heads

        self.prefill_use_fusedsdpa = "fsdpa" in enabled_flags()
        if self.prefill_use_fusedsdpa:
            assert alibi_slopes is None, \
                'Prefill with FusedSDPA not supported with alibi slopes!'
            try:
                from habana_frameworks.torch.hpex.kernels import FusedSDPA
                self.fused_scaled_dot_product_attention = ModuleFusedSDPA(
                    FusedSDPA)
            except ImportError:
                logger().warning("Could not import HPU FusedSDPA kernel. "
                                 "vLLM will use native implementation.")

        self.prefill_use_flex_attention = "flex_attention" in enabled_flags()

        self.use_contiguous_pa = "contiguous_pa" in enabled_flags()
        if not self.use_contiguous_pa:
            assert alibi_slopes is None, \
                'Non-contiguous PA not supported with alibi slopes!'

        suppored_head_sizes = HPUPagedAttention.get_supported_head_sizes()
        if head_size not in suppored_head_sizes:
            raise ValueError(
                f"Head size {head_size} is not supported by PagedAttention. "
                f"Supported head sizes are: {suppored_head_sizes}.")

        self.attn_type = attn_type
        if (self.attn_type != AttentionType.DECODER
                and self.attn_type != AttentionType.ENCODER_DECODER
                and self.attn_type != AttentionType.ENCODER_ONLY):
            raise NotImplementedError("Encoder self-attention "
                                      "is not implemented for "
                                      "HPUAttentionImpl")

    def _maybe_init_alibi_biases(
        self,
        max_seq_len: int = 4096,
        prev_attn: Optional[torch.nn.Module] = None,
    ) -> None:
        # Set upper bound on sequence length
        max_seq_len_upper = int(
            os.getenv(
                'VLLM_PROMPT_ALIBI_MAX_SEQ_LEN',
                max_seq_len,
            ))
        # Set lower bound on sequence length
        self.max_seq_len = max([
            max_seq_len_upper,
            int(os.getenv('VLLM_PROMPT_SEQ_BUCKET_MAX', '0')),
        ])
        self.prev_attn = None if prev_attn is None else prev_attn.impl
        if self.alibi_slopes is not None:
            if (self.prev_attn is not None
                    and self.prev_attn.tp_rank == self.tp_rank):
                self.alibi_slopes = self.prev_attn.alibi_slopes
                self.prompt_position_bias = self.prev_attn.prompt_position_bias
            else:
                # Creating the prompt_position_bias once and reusing it
                # if seq_len permits.
                self.prompt_position_bias = _make_prompt_alibi_bias(
                    alibi_slopes=self.alibi_slopes,
                    seq_len=self.max_seq_len,
                    dtype=self.alibi_slopes.dtype,
                )

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
        if attn_metadata.is_prompt and self.attn_type \
            is not AttentionType.ENCODER_ONLY:
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
                if (not self.prefill_use_fusedsdpa
                        and not self.prefill_use_flex_attention):
                    # TODO: move this outside of model
                    assert attn_metadata.attn_bias is not None, \
                            'attn_bias must be set before calling model.forward'
                    # If we have alibi_slopes, incorporate them with
                    attn_bias = attn_metadata.attn_bias
                    position_bias = None
                    if (self.prompt_position_bias is not None
                            and self.alibi_slopes is not None):
                        if self.max_seq_len >= max(attn_bias.size(-2),
                                                   attn_bias.size(-1)):
                            # Using pre-computed prompt_position_bias subset.
                            position_bias = self.prompt_position_bias[:, :,
                                                                      -attn_bias
                                                                      .size(-2
                                                                            ):,
                                                                      -attn_bias
                                                                      .size(-1
                                                                            ):]
                        else:
                            # For longer sequences than precomputed,
                            # recreate the bias. This is memory inefficient.
                            position_bias = _make_prompt_alibi_bias(
                                alibi_slopes=self.alibi_slopes,
                                seq_len=max(attn_bias.size(-2),
                                            attn_bias.size(-1)),
                                dtype=self.alibi_slopes.dtype,
                            )
                else:
                    attn_bias = attn_metadata.attn_bias
                    position_bias = None

                if not self.prefill_use_flex_attention:
                    out = ops.prompt_attention(
                        query.view(query_shape),
                        key.view(kv_shape),
                        value.view(kv_shape),
                        attn_bias=attn_bias,
                        position_bias=position_bias,
                        p=0.0,
                        scale=self.scale,
                        matmul_qk_op=self.matmul_qk,
                        softmax_op=self.softmax,
                        matmul_av_op=self.matmul_av,
                        valid_seq_lengths=attn_metadata.seq_lens_tensor,
                        fsdpa_op=self.fused_scaled_dot_product_attention
                        if self.prefill_use_fusedsdpa else None,
                    )
                else:
                    out = ops.flex_attention(
                        query.view(query_shape),
                        key.view(kv_shape),
                        value.view(kv_shape),
                        scale=self.scale,
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
            self.position_bias = None
            alibi_blocks = attn_metadata.alibi_blocks
            if self.alibi_slopes is not None and alibi_blocks is not None:
                if (self.prev_attn is not None
                        and self.prev_attn.tp_rank == self.tp_rank):
                    self.position_bias = self.prev_attn.position_bias
                else:
                    # For decoding, compute position bias using alibi_blocks.
                    self.position_bias = _make_decode_alibi_bias(
                        alibi_blocks=alibi_blocks,
                        alibi_slopes=self.alibi_slopes,
                        dtype=self.alibi_slopes.dtype,
                    )

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
                position_bias=self.position_bias,
                matmul_qk_op=self.matmul_qk,
                matmul_av_op=self.matmul_av,
                batch2block_matmul_op=self.batch2block_matmul,
                block2batch_matmul_op=self.block2batch_matmul,
                keys_fetch_func=self.k_cache.fetch_from_cache,
                values_fetch_func=self.v_cache.fetch_from_cache,
            )

        # Reshape the output tensor.
        output = output.view(batch_size, seq_len, hidden_size)
        return output

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
                fsdpa_op=self.fused_scaled_dot_product_attention
                if self.prefill_use_fusedsdpa else None,
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


def _make_prompt_alibi_bias(
    alibi_slopes: torch.Tensor,
    seq_len: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    """
    Create the ALiBi position bias tensor for prompt stage.
    This tensor is reused or tiled as needed for each forward pass.
    Does not scale with batch size or number of blocks.

    Args:
        alibi_slopes: shape = [num_heads]
        seq_len: int
        dtype: torch.dtype

    Returns:
        A per-head bias tensor of shape [1, num_heads, seq_len, seq_len].
        This bias encodes positional information via ALiBi slopes.
    """
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
    per_head_bias = torch.empty(
        1,
        num_heads,
        seq_len,
        padded_len,
        device=alibi_slopes.device,
        dtype=dtype,
    )[:, :, :, :seq_len]
    # NOTE(Tanner):
    # .copy_ was not performing broadcasting of bias
    # to all 32 heads in Eager mode.
    per_head_bias[:, :] = bias
    per_head_bias.mul_(alibi_slopes[:, None, None])

    return per_head_bias


def _make_decode_alibi_bias(
    alibi_blocks: torch.Tensor,
    alibi_slopes: torch.Tensor,
    dtype: torch.dtype,
) -> torch.Tensor:
    """
    Create the ALiBi position bias tensor for decode stage.
    Uses stored alibi_blocks and slopes for final scaling.
    Scales with number of blocks, not with batch size.

    Args:
        alibi_blocks: shape = [num_blocks, block_size]
        alibi_slopes: shape = [num_heads]
        dtype: torch.dtype

    Returns:
        A per-head bias tensor of shape [num_blocks, num_heads, block_size].
        Each row encodes position-dependent ALiBi slopes for decoding steps.
    """
    num_heads = alibi_slopes.shape[0]
    per_head_bias = torch.empty(
        alibi_blocks.size(0),
        num_heads,
        alibi_blocks.size(-1),
        device=alibi_slopes.device,
        dtype=dtype,
    )
    # NOTE(Tanner):
    # .copy_ was not performing broadcasting of bias
    # to all 32 heads in Eager mode.
    per_head_bias[:, :] = alibi_blocks.unsqueeze(-2)
    per_head_bias.mul_(alibi_slopes[None, :, None])

    return per_head_bias
