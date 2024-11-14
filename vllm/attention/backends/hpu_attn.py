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
from vllm_hpu_extension.runtime import get_config
from vllm_hpu_extension.utils import (FP8Matmul, Matmul, ModuleFusedSDPA,
                                      Softmax, VLLMFP8KVCache, VLLMKVCache)

from vllm.attention.backends.abstract import (AttentionBackend, AttentionImpl,
                                              AttentionLayer,
                                              AttentionMetadata, AttentionType)
from vllm.attention.backends.mla.common import MLACommonImpl
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


class HPUMLAAttentionBackend(AttentionBackend):

    @staticmethod
    def get_name() -> str:
        return "HPU_MLA"

    @staticmethod
    def get_impl_cls() -> Type["HPUMLAImpl"]:
        return HPUMLAImpl

    @staticmethod
    def get_metadata_cls() -> Type["AttentionMetadata"]:
        return HPUMLAMetadata

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
        return (num_blocks * block_size, head_size)

    @staticmethod
    def swap_blocks(
        src_kv_cache: torch.Tensor,
        dst_kv_cache: torch.Tensor,
        src_to_dst: torch.Tensor,
    ) -> None:
        HPUPagedAttention.swap_blocks(src_kv_cache, dst_kv_cache, src_to_dst)

    @staticmethod
    def copy_blocks(
        kv_caches: List[torch.Tensor],
        src_to_dists: torch.Tensor,
    ) -> None:
        HPUPagedAttention.copy_blocks(kv_caches, src_to_dists)


@dataclass
class HPUAttentionMetadata(HPUPagedAttentionMetadata, AttentionMetadata):
    """Metadata for HPUAttentionbackend."""
    # Currently, input sequences can only contain all prompts
    # or all decoding. True if all sequences are prompts.
    is_prompt: bool
    block_size: int
    attn_bias: Optional[torch.Tensor]
    seq_lens_tensor: Optional[torch.Tensor]
    context_lens_tensor: Optional[torch.Tensor]
    input_positions: torch.Tensor
    seq_lens: Optional[List[int]] = None
    encoder_seq_lens: Optional[List[int]] = None
    encoder_seq_lens_tensor: Optional[torch.Tensor] = None
    max_encoder_seq_len: Optional[int] = None
    cross_block_list: Optional[torch.Tensor] = None
    cross_slot_mapping: Optional[torch.Tensor] = None
    cross_block_mapping: Optional[torch.Tensor] = None
    cross_block_groups: Optional[torch.Tensor] = None
    cross_block_usage: Optional[torch.Tensor] = None
    cross_attn_bias: Optional[torch.Tensor] = None


@dataclass
class HPUMLAMetadata(HPUAttentionMetadata, AttentionMetadata):
    pass


class HPUMLAImpl(MLACommonImpl[HPUAttentionMetadata], torch.nn.Module):

    def __init__(
            self,
            num_heads: int,
            head_size: int,
            scale: float,
            num_kv_heads: int,
            alibi_slopes: Optional[List[float]],
            sliding_window: Optional[int],
            kv_cache_dtype: str,
            blocksparse_params: Optional[Dict[str, Any]],
            logits_soft_cap: Optional[float],
            attn_type: str,
            # MLA Specific Arguments
            **kwargs) -> None:
        torch.nn.Module.__init__(self)
        MLACommonImpl.__init__(self, num_heads, head_size, scale, num_kv_heads,
                               alibi_slopes, sliding_window, kv_cache_dtype,
                               blocksparse_params, logits_soft_cap, attn_type,
                               **kwargs)
        self.enable_fp8_attn = kv_cache_dtype == 'fp8_inc' and os.environ.get(
            'QUANT_CONFIG', None) is None
        self.matmul_qk = Matmul() if not self.enable_fp8_attn \
            else FP8Matmul()
        self.softmax = Softmax()
        self.matmul_av = Matmul() if not self.enable_fp8_attn \
            else FP8Matmul()
        self.batch2block_matmul = Matmul() if not self.enable_fp8_attn \
            else FP8Matmul()
        self.block2batch_matmul = Matmul() if not self.enable_fp8_attn \
            else FP8Matmul()
        self.latent_cache_k = VLLMKVCache() if not self.enable_fp8_attn \
            else VLLMFP8KVCache()
        self.fused_scaled_dot_product_attention = kernels.fsdpa()

        self.prefill_impl = get_config().prompt_attn_impl
        assert self.prefill_impl != 'fsdpa_impl' or alibi_slopes is None, \
            'Prefill with FusedSDPA not supported with alibi slopes!'

        unsupported_features = [
            alibi_slopes, sliding_window, blocksparse_params, logits_soft_cap
        ]
        if any(unsupported_features):
            raise NotImplementedError(
                "HPUMLAImpl does not support one of the following: "
                "alibi_slopes, sliding_window, blocksparse_params, "
                "logits_soft_cap")

        if attn_type != AttentionType.DECODER:
            raise NotImplementedError("Encoder self-attention and "
                                      "encoder/decoder cross-attention "
                                      "are not implemented for "
                                      "TritonMLAImpl")

    def forward(
        self,
        layer: AttentionLayer,
        q: torch.Tensor,
        k_c_normed: torch.Tensor,  # key in unified attn
        k_pe: torch.Tensor,  # value in unified attn
        kv_cache: torch.Tensor,
        attn_metadata: HPUAttentionMetadata,
        output: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if output is not None:
            raise NotImplementedError(
                "output is not yet supported for MLAImplBase")

        batch_size = q.shape[0]
        is_prefill = attn_metadata.is_prompt

        # Restore head dim (for rotary embedding)
        k_pe = k_pe.view(-1, 1, self.qk_rope_head_dim)
        q = q.view(-1, self.num_heads, self.qk_head_dim)
        assert hasattr(attn_metadata,
                       "input_positions"), f"attn meta: {attn_metadata}"

        input_positions = attn_metadata.input_positions.view(-1)
        if not is_prefill:
            # decode
            q_nope, q_pe = q.split(
                [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
            # Convert from (B, N, P) to (N, B, P)
            q_nope = q_nope.transpose(0, 1)
            # Multiply (N, B, P) x (N, P, L) -> (N, B, L)
            decode_ql_nope = torch.bmm(q_nope, self.W_UK_T)
            # Convert from (N, B, L) to (B, N, L)
            decode_ql_nope = decode_ql_nope.transpose(0, 1)
            q_pe, k_pe = \
                self.rotary_emb(input_positions, q_pe, k_pe)
        else:
            # prefill
            q_pe = q[..., self.qk_nope_head_dim:]
            q[..., self.qk_nope_head_dim:], k_pe = \
                self.rotary_emb(input_positions, q_pe, k_pe)

        slot_mapping = attn_metadata.slot_mapping.flatten(
        ) if attn_metadata.slot_mapping is not None else None

        latent_vec_k = torch.concat(
            (k_c_normed, k_pe.view(batch_size, -1, self.qk_rope_head_dim)),
            dim=-1)
        latent_vec_k = latent_vec_k.view(
            -1, self.qk_rope_head_dim + self.kv_lora_rank)

        # write the latent and rope to kv cache
        if kv_cache is not None and len(kv_cache) == 2:
            self.latent_cache_k(latent_vec_k, kv_cache[0], slot_mapping)
            k_cache = kv_cache[0]
            v_cache = None

        if is_prefill:
            return self._forward_prefill(q, k_c_normed, k_pe, attn_metadata,
                                         batch_size)
        else:
            return self._forward_decode(decode_ql_nope, q_pe,
                                        (k_cache, v_cache), attn_metadata,
                                        batch_size)

    def _forward_prefill(  # type: ignore
            self, q: torch.Tensor, k_c_normed: torch.Tensor,
            k_pe: torch.Tensor, attn_metadata: HPUAttentionMetadata,
            batch_size: int) -> torch.Tensor:
        kv_nope = self.kv_b_proj(k_c_normed)[0]\
            .view(-1, self.num_heads, self.qk_nope_head_dim + self.v_head_dim)
        k_nope, v = kv_nope\
            .split([self.qk_nope_head_dim, self.v_head_dim], dim=-1)

        k = torch.cat((k_nope, k_pe.expand((*k_nope.shape[:-1], -1))), dim=-1)

        q = q.view(batch_size, -1, self.num_heads, self.qk_head_dim)
        k = k.view(batch_size, -1, self.num_heads, self.qk_head_dim)
        v = v.view(batch_size, -1, self.num_heads, self.v_head_dim)

        to_pad = self.qk_head_dim - self.v_head_dim
        if to_pad > 0:
            v_padding = torch.zeros(*v.shape[:-1],
                                    q.shape[-1] - v.shape[-1],
                                    device=v.device,
                                    dtype=v.dtype)
            v_padded = torch.cat((v, v_padding), dim=-1)
        else:
            v_padded = v

        out = ops.prompt_attention(
            impl=self.prefill_impl,
            query=q,
            key=k,
            value=v_padded,
            is_causal=True,
            attn_bias=attn_metadata.attn_bias,
            valid_seq_lengths=attn_metadata.seq_lens_tensor,
            scale=self.scale,
            matmul_qk_op=self.matmul_qk,
            softmax_op=self.softmax,
            matmul_av_op=self.matmul_av,
            fsdpa_op=self.fused_scaled_dot_product_attention.apply \
            if self.fused_scaled_dot_product_attention is not None else None)
        attn_output = out.view(batch_size, -1, self.num_heads, q.shape[-1])
        attn_output = attn_output[..., :v.shape[-1]]\
                .reshape(batch_size, -1, self.num_heads * v.shape[-1])

        return attn_output

    def _forward_decode(  # type: ignore
            self, q_nope: torch.Tensor, q_pe: torch.Tensor,
            kv_cache: torch.Tensor, attn_metadata: HPUAttentionMetadata,
            batch_size: int) -> torch.Tensor:
        query = torch.cat([q_nope, q_pe], dim=-1)
        key_cache = kv_cache[0].unsqueeze(1)
        value_cache = kv_cache[1]  # value_cache is None
        output = HPUPagedAttention.forward_decode(
            query=query,
            key_cache=key_cache,
            value_cache=value_cache,
            block_list=attn_metadata.block_list,
            block_mapping=attn_metadata.block_mapping,
            block_bias=attn_metadata.attn_bias,
            block_groups=attn_metadata.block_groups,
            block_size=attn_metadata.block_size,
            scale=self.scale,
            matmul_qk_op=self.matmul_qk,
            matmul_av_op=self.matmul_av,
            batch2block_matmul_op=self.batch2block_matmul,
            block2batch_matmul_op=self.block2batch_matmul,
            keys_fetch_func=self.latent_cache_k.fetch_from_cache,
            values_fetch_func=None,
            kv_lora_rank=self.kv_lora_rank)
        output = output.view(batch_size, 1, -1)
        result = self._v_up_proj(output)
        result = result.view(batch_size, 1, -1)
        return result


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
        use_irope: bool = False,
    ) -> None:
        super(AttentionImpl, self).__init__()
        if use_irope:
            logger.warning_once(
                "Using irope in HPU is not supported yet, it will fall back "
                "to global attention for long context.")
        self.enable_fp8_attn = kv_cache_dtype == 'fp8_inc' and os.environ.get(
            'QUANT_CONFIG', None) is None
        self.kv_cache_dtype = kv_cache_dtype
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = float(scale)
        self.matmul_qk = Matmul() if not self.enable_fp8_attn \
            else FP8Matmul()
        self.softmax = Softmax()
        self.matmul_av = Matmul() if not self.enable_fp8_attn \
            else FP8Matmul()
        self.batch2block_matmul = Matmul() if not self.enable_fp8_attn \
            else FP8Matmul()
        self.block2batch_matmul = Matmul() if not self.enable_fp8_attn \
            else FP8Matmul()
        self.k_cache = VLLMKVCache() if not self.enable_fp8_attn \
            else VLLMFP8KVCache()
        self.v_cache = VLLMKVCache() if not self.enable_fp8_attn \
            else VLLMFP8KVCache()
        HPUFusedSDPA = kernels.fsdpa()
        self.fused_scaled_dot_product_attention = None if HPUFusedSDPA is None \
            else ModuleFusedSDPA(HPUFusedSDPA)

        self.prefill_impl = get_config().prompt_attn_impl
        self.use_contiguous_pa = get_config().use_contiguous_pa
        if alibi_slopes is not None:
            assert self.prefill_impl != 'flex_impl', \
                'Prefill with Flex Attention not supported with alibi slopes!'
            assert self.prefill_impl != 'fsdpa_impl', \
                'Prefill with FusedSDPA not supported with alibi slopes!'
            assert self.use_contiguous_pa, \
                'Non-contiguous PA not supported with alibi slopes!'

        self.num_kv_heads = num_heads if num_kv_heads is None else num_kv_heads
        self.sliding_window = sliding_window
        self.prompt_position_bias = None
        self.prev_attn = None
        self.alibi_slopes = None
        if alibi_slopes is not None:
            slope_tensor_dtype = torch.float32 if \
                get_config().fp32_alibi_biases else torch.bfloat16
            alibi_slopes_tensor = torch.tensor(alibi_slopes,
                                               dtype=slope_tensor_dtype)
            self.alibi_slopes = alibi_slopes_tensor

        assert self.num_heads % self.num_kv_heads == 0
        self.num_queries_per_kv = self.num_heads // self.num_kv_heads

        supported_head_sizes = HPUPagedAttention.get_supported_head_sizes()
        if head_size not in supported_head_sizes:
            raise ValueError(
                f"Head size {head_size} is not supported by PagedAttention. "
                f"Supported head sizes are: {supported_head_sizes}.")

        self.attn_type = attn_type
        if (self.attn_type != AttentionType.DECODER
                and self.attn_type != AttentionType.ENCODER_DECODER
                and self.attn_type != AttentionType.ENCODER_ONLY):
            raise NotImplementedError("Encoder self-attention "
                                      "is not implemented for "
                                      "HPUAttentionImpl")

    def _maybe_init_alibi_biases(
        self,
        max_seq_len,
        prev_attn: Optional[torch.nn.Module] = None,
    ) -> None:
        self.max_seq_len = max_seq_len
        self.prev_attn = None if prev_attn is None else prev_attn.impl
        if self.alibi_slopes is not None:
            if self.prev_attn is not None:
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

        key = key.view(-1, self.num_kv_heads, self.head_size)
        value = value.view(-1, self.num_kv_heads, self.head_size)
        slot_mapping = attn_metadata.slot_mapping.flatten(
        ) if attn_metadata.slot_mapping is not None else None
        key_cache = None
        value_cache = None
        if kv_cache is not None and isinstance(kv_cache, tuple):
            key_cache, value_cache = HPUPagedAttention.split_kv_cache(
                kv_cache, self.num_kv_heads, self.head_size)

            # Reshape the input keys and values and store them in the cache.
            # If kv_cache is not provided, the new key and value tensors are
            # not cached. This happens during the initial memory profiling run.
            key_cache = self.k_cache(key, key_cache, slot_mapping)
            value_cache = self.v_cache(value, value_cache, slot_mapping)

        if attn_metadata.is_prompt:
            # Prompt run.
            query_shape = (batch_size, seq_len, self.num_heads, self.head_size)
            kv_shape = (batch_size, seq_len_kv, self.num_kv_heads,
                        self.head_size)

            attn_bias = attn_metadata.attn_bias
            position_bias = None
            # If we have alibi_slopes, incorporate them with
            if (attn_metadata.block_list is None
                    and self.prompt_position_bias is not None
                    and self.alibi_slopes is not None):
                assert attn_bias is not None, \
                        'attn_bias must be set before calling ' \
                        'model.forward with alibi biases'
                slice_1_size = attn_bias.size(-2)
                slice_2_size = attn_bias.size(-1)
                if self.max_seq_len >= max(slice_1_size, slice_2_size):
                    # Using pre-computed prompt_position_bias subset.
                    position_bias = self.prompt_position_bias[:, :,
                                                              -slice_1_size:,
                                                              -slice_2_size:]

                else:
                    # For longer sequences than precomputed,
                    # recreate the bias. This is memory inefficient.
                    position_bias = _make_prompt_alibi_bias(
                        alibi_slopes=self.alibi_slopes,
                        seq_len=max(slice_1_size, slice_2_size),
                        dtype=self.alibi_slopes.dtype,
                    )

            block_list = attn_metadata.block_list if attn_metadata \
                and attn_metadata.block_list is not None else None

            out = ops.prompt_attention(
                impl=self.prefill_impl,
                query=query.view(query_shape),
                key=key.view(kv_shape),
                value=value.view(kv_shape),
                is_causal=True,
                attn_bias=attn_bias,
                position_bias=position_bias,
                valid_seq_lengths=attn_metadata.seq_lens_tensor,
                **self.common_attention_args(block_list, key_cache,
                                             value_cache,
                                             attn_metadata.block_size))
            output = out.reshape(batch_size, seq_len, hidden_size)
        else:
            # Decoding run.
            self.position_bias = None
            alibi_blocks = getattr(attn_metadata, 'alibi_blocks', None)
            if self.alibi_slopes is not None and alibi_blocks is not None:
                if self.prev_attn is not None:
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
                block_mapping=attn_metadata.block_mapping,
                block_bias=attn_metadata.attn_bias,
                block_groups=attn_metadata.block_groups,
                position_bias=self.position_bias,
                **self.common_attention_args(attn_metadata.block_list,
                                             key_cache, value_cache,
                                             attn_metadata.block_size))
        # Reshape the output tensor.
        return output.view(batch_size, seq_len, hidden_size)

    def common_attention_args(self,
                              block_list=None,
                              key_cache=None,
                              value_cache=None,
                              block_size=None):
        return {
            'scale': self.scale,
            'matmul_qk_op': self.matmul_qk,
            'matmul_av_op': self.matmul_av,
            'batch2block_matmul_op': self.batch2block_matmul,
            'block2batch_matmul_op': self.block2batch_matmul,
            'fsdpa_op': self.fused_scaled_dot_product_attention,
            'keys_fetch_func': self.k_cache.fetch_from_cache,
            'values_fetch_func': self.v_cache.fetch_from_cache,
            'softmax_op': self.softmax,
            'block_list': block_list,
            'key_cache': key_cache,
            'value_cache': value_cache,
            'block_size': block_size,
        }

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

        query = query.unsqueeze(1)
        if key is not None:
            assert value is not None
            key = key.view(-1, self.num_kv_heads, self.head_size)
            value = value.view(-1, self.num_kv_heads, self.head_size)
        else:
            assert value is None

        cross_slot_mapping = attn_metadata.cross_slot_mapping.flatten(
        ) if attn_metadata.cross_slot_mapping is not None else None
        if kv_cache is not None and isinstance(kv_cache, tuple):
            key_cache, value_cache = HPUPagedAttention.split_kv_cache(
                kv_cache, self.num_kv_heads, self.head_size)

            # Reshape the input keys and values and store them in the cache.
            # If kv_cache is not provided, the new key and value tensors are
            # not cached. This happens during the initial memory profiling run.
            key_cache = self.k_cache(key, key_cache, cross_slot_mapping)
            value_cache = self.v_cache(value, value_cache, cross_slot_mapping)

        if attn_metadata.is_prompt:
            # Prompt run.
            batch_size = attn_metadata.num_prefills

            query_shape = (batch_size, -1, self.num_heads, self.head_size)
            kv_shape = (batch_size, -1, self.num_kv_heads, self.head_size)
            out = ops.prompt_attention(impl=self.prefill_impl,
                                       query=query.view(query_shape),
                                       key=key.view(kv_shape),
                                       value=value.view(kv_shape),
                                       attn_bias=None,
                                       is_causal=False,
                                       **self.common_attention_args())
            output = out.reshape(batch_size, seq_len, hidden_size)
        else:
            # Enc/dec cross-attention KVs match encoder sequence length;
            # cross-attention utilizes special "cross" block tables
            block_list = attn_metadata.cross_block_list
            block_mapping = attn_metadata.cross_block_mapping
            block_groups = attn_metadata.cross_block_groups
            attn_bias = attn_metadata.cross_attn_bias
            # Decoding run.
            output = HPUPagedAttention.forward_decode(
                query=query,
                block_mapping=block_mapping,
                block_bias=attn_bias,
                block_groups=block_groups,
                position_bias=None,
                **self.common_attention_args(block_list, key_cache,
                                             value_cache,
                                             attn_metadata.block_size))
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
    # Create the bias matrix for positional differences
    bias = torch.arange(seq_len, dtype=dtype, device=alibi_slopes.device)
    bias = bias[None, :] - bias[:, None]  # Shape: [seq_len, seq_len]

    #padded_len = (seq_len + 7) // 8 * 8
    num_heads = alibi_slopes.shape[0]
    per_head_bias = torch.empty(
        1,
        num_heads,
        seq_len,
        seq_len,  # Directly use seq_len instead of padded_len
        device=alibi_slopes.device,
        dtype=dtype,
    )

    # Copy the bias matrix into each head
    per_head_bias[:, :] = bias

    # Scale the bias by the ALiBi slopes
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
