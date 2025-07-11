# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Attention layer with TreeAttention."""

import ast
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional

import torch

try:
    from xformers.ops.fmha import triton_splitk
    from xformers.ops.fmha.attn_bias import (AttentionBias,
                                             PagedBlockDiagonalPaddedKeysMask)
    from xformers.ops.tree_attention import (_get_depth_counts,
                                             _prepare_tree_attn_bias,
                                             tree_attention)
    XFORMERS_AVAILABLE = True
except ImportError:
    XFORMERS_AVAILABLE = False

from vllm.attention.backends.abstract import (AttentionBackend, AttentionImpl,
                                              AttentionMetadata, AttentionType)
from vllm.attention.ops.triton_unified_attention import unified_attention
from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.v1.attention.backends.utils import (AttentionMetadataBuilder,
                                              CommonAttentionMetadata)
from vllm.v1.kv_cache_interface import AttentionSpec

if TYPE_CHECKING:
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.worker.gpu_input_batch import InputBatch

from vllm import _custom_ops as ops

logger = init_logger(__name__)


class TreeAttentionBackend(AttentionBackend):

    accept_output_buffer: bool = True

    @staticmethod
    def get_supported_head_sizes() -> list[int]:
        return [32, 64, 96, 128, 160, 192, 224, 256]

    @staticmethod
    def get_name() -> str:
        return "TREE_ATTN_VLLM_V1"

    @staticmethod
    def get_impl_cls() -> type["TreeAttentionImpl"]:
        return TreeAttentionImpl

    @staticmethod
    def get_metadata_cls() -> type["AttentionMetadata"]:
        return TreeAttentionMetadata

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
    def get_builder_cls() -> type["TreeAttentionMetadataBuilder"]:
        return TreeAttentionMetadataBuilder


@dataclass
class TreeAttentionMetadata:
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

    prefix_attn_bias: Optional["AttentionBias"] = None
    spec_attn_bias: Optional[torch.Tensor] = None

    # Cached Prefill/decode metadata.
    _cached_prefill_metadata: Optional["TreeAttentionMetadata"] = None
    _cached_decode_metadata: Optional["TreeAttentionMetadata"] = None

    @property
    def prefill_metadata(self) -> Optional["TreeAttentionMetadata"]:
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
        self._cached_prefill_metadata = TreeAttentionMetadata(
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
    def decode_metadata(self) -> Optional["TreeAttentionMetadata"]:
        if self.num_decode_tokens == 0:
            return None

        if self._cached_decode_metadata is not None:
            # Recover cached decode-phase attention
            # metadata structure
            return self._cached_decode_metadata

        q_start_loc = self.query_start_loc[:self.num_decodes + 1]
        q_seqlens = torch.diff(q_start_loc)
        kv_seqlens = self.seq_lens[:self.num_decodes]
        # Construct & cache decode-phase attention metadata structure
        self._cached_decode_metadata = TreeAttentionMetadata(
            num_actual_tokens=self.num_decode_tokens,
            max_query_len=int(q_seqlens.max().item()),
            query_start_loc=q_start_loc,
            max_seq_len=int(kv_seqlens.max().item()),
            seq_lens=kv_seqlens,
            block_table=self.block_table[:self.num_decodes],
            slot_mapping=self.slot_mapping[:self.num_decode_tokens],
            prefix_attn_bias=self.prefix_attn_bias,
            spec_attn_bias=self.spec_attn_bias,
        )
        return self._cached_decode_metadata


class TreeAttentionMetadataBuilder(
        AttentionMetadataBuilder[TreeAttentionMetadata]):

    def __init__(self, kv_cache_spec: AttentionSpec, vllm_config: VllmConfig,
                 device: torch.device):
        assert XFORMERS_AVAILABLE
        self.kv_cache_spec = kv_cache_spec
        self.block_size = kv_cache_spec.block_size

        spec_config = vllm_config.speculative_config
        spec_token_tree = spec_config.speculative_token_tree
        tree_choices: list[tuple[int,
                                 ...]] = (ast.literal_eval(spec_token_tree) if
                                          spec_token_tree is not None else [])
        # Construct the tree attention bias.
        depth_counts = _get_depth_counts(tree_choices)
        self.tree_attn_bias = _prepare_tree_attn_bias(
            tree_choices,
            depth_counts,
            dtype=self.kv_cache_spec.dtype,
            device=device,
        )
        self.suffix_attn_bias = self.tree_attn_bias
        self._num_decodes = 0
        self._num_decode_tokens = 0

    def reorder_batch(self, input_batch: "InputBatch",
                      scheduler_output: "SchedulerOutput") -> bool:
        # We now want to reorder the batch so that the "decode" requests are and
        # the front and the "prefill" requests are at the using the least amount
        # swaps possible. (NOTE for now we loosely use "decode" to mean requests
        # where attention is likely memory-bound and "prefill" to mean requests
        # where attention is likely compute-bound, TODO(lucas): figure out a
        # better naming here)
        decodes = []
        prefills = []
        num_decode_tokens = 0

        for i, req_id in enumerate(input_batch.req_ids):
            num_tokens = scheduler_output.num_scheduled_tokens[req_id]
            # for now treat 1 scheduled token as "decode" even if its not,
            # we should update this to something like < 8 in the future but
            # currently the decode run only supports num_tokens = 1
            # For now, treat any decode step with exactly
            if num_tokens == self.suffix_attn_bias.shape[0]:
                decodes.append(i)
                num_decode_tokens += num_tokens
            else:
                prefills.append(i)

        # We hope that this is fairly minimal since decodes
        # should be around for a number of iterations so hopefully they are
        # relatively stationary (and new request are generally appended to the
        # persistent batch so already should be at the back)
        # To achieve this we loop over the decodes in descending order and
        # the prefills in ascending order. We swap decodes from the  "back"
        # i.e. past where the last decode should be in the reodorered with
        # prefills from the front of the batch.
        # `decodes` and `prefills` are already in ascending order just based on
        # the above loop
        num_decodes = len(decodes)
        num_prefills = len(prefills)
        modified_batch = False

        for i in range(1, min(num_decodes, num_prefills) + 1):
            # If the decode is at the "back" of the batch, i, we can swap it
            # with the prefill closest to the front of the batch
            decode_idx = decodes[num_decodes - i]
            if decode_idx < num_decodes:
                break

            input_batch.swap_states(prefills[i - 1], decode_idx)
            modified_batch = True

        # Save for next `build` call
        # TODO(lucas): this is a bit of a hack, we should probably have a
        # better way of doing this
        self._num_decodes = num_decodes
        self._num_decode_tokens = num_decode_tokens

        return modified_batch

    def build(self,
              common_prefix_len: int,
              common_attn_metadata: CommonAttentionMetadata,
              fast_build: bool = False) -> TreeAttentionMetadata:
        num_reqs = common_attn_metadata.num_reqs
        num_decodes = self._num_decodes
        num_prefills = num_reqs - num_decodes
        num_actual_tokens = common_attn_metadata.num_actual_tokens
        num_decode_tokens = self._num_decode_tokens
        num_prefill_tokens = num_actual_tokens - num_decode_tokens
        q_start_loc = common_attn_metadata.query_start_loc
        q_seqlens = torch.diff(q_start_loc)
        max_query_len = common_attn_metadata.max_query_len
        kv_seqlens = common_attn_metadata.seq_lens
        max_seq_len = int(common_attn_metadata.seq_lens_cpu.max())
        block_table = common_attn_metadata.block_table_tensor
        slot_mapping = common_attn_metadata.slot_mapping

        prefix_attn_bias = None
        if num_decodes > 0:
            # Construct the prefix bias.
            decode_q_seqlens = q_seqlens[:num_decodes]
            decode_kv_seqlens = kv_seqlens[:num_decodes]
            prefix_kv_seqlens = decode_kv_seqlens - decode_q_seqlens
            prefix_attn_bias = PagedBlockDiagonalPaddedKeysMask.from_seqlens(
                q_seqlen=decode_q_seqlens.tolist(),
                kv_seqlen=prefix_kv_seqlens.tolist(),
                page_size=self.block_size,
                block_tables=block_table[:num_decodes],
                device=block_table.device,
            )

        return TreeAttentionMetadata(
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
            prefix_attn_bias=prefix_attn_bias,
            spec_attn_bias=self.suffix_attn_bias,
        )

    def build_for_drafting(
        self,
        common_attn_metadata: CommonAttentionMetadata,
        tree_level_offset: int,
    ) -> TreeAttentionMetadata:
        orig_num_decodes = self._num_decodes
        orig_num_decode_tokens = self._num_decode_tokens

        if tree_level_offset == 0:
            # Use prefill for drafting.
            self._num_decodes = 0
            self._num_decode_tokens = 0
            attn_metadata = self.build(0,
                                       common_attn_metadata,
                                       fast_build=True)

        else:
            # While drafting, all requests are treated as decodes.
            self._num_decodes = common_attn_metadata.num_reqs
            self._num_decode_tokens = common_attn_metadata.num_actual_tokens

            # Slice the suffix attention bias so that
            query_len = common_attn_metadata.max_query_len
            start, end = tree_level_offset, tree_level_offset + query_len
            self.suffix_attn_bias = self.tree_attn_bias[
                start:end, start:end].contiguous()

            # Build attention bias.
            attn_metadata = self.build(0,
                                       common_attn_metadata,
                                       fast_build=True)

        # Reset properties to original values.
        self._num_decodes = orig_num_decodes
        self._num_decode_tokens = orig_num_decode_tokens
        self.suffix_attn_bias = self.tree_attn_bias
        return attn_metadata


class TreeAttentionImpl(AttentionImpl):

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
        kv_sharing_target_layer_name: Optional[str] = None,
        use_irope: bool = False,
    ) -> None:
        assert XFORMERS_AVAILABLE
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = float(scale)
        self.num_kv_heads = num_kv_heads
        self.kv_cache_dtype = kv_cache_dtype
        self.num_queries_per_kv = self.num_heads // self.num_kv_heads
        self.kv_sharing_target_layer_name = kv_sharing_target_layer_name
        if alibi_slopes is not None:
            alibi_slopes = torch.tensor(alibi_slopes, dtype=torch.float32)
        self.alibi_slopes = alibi_slopes
        if logits_soft_cap is None:
            # In flash-attn, setting logits_soft_cap as 0 means no soft cap.
            logits_soft_cap = 0
        self.logits_soft_cap = logits_soft_cap
        if sliding_window is None:
            self.sliding_window = (-1, -1)
        else:
            self.sliding_window = (sliding_window - 1, 0)

        support_head_sizes = TreeAttentionBackend.get_supported_head_sizes()
        if head_size not in support_head_sizes:
            raise ValueError(
                f"Head size {head_size} is not supported by TreeAttention. "
                f"Supported head sizes are: {support_head_sizes}. "
                "Set VLLM_USE_V1=0 to use another attention backend.")

    def forward(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: TreeAttentionMetadata,
        output: Optional[torch.Tensor] = None,
        output_scale: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass with TreeAttention.

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
                " for TreeAttentionImpl")

        if attn_metadata is None:
            # Profiling run.
            return output

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
        num_decodes = attn_metadata.num_decodes
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
            # Get only the speculatively decoded q, k, and vs.
            spec_q = query[:num_decode_tokens]
            spec_k = key[:num_decode_tokens]
            spec_v = value[:num_decode_tokens]

            # Reshape q, k, and vs to [B, M, H, D].
            spec_q = spec_q.view(num_decodes, -1, self.num_heads,
                                 self.head_size)
            spec_k = spec_k.view(num_decodes, -1, self.num_kv_heads,
                                 self.head_size)
            spec_v = spec_v.view(num_decodes, -1, self.num_kv_heads,
                                 self.head_size)
            # Reshape the KV cache to [Bkv, Mk, H, D]
            cache_k = key_cache.view(1, -1, self.num_kv_heads, self.head_size)
            cache_v = value_cache.view(1, -1, self.num_kv_heads,
                                       self.head_size)

            if self.num_kv_heads != self.num_heads:
                # GQA/MQA. Reshape q, k, and v to [B, M, G, H, K].
                spec_q = spec_q.view(
                    spec_q.shape[0],
                    spec_q.shape[1],
                    self.num_kv_heads,
                    self.num_queries_per_kv,
                    spec_q.shape[-1],
                )
                spec_k = spec_k[:, :, :, None, :].expand(
                    spec_k.shape[0],
                    spec_k.shape[1],
                    self.num_kv_heads,
                    self.num_queries_per_kv,
                    spec_k.shape[-1],
                )
                spec_v = spec_v[:, :, :, None, :].expand(
                    spec_v.shape[0],
                    spec_v.shape[1],
                    self.num_kv_heads,
                    self.num_queries_per_kv,
                    spec_v.shape[-1],
                )
                # Reshape the KV cache to [Bkv, Mk, G, H, K]
                cache_k = cache_k[:, :, :, None, :].expand(
                    cache_k.shape[0],
                    cache_k.shape[1],
                    self.num_kv_heads,
                    self.num_queries_per_kv,
                    cache_k.shape[-1],
                )
                cache_v = cache_v[:, :, :, None, :].expand(
                    cache_v.shape[0],
                    cache_v.shape[1],
                    self.num_kv_heads,
                    self.num_queries_per_kv,
                    cache_v.shape[-1],
                )

            # Perform tree attention on the speculatively decoded tokens.
            output[:num_decode_tokens] = tree_attention(
                q=spec_q,
                spec_k=spec_k,
                spec_v=spec_v,
                cache_k=cache_k,
                cache_v=cache_v,
                prefix_op=triton_splitk.FwOp,
                suffix_op=triton_splitk.FwOp,
                prefix_attn_bias=decode_meta.prefix_attn_bias,
                spec_attn_bias=decode_meta.spec_attn_bias,
            ).view(-1, self.num_heads, self.head_size)
        return output
