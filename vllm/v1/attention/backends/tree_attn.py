# SPDX-License-Identifier: Apache-2.0
"""Attention layer with TreeAttention."""

import ast
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional

import torch
from xformers.ops.fmha import triton_splitk
from xformers.ops.fmha.attn_bias import (AttentionBias,
                                         PagedBlockDiagonalPaddedKeysMask)
from xformers.ops.tree_attention import tree_attention

from vllm.attention.backends.abstract import (AttentionBackend, AttentionImpl,
                                              AttentionMetadata, AttentionType)
from vllm.logger import init_logger
from vllm.v1.attention.backends.flash_attn import (
    FlashAttentionImpl, FlashAttentionMetadata, FlashAttentionMetadataBuilder)
from vllm.v1.attention.backends.utils import (AttentionMetadataBuilder,
                                              CommonAttentionMetadata)
from vllm.v1.kv_cache_interface import AttentionSpec
from vllm.v1.worker.block_table import BlockTable

if TYPE_CHECKING:
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.worker.gpu_input_batch import InputBatch
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner

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
    prefix_attn_bias: Optional[AttentionBias]
    spec_attn_bias: Optional[torch.Tensor]

    # Attention metadata for prefill.
    prefill_attn_metadata: Optional[FlashAttentionMetadata]


class TreeAttentionMetadataBuilder(
        AttentionMetadataBuilder[TreeAttentionMetadata]):

    def __init__(
        self,
        runner: "GPUModelRunner",
        kv_cache_spec: AttentionSpec,
        block_table: BlockTable,
    ):
        self.runner = runner
        self.kv_cache_spec = kv_cache_spec
        self.block_table = block_table
        self.block_size = kv_cache_spec.block_size

        spec_config = runner.vllm_config.speculative_config
        spec_token_tree = spec_config.speculative_token_tree
        self.tree_choices: list[tuple[int, ...]] = (
            ast.literal_eval(spec_token_tree)
            if spec_token_tree is not None else [])
        self.tree_size = len(self.tree_choices) + 1

        self.prefill_attn_metadata_builder: FlashAttentionMetadataBuilder = (
            FlashAttentionMetadataBuilder(
                runner,
                kv_cache_spec,
                block_table,
            ))

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
        num_prefill_tokens = 0

        for i, req_id in enumerate(input_batch.req_ids):
            num_tokens = scheduler_output.num_scheduled_tokens[req_id]
            # for now treat 1 scheduled token as "decode" even if its not,
            # we should update this to something like < 8 in the future but
            # currently the decode run only supports num_tokens = 1
            # For now, treat any decode step with exactly
            if num_tokens == self.tree_size:
                decodes.append(i)
                num_decode_tokens += num_tokens
            else:
                prefills.append(i)
                num_prefill_tokens += num_tokens

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
        self._num_prefills = num_prefills
        self._num_decode_tokens = num_decode_tokens
        self._num_prefill_tokens = num_prefill_tokens

        return modified_batch

    def build(
        self, common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata
    ) -> TreeAttentionMetadata:
        num_reqs = common_attn_metadata.num_reqs
        num_decodes = self._num_decodes
        num_prefills = self._num_prefills
        num_actual_tokens = common_attn_metadata.num_actual_tokens
        num_decode_tokens = self._num_decode_tokens
        num_prefill_tokens = self._num_prefill_tokens
        q_start_loc = common_attn_metadata.query_start_loc
        q_seqlens = torch.diff(q_start_loc)
        max_query_len = common_attn_metadata.max_query_len
        kv_seqlens = common_attn_metadata.seq_lens
        max_seq_len = int(self.runner.seq_lens_np[:num_reqs].max())
        block_table = self.block_table
        slot_mapping = block_table.slot_mapping

        # If there are any prefill requests, construct the prefill
        # attention metadata.
        prefill_attn_metadata = None
        if num_prefills > 0:
            # Temporarily set the block table slot mapping tensor to the
            # slice for prefill.
            block_table.slot_mapping = slot_mapping[num_decode_tokens:]
            # Build prefill attention metadata.
            prefill_attn_metadata = self.prefill_attn_metadata_builder.build(
                common_prefix_len,
                CommonAttentionMetadata(
                    query_start_loc=q_start_loc[num_decodes:] -
                    q_start_loc[num_decodes],
                    seq_lens=kv_seqlens[num_decodes:],
                    num_reqs=num_prefills,
                    num_actual_tokens=num_prefill_tokens,
                    max_query_len=int(q_seqlens[num_decodes:].max().item()),
                ),
            )
            # Restore block table slot mapping to the original, full tensor.
            block_table.slot_mapping = slot_mapping

        # Get the block table and slot mapping for paged KV.
        block_table_tensor = block_table.get_device_tensor()[:num_reqs]
        slot_mapping[:num_decode_tokens].copy_(
            block_table.slot_mapping_cpu[:num_decode_tokens],
            non_blocking=True,
        )
        # Fill unused with -1. Needed for reshape_and_cache in full cuda graph
        # mode.
        slot_mapping[num_actual_tokens:].fill_(-1)

        prefix_attn_bias = None
        spec_attn_bias = None
        if num_decodes > 0:
            # Construct the prefix bias.
            decode_q_seqlens = q_seqlens[:num_decodes]
            decode_kv_seqlens = kv_seqlens[:num_decodes]
            prefix_kv_seqlens = decode_kv_seqlens - decode_q_seqlens
            prefix_attn_bias = PagedBlockDiagonalPaddedKeysMask.from_seqlens(
                q_seqlen=decode_q_seqlens.tolist(),
                kv_seqlen=prefix_kv_seqlens.tolist(),
                page_size=self.block_size,
                block_tables=block_table_tensor[:num_decodes],
                device=block_table.device,
            )
            # Construct the tree attention (suffix) bias.
            spec_attn_bias = _prepare_tree_attn_bias(
                self.tree_choices,
                self.kv_cache_spec.dtype,
                device=block_table.device,
            ).T

        return TreeAttentionMetadata(
            num_actual_tokens=num_actual_tokens,
            max_query_len=max_query_len,
            query_start_loc=q_start_loc,
            max_seq_len=max_seq_len,
            seq_lens=kv_seqlens,
            block_table=block_table_tensor,
            slot_mapping=slot_mapping,
            prefix_attn_bias=prefix_attn_bias,
            spec_attn_bias=spec_attn_bias,
            prefill_attn_metadata=prefill_attn_metadata,
        )


def _get_depth_counts(sorted_tree_choices: list[tuple[int, ...]]) -> list[int]:
    # Initialize depth_counts to keep track of how many choices have a
    # particular depth.
    depth_counts = []
    prev_depth = 0
    for path in sorted_tree_choices:
        depth = len(path)
        if depth != prev_depth:
            depth_counts.append(0)
        depth_counts[depth - 1] += 1
        prev_depth = depth
    return depth_counts


def _prepare_tree_attn_bias(
    sorted_tree_choices: list[tuple[int, ...]],
    dtype: Optional[torch.dtype],
    device: Optional[torch.device],
) -> torch.Tensor:
    """
    Construct a Medusa-style tree attention bias as an explicit tensor.
    It can be used as a spec_attn_bias ("right" or "suffix" attention part)
    in tree_attention. See run_tree_attention_inner in test for a usage example.
    Args:
        sorted_tree_choices: tree description in the style of
            https://github.com/FasterDecoding/Medusa/blob/5e9805386/medusa/model/medusa_choices.py
            A typical tree description would look like:
            [(node0, node1, ...),
             (node0, node2),
             (node0, node3),
             (node1, node3), ...,
             (node0, node2, ..., nodeN)]
            Every tuple is corresponds to one node in the tree, encoded as a
            path from one of the root nodes to the node in question. Passed
            in sorted order.

            For example, a node encoded as (1, 0, 3, ..., 2) is understood as:
            list all the root nodes and take node number 1
            list all children of that node and take node number 0
            list all children of that node and take node number 3
            ...
            list all children of that node and take node number 2 - that's the
            node encoded by this tuple
        dtype: data type of the output tensor.
        device: device of the output tensor.
    Returns:
        attention bias of shape (tree_size, tree_size),
        where tree_size is the total number of nodes in the tree.
    """
    depth_counts = _get_depth_counts(sorted_tree_choices)

    # +1 comes from the additional root node
    tree_len = len(sorted_tree_choices) + 1
    tree_attn_mask = torch.full((tree_len, tree_len),
                                -torch.inf,
                                device=device,
                                dtype=dtype)

    mask_val = 0
    for i in range(tree_len):
        tree_attn_mask[i, i] = mask_val

    tree_attn_mask[:, 0] = mask_val
    start = 0
    for i in range(len(depth_counts)):
        for j in range(depth_counts[i]):
            cur_tree_choice = sorted_tree_choices[start + j]
            # retrieve ancestor position
            if len(cur_tree_choice) == 1:
                continue
            ancestor_idx = []
            for c in range(len(cur_tree_choice) - 1):
                ancestor_idx.append(
                    sorted_tree_choices.index(cur_tree_choice[:c + 1]) + 1)
            tree_attn_mask[j + start + 1, ancestor_idx] = mask_val
        start += depth_counts[i]
    return tree_attn_mask


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
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = float(scale)
        self.num_kv_heads = num_kv_heads
        self.kv_cache_dtype = kv_cache_dtype
        self.num_queries_per_kv = self.num_heads // self.num_kv_heads
        self.kv_sharing_target_layer_name = kv_sharing_target_layer_name

        support_head_sizes = TreeAttentionBackend.get_supported_head_sizes()
        if head_size not in support_head_sizes:
            raise ValueError(
                f"Head size {head_size} is not supported by TreeAttention. "
                f"Supported head sizes are: {support_head_sizes}. "
                "Set VLLM_USE_V1=0 to use another attention backend.")

        self.prefill_attention_impl = FlashAttentionImpl(
            num_heads=num_heads,
            head_size=head_size,
            scale=scale,
            num_kv_heads=num_kv_heads,
            alibi_slopes=alibi_slopes,
            sliding_window=sliding_window,
            kv_cache_dtype=kv_cache_dtype,
            blocksparse_params=blocksparse_params,
            logits_soft_cap=logits_soft_cap,
            attn_type=attn_type,
            kv_sharing_target_layer_name=
            None,  # Skip KV reshape and cache. This class handles it.
            use_irope=use_irope,
        )

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

        num_decode_tokens = attn_metadata.num_actual_tokens
        num_decodes = attn_metadata.query_start_loc.shape[0] - 1
        prefill_attn_metadata = attn_metadata.prefill_attn_metadata
        if prefill_attn_metadata is not None:
            num_decode_tokens -= prefill_attn_metadata.num_actual_tokens
            num_decodes -= prefill_attn_metadata.query_start_loc.shape[0] - 1
            # Perform prefill flash attention.
            self.prefill_attention_impl.forward(
                layer,
                query[num_decode_tokens:],
                key[num_decode_tokens:],
                value[num_decode_tokens:],
                kv_cache,
                prefill_attn_metadata,
                output[num_decode_tokens:],
                None,
            )

        if num_decodes == 0:
            # No decode requests, abort early.
            return output

        # Get only the speculatively decoded q, k, and vs.
        spec_q = query[:num_decode_tokens]
        spec_k = key[:num_decode_tokens]
        spec_v = value[:num_decode_tokens]

        # Reshape q, k, and vs to [B, M, H, D].
        spec_q = spec_q.view(num_decodes, -1, self.num_heads, self.head_size)
        spec_k = spec_k.view(num_decodes, -1, self.num_kv_heads,
                             self.head_size)
        spec_v = spec_v.view(num_decodes, -1, self.num_kv_heads,
                             self.head_size)
        # Reshape the KV cache to [Bkv, Mk, H, D]
        cache_k = key_cache.view(1, -1, self.num_kv_heads, self.head_size)
        cache_v = value_cache.view(1, -1, self.num_kv_heads, self.head_size)

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
            prefix_attn_bias=attn_metadata.prefix_attn_bias,
            spec_attn_bias=attn_metadata.spec_attn_bias,
        ).view(-1, self.num_heads, self.head_size)
        return output
