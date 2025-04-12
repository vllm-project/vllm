# SPDX-License-Identifier: Apache-2.0
"""Attention layer with FlashAttention."""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional

import torch
from attn_gym.masks.document_mask import _offsets_to_doc_ids_tensor
from torch.nn.attention.flex_attention import (BlockMask, _mask_mod_signature,
                                               _score_mod_signature,
                                               create_block_mask,
                                               flex_attention)

from vllm.attention.backends.abstract import (AttentionBackend, AttentionImpl,
                                              AttentionMetadata, AttentionType,
                                              is_quantized_kv_cache)
from vllm.logger import init_logger
from vllm.platforms import current_platform

if current_platform.is_cuda():
    pass

logger = init_logger(__name__)

if TYPE_CHECKING:
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.worker.gpu_input_batch import InputBatch
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner

create_block_mask_compiled = torch.compile(create_block_mask, fullgraph=True)
flex_attention_compiled = torch.compile(flex_attention, fullgraph=True)


class FlexAttentionBackend(AttentionBackend):
    accept_output_buffer: bool = True

    @staticmethod
    def get_supported_head_sizes() -> list[int]:
        return [32, 64, 96, 128, 160, 192, 224, 256]

    @staticmethod
    def get_name() -> str:
        return "FLEX_ATTENTION_VLLM_V1"

    @staticmethod
    def get_impl_cls() -> type["FlexAttentionImpl"]:
        return FlexAttentionImpl

    @staticmethod
    def get_metadata_cls() -> type["AttentionMetadata"]:
        return FlexAttentionMetadata

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
    def get_builder_cls() -> type["FlexAttentionMetadataBuilder"]:
        return FlexAttentionMetadataBuilder

    @staticmethod
    def use_cascade_attention(*args, **kwargs) -> bool:
        return False


# @torch.compile()
def physical_to_logical_mapping(
        block_table: torch.Tensor,
        total_blocks: Optional[int] = None) -> torch.Tensor:
    """
    Creates an inverse mapping from physical block locations to logical indices.

    The original block_table maps from logical blocks to physical locations:

    Logical to Physical (Original block_table):
    ┌───────────────────────────────────────────┐
    │ Request 0:                                │
    │                                           │
    │ Logical Blocks:  0  1  2  3  4  5  6  7   │
    │                  │  │  │  │  │  │  │  │   │
    │                  v  v  v  v  v  v  v  v   │
    │ Physical Blocks: 3  5  1  7  4  2  0  6   │
    └───────────────────────────────────────────┘

    This function creates the inverse mapping:

    Physical to Logical (Inverse mapping):
    ┌───────────────────────────────────────────┐
    │ Request 0:                                │
    │                                           │
    │ Physical Blocks: 0  1  2  3  4  5  6  7   │
    │                  │  │  │  │  │  │  │  │   │
    │                  v  v  v  v  v  v  v  v   │
    │ Logical Blocks:  6  2  5  0  4  1  7  3   │
    └───────────────────────────────────────────┘

    If multiple logical blocks map to the same physical block,
    this function returns the first (minimum) logical block index.

    If a physical block is not mapped to by any logical block,
    its value in the result will be -1.


    Args:
        block_table: Tensor of shape [max_reqs, max_num_blocks] 
            mapping logical blocks to physical locations

    Returns:
        A tensor of shape [max_reqs, max_physical_block]
    """
    max_reqs, max_num_blocks = block_table.shape
    device = block_table.device

    physical_to_logical = torch.full((max_reqs, total_blocks),
                                     -1,
                                     dtype=torch.long,
                                     device=device)

    logical_indices = (torch.arange(max_num_blocks,
                                    device=device).unsqueeze(0).expand(
                                        max_reqs, -1))

    values = block_table.gather(-1, logical_indices)
    physical_to_logical.scatter_(-1, values.to(torch.int64), logical_indices)
    # TODO Confirm - Seems like block 0 is always empty so we reset it manually
    physical_to_logical[:, 0] = -1
    return physical_to_logical


def causal_mask_mod(b: torch.Tensor, h: torch.Tensor, q_idx: torch.Tensor,
                    kv_idx: torch.Tensor):
    return q_idx >= kv_idx


@dataclass
class FlexAttentionMetadata:
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

    # Block info
    total_cache_tokens: int
    block_size: int
    max_possible_sequence_length: int
    num_reqs: int
    physical_to_logical: torch.Tensor
    decode_offset: torch.Tensor

    # For logging.
    num_input_tokens: int = 0  # Number of tokens including padding.

    # Flex Metadata
    num_blocks = 0
    block_mask: Optional[BlockMask] = None
    score_mod: Optional[_score_mod_signature] = None
    mask_mod: Optional[_mask_mod_signature] = None
    logical_mask_mod: _mask_mod_signature = causal_mask_mod

    def get_mask_mod(self) -> _mask_mod_signature:
        """Creates the mask_mod function for FlexAttention.
        
        This function creates the combined mask mod function that handles:
            1. The paged attention block mapping 
            2. The mapping from packed query sequences to logical query entries
        
        It also by defaults adds the decoding offset to the query indices.
        With this info we create the "logical" indices that are passed to
        mask_mod functions. This allows mask mod functions to be agnostic to the
        layout of the query and key/value tensors.

        TODO is_within_lower_bound: do sequences start on block_boundaries?
        """
        # Create a lookup mapping from query indices -> request number
        request_lookup = _offsets_to_doc_ids_tensor(self.query_start_loc)

        def final_mask_mod(
            b: torch.Tensor,
            h: torch.Tensor,
            q_idx: torch.Tensor,
            physical_kv_idx: torch.Tensor,
        ) -> torch.Tensor:
            # Map query indices to corresponding request indices
            q_req = request_lookup[q_idx]

            # Convert physical KV indices to logical indices
            physical_kv_block = physical_kv_idx // self.block_size
            physical_kv_offset = physical_kv_idx % self.block_size
            logical_block_idx = self.physical_to_logical[q_req,
                                                         physical_kv_block]
            logical_kv_idx = logical_block_idx * self.block_size + physical_kv_offset

            # Determine valid kv indices
            live_block = logical_block_idx >= 0
            within_upper_bound = logical_kv_idx < self.seq_lens[q_req]
            within_lower_bound = logical_kv_idx >= 0

            is_valid = live_block & within_upper_bound & within_lower_bound

            # Convert physical query indices to logical indices
            local_q_idx = q_idx - self.query_start_loc[q_req]
            logical_q_idx = local_q_idx + self.decode_offset[q_req]

            # Apply mask modification only for valid indices
            return torch.where(
                is_valid,
                self.logical_mask_mod(b, h, logical_q_idx, logical_kv_idx),
                False,
            )

        return final_mask_mod

    def build_block_mask(self) -> BlockMask:
        assert self.mask_mod is not None
        return create_block_mask_compiled(
            self.mask_mod,
            None,
            None,
            self.num_actual_tokens,
            self.total_cache_tokens,
        )

    def __post_init__(self):
        assert self.use_cascade is False, "Not implemented yet."
        assert self.common_prefix_len == 0, "Not implemented yet."
        assert self.cu_prefix_query_lens is None, "Not implemented yet."
        assert self.prefix_kv_lens is None, "Not implemented yet."
        assert self.suffix_kv_lens is None, "Not implemented yet."
        self.num_blocks = self.total_cache_tokens // self.block_size
        self.mask_mod = self.get_mask_mod()
        self.block_mask = self.build_block_mask()


class FlexAttentionMetadataBuilder:

    def __init__(self, runner: "GPUModelRunner"):
        self.runner = runner

    def reorder_batch(self, input_batch: "InputBatch",
                      scheduler_output: "SchedulerOutput") -> bool:
        return False

    def build(
        self,
        num_reqs: int,
        num_actual_tokens: int,
        max_query_len: int,
        common_prefix_len: int,
    ):
        max_seq_len = self.runner.seq_lens_np[:num_reqs].max()
        query_start_loc = self.runner.query_start_loc_cpu[:num_reqs + 1].to(
            self.runner.device, non_blocking=True)
        seq_lens = self.runner.seq_lens_cpu[:num_reqs].to(self.runner.device,
                                                          non_blocking=True)
        block_table = self.runner.input_batch.block_table.get_device_tensor(
        )[:num_reqs]
        slot_mapping = (self.runner.slot_mapping_cpu[:num_actual_tokens].to(
            self.runner.device, non_blocking=True).long())

        use_cascade = common_prefix_len > 0
        cu_prefix_query_lens = None
        prefix_kv_lens = None
        suffix_kv_lens = None
        if use_cascade:
            raise NotImplementedError("Not yet my friend")

        block_size = self.runner.block_size
        max_possible_seq_len = self.runner.model_config.max_model_len
        total_cache_tokens = (self.runner.cache_config.num_gpu_blocks *
                              block_size)

        inverse_block_table = physical_to_logical_mapping(
            block_table, self.runner.cache_config.num_gpu_blocks)

        # Get the original offset tensor
        offset_tensor = torch.tensor(
            self.runner.input_batch.num_computed_tokens_cpu[:num_reqs]).to(
                self.runner.device, non_blocking=True)

        out = FlexAttentionMetadata(
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
            block_size=block_size,
            max_possible_sequence_length=max_possible_seq_len,
            num_reqs=num_reqs,
            physical_to_logical=inverse_block_table,
            total_cache_tokens=total_cache_tokens,
            decode_offset=offset_tensor,
        )
        return out


class FlexAttentionImpl(AttentionImpl):
    sliding_window: Optional[tuple[int, int]]
    alibi_slopes: Optional[torch.Tensor]
    logits_soft_cap: Optional[float]

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
    ) -> None:
        if blocksparse_params is not None:
            # TODO we should support this :think
            raise ValueError(
                "FlashAttention does not support block-sparse attention.")
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = float(scale)
        self.num_kv_heads = num_kv_heads

        if alibi_slopes is not None:
            self.alibi_slopes = torch.tensor(alibi_slopes, dtype=torch.float32)
        else:
            self.alibi_slopes = None
        if sliding_window is None:
            self.sliding_window = (-1, -1)
        else:
            self.sliding_window = (sliding_window - 1, 0)
        self.kv_cache_dtype = kv_cache_dtype
        self.logits_soft_cap = logits_soft_cap

        assert self.num_heads % self.num_kv_heads == 0
        self.num_queries_per_kv = self.num_heads // self.num_kv_heads

        support_head_sizes = FlexAttentionBackend.get_supported_head_sizes()
        if head_size not in support_head_sizes:
            raise ValueError(
                f"Head size {head_size} is not supported by FlashAttention. "
                f"Supported head sizes are: {support_head_sizes}. "
                "Set VLLM_USE_V1=0 to use another attention backend.")
        if is_quantized_kv_cache(self.kv_cache_dtype):
            raise NotImplementedError(
                "FlexAttention does not support quantized kv-cache. Yet")

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

        if attn_metadata is None:
            # Profiling run.
            return output

        key_cache, value_cache = kv_cache.unbind(0)
        # View out the block_size dim
        key_cache = key_cache.view(-1, self.num_kv_heads, self.head_size)
        value_cache = value_cache.view(-1, self.num_kv_heads, self.head_size)
        write_to_kv_cache(key, value, key_cache, value_cache,
                          attn_metadata.slot_mapping)

        query, key_cache, value_cache = map(
            lambda x: self.view_as_4d(x).permute(0, 2, 1, 3),
            (query, key_cache, value_cache),
        )
        out = flex_attention_compiled(
            query,
            key_cache,
            value_cache,
            attn_metadata.score_mod,
            attn_metadata.block_mask,
            self.scale,
            enable_gqa=enable_gqa,
            kernel_options={"FORCE_USE_FLEX_ATTENTION": False},
        )
        # TODO I wasted 4 hours of my life not realizing
        # I needed to return output
        # Flex doesn't have a non functional form
        out = out.permute(0, 2, 1, 3).squeeze(0)
        output.copy_(out)
        return output


def write_to_kv_cache(
    key: torch.Tensor,
    value: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
) -> None:
    """Writes the key and values to the KV cache using slot_mapping.
    Args:
        key: shape = [num_tokens, num_kv_heads, head_size]
        value: shape = [num_tokens, num_kv_heads, head_size]
        key_cache: shape = [num_blocks, block_size, num_kv_heads, head_size]
        value_cache: shape = [num_blocks, block_size, num_kv_heads, head_size]
        slot_mapping: shape = [num_tokens] maps each token to the cache

    """
    key_cache.index_copy_(0, slot_mapping, key)
    value_cache.index_copy_(0, slot_mapping, value)
