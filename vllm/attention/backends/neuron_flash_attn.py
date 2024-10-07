""" Attention layer with torch scaled_dot_product_attention
    and PagedAttention."""
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Type

import torch
from torch.nn.functional import scaled_dot_product_attention

from vllm.attention.backends.abstract import (AttentionBackend, AttentionImpl,
                                              AttentionMetadata)
from vllm.attention.ops.paged_attn import (PagedAttention,
                                           PagedAttentionMetadata)


class NeuronFlashAttentionBackend(AttentionBackend):

    @staticmethod
    def get_impl_cls() -> Type["NeuronFlashAttentionBackendImpl"]:
        raise NotImplementedError

    @staticmethod
    def make_metadata(*args, **kwargs) -> "NeuronFlashAttentionMetadata":
        return NeuronFlashAttentionMetadata(*args, **kwargs)

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
    ) -> Tuple[int, ...]:
        raise NotImplementedError

    @staticmethod
    def swap_blocks(
        src_kv_cache: torch.Tensor,
        dst_kv_cache: torch.Tensor,
        src_to_dst: Dict[int, int],
    ) -> None:
        raise NotImplementedError

    @staticmethod
    def copy_blocks(
        kv_caches: List[torch.Tensor],
        src_to_dists: Dict[int, List[int]],
    ) -> None:
        raise NotImplementedError


@dataclass
class NeuronFlashAttentionMetadata(AttentionMetadata, PagedAttentionMetadata):
    """Metadata for NeuronFlashAttentionBackend.
    """
    # Currently, input sequences can only contain all prompts
    # or all decoding. True if all sequences are prompts.
    is_prompt: bool
    slot_mapping: torch.Tensor
    seq_lens: Optional[List[int]]
    context_lens: Optional[torch.Tensor]
    # prompt_lens stored as a tensor.
    prompt_lens_tensor: Optional[torch.Tensor]

    def __post_init__(self):
        # Set during the execution of the first attention op.
        # It is a list because it is needed to set per prompt
        # when alibi slopes is used. It is because of the limitation
        # from xformer API.
        # will not appear in the __repr__ and __init__
        self.attn_bias: Optional[List[torch.Tensor]] = None
