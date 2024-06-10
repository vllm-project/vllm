from dataclasses import dataclass
from typing import List, Tuple

import openvino as ov
import torch

from vllm.attention.backends.abstract import (AttentionBackend,
                                              AttentionMetadata)


class OpenVINOAttentionBackend(AttentionBackend):

    @staticmethod
    def get_name() -> str:
        return "openvino"

    @staticmethod
    def get_impl_cls():
        # OpenVINO implements PagedAttention as part of the Optimum
        # exported model
        raise NotImplementedError

    @staticmethod
    def make_metadata(*args, **kwargs) -> "AttentionMetadata":
        raise NotImplementedError

    @staticmethod
    def make_openvino_metadata(*args, **kwargs) -> "OpenVINOAttentionMetadata":
        return OpenVINOAttentionMetadata(*args, **kwargs)

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
    ) -> Tuple[int, ...]:
        return (2, num_blocks, num_kv_heads, block_size, head_size)

    @staticmethod
    def swap_blocks(
        src_kv_cache: ov.Tensor,
        dst_kv_cache: ov.Tensor,
        src_to_dst: torch.Tensor,
    ) -> None:
        # OpenVINO currently supports only CPU, which does not require
        # swap of KV cache blocks
        raise NotImplementedError

    @staticmethod
    def copy_blocks(
        kv_caches: List[Tuple[ov.Tensor, ov.Tensor]],
        src_to_dists: List[Tuple[int, int]],
    ) -> None:
        for src, dst in src_to_dists:
            for key_cache, value_cache in kv_caches:
                key_cache.data[dst, :] = key_cache.data[src, :]
                value_cache.data[dst, :] = value_cache.data[src, :]


@dataclass
class OpenVINOAttentionMetadata:
    """Metadata for OpenVINOAttentionBackend.
    """
    past_lens: torch.Tensor
    subsequence_begins: torch.Tensor
    block_indices: torch.Tensor
    block_indices_begins: torch.Tensor
    max_context_len: torch.Tensor
