from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple, Type

import torch


class AttentionBackend(ABC):
    """Abstract class for attention backends."""

    @abstractmethod
    @staticmethod
    def get_attention_impl_cls() -> Type["AttentionImpl"]:
        raise NotImplementedError

    @abstractmethod
    @staticmethod
    def get_attention_metadata_cls() -> Type["AttentionMetadata"]:
        raise NotImplementedError


class AttentionMetadata:

    is_prompt: bool
    slot_mapping: torch.Tensor
    prompt_lens: Optional[torch.Tensor]
    max_seq_len: Optional[int]
    start_loc: Optional[torch.Tensor]
    max_context_len: Optional[int]
    context_lens: Optional[torch.Tensor]
    block_tables: Optional[torch.Tensor]
    use_cuda_graph: bool
    kv_cache_dtype: str


class AttentionImpl(ABC):

    @abstractmethod
    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: Optional[int] = None,
        alibi_slopes: Optional[List[float]] = None,
        sliding_window: Optional[int] = None,
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: AttentionMetadata,
    ) -> torch.Tensor:
        raise NotImplementedError
