from abc import ABC, abstractmethod
from typing import Any, List, Optional, Tuple, Type

import torch


class AttentionBackend(ABC):
    """Abstract class for attention backends."""

    @abstractmethod
    @staticmethod
    def get_attention_impl_cls() -> Type["AttentionImpl"]:
        raise NotImplementedError


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
        input_metadata: Any,
    ) -> torch.Tensor:
        raise NotImplementedError
