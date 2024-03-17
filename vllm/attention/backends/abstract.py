from abc import ABC, abstractmethod
from typing import List, Optional

import torch


class AttentionBackend(ABC):
    """Abstract class for attention backends."""

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
        kv_cache,  # FIXME
        input_metadata,  # FIXME
    ) -> torch.Tensor:
        raise NotImplementedError
