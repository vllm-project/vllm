from abc import ABC, abstractmethod
from typing import (Tuple, List)
from functools import reduce
import operator
import torch
import os

from vllm.config import CacheConfig, ModelConfig, ParallelConfig
from vllm.utils import get_dtype_size
from vllm.attention.backends.abstract import AttentionBackend


class ExternalSwapperBase(ABC):
    """Base class for external swapper."""

    @staticmethod
    def get_external_swapper_class(external_swapper: str):
        external_swapper = external_swapper.lower()

        raise ValueError(f"Unknown external_swapper_type {external_swapper=}")

    @abstractmethod
    def _allocate_kv_cache(self) -> List[Tuple[str, str]]:
        """Allocate KV cache."""
        raise NotImplementedError

    @abstractmethod
    def swap_out(self, src_to_dst: torch.Tensor) -> None:
        """Swap out blocks from GPU -> NVMf."""
        raise NotImplementedError

    @abstractmethod
    def swap_in(self, src_to_dst: torch.Tensor) -> None:
        """Swap in blocks from NVMf -> GPU."""
        raise NotImplementedError
