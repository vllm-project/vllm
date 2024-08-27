import operator
from abc import ABC, abstractmethod
from functools import reduce
from typing import List, Tuple

import torch

from vllm.attention.backends.abstract import AttentionBackend
from vllm.config import CacheConfig, ModelConfig, ParallelConfig
from vllm.utils import get_dtype_size


class ExternalSwapperBase(ABC):
    """Base class for external swapper."""

    @staticmethod
    def get_external_swapper_class(external_swapper: str):
        external_swapper = external_swapper.lower()

        if external_swapper.startswith("file://"):
            return LocalFileSwapper

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


class LocalFileSwapper(ExternalSwapperBase):
    """External swapper for local file."""

    def __init__(
        self,
        cache_config: CacheConfig,
        model_config: ModelConfig,
        parallel_config: ParallelConfig,
        dtype: torch.dtype,
        attn_backend: AttentionBackend,
        gpu_cache: List[torch.Tensor],
        cache_engine_identifier: str,
    ) -> None:
        self.head_size = model_config.get_head_size()
        self.num_attention_layers = model_config.get_num_attention_layers(
            parallel_config)
        self.num_kv_heads = model_config.get_num_kv_heads(parallel_config)

        self.block_size = cache_config.block_size
        self.external_swapper_space_bytes = \
            cache_config.external_swapper_space_bytes
        self.num_external_blocks = cache_config.num_external_blocks
        self.external_swapper = cache_config.external_swapper

        self.dtype = dtype
        self.attn_backend = attn_backend
        self.gpu_cache = gpu_cache

        self.cache_engine_identifier = cache_engine_identifier
        self.kv_cache_shape = self.attn_backend.get_kv_cache_shape(
            self.num_external_blocks, self.block_size, self.num_kv_heads,
            self.head_size)

        self.directory = self._parse_swapper_parameters(self.external_swapper)
        self.kv_cache = self._allocate_kv_cache()

    def _parse_swapper_parameters(self,
                                  external_swapper_parameters: str) -> str:
        if not external_swapper_parameters.startswith("file://"):
            raise ValueError(
                f"Invalid external swapper name: {external_swapper_parameters}"
            )

        directory = external_swapper_parameters[len("file://"):].strip()
        return directory

    def _allocate_kv_cache(self) -> List[Tuple[str, str]]:
        dtype_size = get_dtype_size(self.dtype)
        kv_attention_layer_bytes = reduce(operator.mul, self.kv_cache_shape,
                                          1) * dtype_size
        if kv_attention_layer_bytes % 2 != 0:
            raise ValueError(
                f"Invalid kv bytes size: {kv_attention_layer_bytes}")
        key_attention_layer_bytes = int(kv_attention_layer_bytes / 2)

        kv_cache: List[Tuple[str, str]] = []
        for i in range(self.num_attention_layers):
            key_file_name = \
                f"{self.directory}/external_{self.cache_engine_identifier}_layer_{i}_key"
            val_file_name = \
                f"{self.directory}/external_{self.cache_engine_identifier}_layer_{i}_val"

            with open(key_file_name, 'wb') as f:
                f.truncate(key_attention_layer_bytes)
            with open(val_file_name, 'wb') as f:
                f.truncate(key_attention_layer_bytes)
            kv_cache.append((key_file_name, val_file_name))
        return kv_cache

    def swap_out(self, src_to_dst: torch.Tensor) -> None:
        for i in range(self.num_attention_layers):
            self.attn_backend.swap_out_to_local_file(self.gpu_cache[i],
                                                     self.kv_cache[i],
                                                     src_to_dst)

    def swap_in(self, src_to_dst: torch.Tensor) -> None:
        for i in range(self.num_attention_layers):
            self.attn_backend.swap_in_form_local_file(self.kv_cache[i],
                                                      self.gpu_cache[i],
                                                      src_to_dst)
