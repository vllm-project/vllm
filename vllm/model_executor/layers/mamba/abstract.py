# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from abc import abstractmethod
from collections.abc import Iterable
from math import prod

import torch

from vllm.config import VllmConfig
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.utils.torch_utils import get_dtype_size
from vllm.v1.attention.backend import AttentionBackend
from vllm.v1.attention.backends.registry import MambaAttentionBackendEnum
from vllm.v1.attention.selector import get_mamba_attn_backend
from vllm.v1.kv_cache_interface import KVCacheSpec, MambaSpec


class MambaBase(AttentionLayerBase):
    """
    Base class for Mamba-like layers which support the v1 engine.
    Inherit from this class if you implement a custom layer.
    """

    # Contains the KV cache (mamba state) for the layer
    # in the shape specified by `self.get_state_shape`.
    kv_cache: tuple[torch.Tensor, ...]
    supports_dcp: bool = False

    def bind_kv_cache(self, kv_cache: torch.Tensor) -> None:
        """Unpack a raw ``[B, 1, 1, C]`` int8 page view into per-state views.

        Each block's ``C`` bytes hold the layer's states (e.g. conv, ssm)
        packed contiguously; slice them out and reinterpret per dtype/shape.
        """
        pages = kv_cache.squeeze(dim=(1, 2))
        states: list[torch.Tensor] = []
        offset = 0
        for shape, dtype in zip(self.get_state_shape(), self.get_state_dtype()):
            nbytes = prod(shape) * get_dtype_size(dtype)
            state = pages[:, offset : offset + nbytes].view(dtype)
            states.append(state.view(-1, *shape))
            offset += nbytes
        self.kv_cache = tuple(states)

    def bind_replayssm_cache(self, cache: tuple[torch.Tensor, ...]) -> None:
        expected_shapes = self.get_replayssm_state_shape()
        expected_dtypes = self.get_replayssm_state_dtype()
        assert len(cache) == len(expected_shapes) == len(expected_dtypes)
        assert all(
            tuple(state.shape[1:]) == shape and state.dtype == dtype
            for state, shape, dtype in zip(cache, expected_shapes, expected_dtypes)
        )
        self.kv_cache = (*self.kv_cache, *cache)

    @abstractmethod
    def get_state_shape(self) -> Iterable[tuple[int, ...]]:
        """
        Defines the shape of the state.
        For mamba layers this is usually a (conv_state, ssm_state) tuple.
        In this case, returns (conv_state_shape, ssm_state_shape).
        """
        pass

    @property
    @abstractmethod
    def mamba_type(self) -> MambaAttentionBackendEnum:
        pass

    @property
    def is_kv_cache_tp_replicated(self) -> bool:
        return False

    @abstractmethod
    def get_state_dtype(self) -> tuple[torch.dtype, ...]:
        pass

    def get_replayssm_state_shape(self) -> tuple[tuple[int, ...], ...]:
        return ()

    def get_replayssm_state_dtype(self) -> tuple[torch.dtype, ...]:
        return ()

    def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec | None:
        mamba_block_size = vllm_config.cache_config.mamba_block_size
        assert mamba_block_size is not None
        page_size_padded = vllm_config.cache_config.mamba_page_size_padded
        return MambaSpec(
            shapes=tuple(self.get_state_shape()),
            dtypes=self.get_state_dtype(),
            replayssm_shapes=self.get_replayssm_state_shape(),
            replayssm_dtypes=self.get_replayssm_state_dtype(),
            block_size=mamba_block_size,
            page_size_padded=page_size_padded,
            mamba_type=self.mamba_type,
            tp_replicated=self.is_kv_cache_tp_replicated,
            mamba_cache_mode=vllm_config.cache_config.mamba_cache_mode,
            # ReplaySSM verifies the whole window off one checkpoint, so it
            # never writes the baseline's per-draft-token state slots.
            num_speculative_blocks=(
                0
                if vllm_config.cache_config.use_replayssm
                else vllm_config.num_speculative_tokens
            ),
        )

    def get_attn_backend(self) -> type[AttentionBackend]:
        """Get the attention backend class for this Mamba layer."""
        return get_mamba_attn_backend(self.mamba_type)
