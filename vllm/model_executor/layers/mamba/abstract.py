# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from abc import abstractmethod
from collections.abc import Iterable

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

    kv_cache: tuple[torch.Tensor, ...]
    _mamba_spec: MambaSpec | None = None

    def set_kv_cache(self, value: torch.Tensor) -> None:
        """Unpack a raw 4D tensor into per-state strided views."""
        spec = self._mamba_spec
        assert spec is not None
        byte_offset = value.storage_offset() * value.element_size()
        total_bytes = value.numel() * value.element_size()
        num_blocks = total_bytes // spec.page_size_bytes
        storage_tensor = torch.tensor([], dtype=torch.int8, device=value.device).set_(
            value.untyped_storage()
        )
        raw_1d = storage_tensor[
            byte_offset : byte_offset + num_blocks * spec.page_size_bytes
        ]
        self.kv_cache = tuple(
            self._unpack_states(
                raw_1d,
                spec.shapes,
                spec.dtypes,
                spec.page_size_bytes,
                num_blocks,
            )
        )

    @staticmethod
    def _unpack_states(
        raw: torch.Tensor,
        shapes: tuple[tuple[int, ...], ...],
        dtypes: tuple[torch.dtype, ...],
        page_size_bytes: int,
        num_blocks: int,
    ) -> list[torch.Tensor]:
        """Unpack a flat raw tensor into per-state strided views."""
        state_tensors: list[torch.Tensor] = []
        base_offset = raw.storage_offset()
        byte_offset = 0
        for shape, dtype in zip(shapes, dtypes):
            dtype_size = get_dtype_size(dtype)
            page_stride = page_size_bytes // dtype_size
            target_shape = (num_blocks, *shape)
            inner_strides = torch.empty(target_shape, device="meta").stride()[1:]
            abs_offset = base_offset + byte_offset
            assert abs_offset % dtype_size == 0
            state_tensors.append(
                torch.as_strided(
                    raw.view(dtype),
                    size=target_shape,
                    stride=(page_stride, *inner_strides),
                    storage_offset=abs_offset // dtype_size,
                )
            )
            byte_offset += (
                torch.empty(target_shape, device="meta").stride()[0] * dtype_size
            )
        return state_tensors

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

    @abstractmethod
    def get_state_dtype(self) -> tuple[torch.dtype, ...]:
        pass

    def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec | None:
        mamba_block_size = vllm_config.cache_config.mamba_block_size
        assert mamba_block_size is not None
        page_size_padded = vllm_config.cache_config.mamba_page_size_padded
        spec = MambaSpec(
            shapes=tuple(self.get_state_shape()),
            dtypes=self.get_state_dtype(),
            block_size=mamba_block_size,
            page_size_padded=page_size_padded,
            mamba_type=self.mamba_type,
            mamba_cache_mode=vllm_config.cache_config.mamba_cache_mode,
            num_speculative_blocks=(
                vllm_config.speculative_config.num_speculative_tokens
                if vllm_config.speculative_config
                else 0
            ),
        )
        self._mamba_spec = spec
        return spec

    def get_attn_backend(self) -> type[AttentionBackend]:
        """Get the attention backend class for this Mamba layer."""
        return get_mamba_attn_backend(self.mamba_type)
