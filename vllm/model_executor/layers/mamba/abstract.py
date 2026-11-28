# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from abc import abstractmethod
from collections.abc import Callable, Iterable, Sequence
from math import lcm, prod

import torch

from vllm.config import VllmConfig, get_layers_from_vllm_config
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.utils.math_utils import round_up
from vllm.utils.torch_utils import get_dtype_size
from vllm.v1.attention.backend import AttentionBackend
from vllm.v1.attention.backends.registry import MambaAttentionBackendEnum
from vllm.v1.attention.selector import get_mamba_attn_backend
from vllm.v1.core.single_type_kv_cache_manager import MambaManager
from vllm.v1.kv_cache_interface import KVCacheLayerConfig, MambaSpec


class MambaBase(AttentionLayerBase):
    """Base class for Mamba-like layers which support the v1 engine.
    Inherit from this class if you implement a custom layer.
    """

    # Contains the KV cache (mamba state) for the layer
    # in the shape specified by `self.get_state_shape`.
    kv_cache: tuple[torch.Tensor, ...]
    supports_dcp: bool = False
    use_replayssm: bool = False
    use_flashinfer_replayssm: bool = False
    use_replayssm_trackers: bool = False
    state_alignment: int = 1
    requires_grid_checkpoint_alignment: bool = False
    _replayssm_ring_start: torch.Tensor
    _replayssm_prev_num_accepted: torch.Tensor
    replayssm_materializer: Callable[[Sequence["MambaBase"]], Callable] | None = None

    def bind_kv_cache(self, kv_cache: torch.Tensor) -> None:
        """Unpack a raw `[B, 1, 1, C]` int8 page view into per-state views.

        Each block's `C` bytes hold the layer's states (e.g. conv, ssm)
        packed by dtype/shape with the layer's state alignment.
        """
        pages = kv_cache.squeeze(dim=(1, 2))
        states: list[torch.Tensor] = []
        offset = 0
        for shape, dtype in zip(self.get_state_shape(), self.get_state_dtype()):
            nbytes = prod(shape) * get_dtype_size(dtype)
            state = pages[:, offset : offset + nbytes].view(dtype)
            states.append(state.view(-1, *shape))
            offset += round_up(nbytes, self.state_alignment)
        self.kv_cache = tuple(states)

    @abstractmethod
    def get_state_shape(self) -> Iterable[tuple[int, ...]]:
        """Defines the shape of the state.
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

    def get_cache_checkpoint_alignment(self) -> int | None:
        """Token alignment of the prefill checkpoint offsets this layer can
        export from inside a single forward pass, or `None` if it cannot.

        Serves as the single source of truth for hybrid block alignment,
        `MambaManager.Params`, and `MambaPrefillCheckpointBuilder`.
        """
        return None

    def get_kv_cache_config(self, vllm_config: VllmConfig) -> KVCacheLayerConfig | None:
        mamba_block_size = vllm_config.cache_config.mamba_block_size
        assert mamba_block_size is not None
        page_size_padded = vllm_config.cache_config.mamba_page_size_padded
        spec = MambaSpec(
            shapes=tuple(self.get_state_shape()),
            dtypes=self.get_state_dtype(),
            state_alignment=self.state_alignment,
            block_stride_alignment=self.state_alignment,
            block_size=mamba_block_size,
            page_size_padded=page_size_padded,
            mamba_type=self.mamba_type,
            tp_replicated=self.is_kv_cache_tp_replicated,
        )
        checkpoint_alignment = get_kv_cache_checkpoint_alignment(vllm_config)
        return MambaManager.layer_config(
            spec,
            enable_prefix_caching=vllm_config.cache_config.enable_prefix_caching,
            num_speculative_blocks=get_mamba_num_speculative_blocks(vllm_config),
            num_prefill_checkpoint_blocks=int(checkpoint_alignment is not None),
            prefill_checkpoint_alignment=checkpoint_alignment,
        )

    def get_attn_backend(self) -> type[AttentionBackend]:
        """Get the attention backend class for this Mamba layer."""
        return get_mamba_attn_backend(self.mamba_type)


def get_mamba_num_speculative_blocks(vllm_config: VllmConfig) -> int:
    """State blocks each request reserves for draft tokens."""
    # ReplaySSM verifies the whole window off one checkpoint, so it never
    # writes the baseline's per-draft-token state slots.
    if vllm_config.cache_config.use_replayssm:
        return 0
    return vllm_config.num_speculative_tokens


def _get_mamba_checkpoint_alignment_info(
    vllm_config: VllmConfig,
) -> tuple[int | None, bool]:
    if not vllm_config.cache_config.enable_prefix_caching:
        return None, False
    layers = get_layers_from_vllm_config(
        vllm_config,
        MambaBase,  # type: ignore[type-abstract]
    ).values()
    if not layers:
        return None, False
    alignment = 1
    requires_grid = False
    for layer in layers:
        layer_alignment = layer.get_cache_checkpoint_alignment()
        if layer_alignment is None:
            return None, False
        alignment = lcm(alignment, layer_alignment)
        requires_grid |= layer.requires_grid_checkpoint_alignment
    return alignment, requires_grid


def get_mamba_prefill_checkpoint_alignment(vllm_config: VllmConfig) -> int | None:
    """LCM of all Mamba layers' prefill checkpoint alignments.

    Returns `None` when prefix caching is disabled, when no Mamba layers are
    registered, or when any Mamba layer cannot export internal prefill
    checkpoints: the scheduler takes internal checkpoints only when every
    Mamba group can export them.
    """
    alignment, _ = _get_mamba_checkpoint_alignment_info(vllm_config)
    return alignment


def get_mamba_prefill_checkpoint_grid(
    vllm_config: VllmConfig, alignment: int | None = None
) -> int:
    """Token grid every Mamba layer keeps its prefill checkpoint offsets on.

    Prefix-cache boundaries, made multiples of the layers' checkpoint alignment
    (the platform already keeps blocks on it). The scheduler checks offsets
    against one alignment, so every Mamba spec must use this one.
    """
    if alignment is None:
        alignment = get_mamba_prefill_checkpoint_alignment(vllm_config)
    assert alignment is not None
    cache_config = vllm_config.cache_config
    if cache_config.prefix_match_unit is not None:
        return lcm(cache_config.prefix_match_unit, alignment)
    assert cache_config.mamba_block_size is not None
    return cache_config.mamba_block_size


def get_kv_cache_checkpoint_alignment(vllm_config: VllmConfig) -> int | None:
    """Model-wide prefill checkpoint alignment shared by every Mamba KV cache
    group and metadata builder, or `None` if prefix caching is disabled or any
    Mamba layer cannot export internal checkpoints.
    """
    alignment, requires_grid = _get_mamba_checkpoint_alignment_info(vllm_config)
    if alignment is None:
        return None
    if requires_grid:
        return get_mamba_prefill_checkpoint_grid(vllm_config, alignment)
    return alignment
