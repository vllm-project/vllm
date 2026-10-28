# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from abc import abstractmethod
from collections.abc import Iterable
from math import lcm, prod

import torch

from vllm.config import VllmConfig, get_layers_from_vllm_config
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.utils.torch_utils import get_dtype_size
from vllm.v1.attention.backend import AttentionBackend
from vllm.v1.attention.backends.registry import MambaAttentionBackendEnum
from vllm.v1.attention.selector import get_mamba_attn_backend
from vllm.v1.kv_cache_interface import KVCacheSpec, MambaSpec


class MambaBase(AttentionLayerBase):
    """Base class for Mamba-like layers which support the v1 engine.
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

    def get_prefill_checkpoint_alignment(self) -> int | None:
        """Token alignment of the checkpoint offsets this layer can export from
        inside a prefill, if any; in align mode, the platform makes Mamba blocks
        multiples of it.
        """
        return None

    def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec | None:
        mamba_block_size = vllm_config.cache_config.mamba_block_size
        assert mamba_block_size is not None
        page_size_padded = vllm_config.cache_config.mamba_page_size_padded
        return MambaSpec(
            shapes=tuple(self.get_state_shape()),
            dtypes=self.get_state_dtype(),
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
            fixed_state_slots=use_mamba_fixed_state_slots(vllm_config),
        )

    def get_attn_backend(self) -> type[AttentionBackend]:
        """Get the attention backend class for this Mamba layer."""
        return get_mamba_attn_backend(self.mamba_type)


def use_mamba_fixed_state_slots(vllm_config: VllmConfig) -> bool:
    """Whether "align" prefix caching keeps Mamba state in fixed slots.

    Only Model Runner V2 implements the layout, for generative models. KV
    connectors, pipeline and context parallelism, and ReplaySSM/RecoverSSM
    still use one block-table column per Mamba block.
    """
    cache_config = vllm_config.cache_config
    parallel_config = vllm_config.parallel_config
    return (
        cache_config.mamba_cache_mode == "align"
        and vllm_config.use_v2_model_runner
        and vllm_config.model_config.runner_type == "generate"
        and vllm_config.kv_transfer_config is None
        and parallel_config.pipeline_parallel_size == 1
        and parallel_config.decode_context_parallel_size == 1
        and parallel_config.prefill_context_parallel_size == 1
        and not cache_config.use_replayssm
        and not cache_config.use_kda_recoverssm
    )


def get_mamba_prefill_checkpoint_alignment(vllm_config: VllmConfig) -> int | None:
    """LCM of the Mamba layers' prefill checkpoint alignments.

    None unless every Mamba layer declares one: the scheduler takes internal
    checkpoints only when every Mamba group can export them.
    """
    alignment = 1
    for layer in get_layers_from_vllm_config(
        vllm_config,
        MambaBase,  # type: ignore[type-abstract]
    ).values():
        layer_alignment = layer.get_prefill_checkpoint_alignment()
        if layer_alignment is None:
            return None
        alignment = lcm(alignment, layer_alignment)
    return alignment


def get_mamba_prefill_checkpoint_grid(vllm_config: VllmConfig) -> int:
    """Token grid every Mamba layer keeps its prefill checkpoint offsets on.

    Prefix-cache boundaries, made multiples of the layers' checkpoint alignment
    (the platform already keeps blocks on it). The scheduler checks offsets
    against one alignment, so every Mamba spec must use this one.
    """
    alignment = get_mamba_prefill_checkpoint_alignment(vllm_config)
    assert alignment is not None
    cache_config = vllm_config.cache_config
    if cache_config.prefix_match_unit is not None:
        return lcm(cache_config.prefix_match_unit, alignment)
    assert cache_config.mamba_block_size is not None
    return cache_config.mamba_block_size
