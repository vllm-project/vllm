# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from abc import abstractmethod
from collections.abc import Iterable
from math import lcm, prod
from typing import TYPE_CHECKING, ClassVar

import torch

from vllm.config import VllmConfig, get_layers_from_vllm_config
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.utils.torch_utils import get_dtype_size
from vllm.v1.attention.backend import AttentionBackend
from vllm.v1.attention.backends.registry import MambaAttentionBackendEnum
from vllm.v1.attention.selector import get_mamba_attn_backend
from vllm.v1.kv_cache_interface import KVCacheSpec, MambaSpec

if TYPE_CHECKING:
    from vllm.v1.worker.gpu.spec_decode.kernels import SpecKernel


class MambaBase(AttentionLayerBase):
    """Base class for Mamba-like layers which support the v1 engine.
    Inherit from this class if you implement a custom layer.
    """

    # Contains the KV cache (mamba state) for the layer
    # in the shape specified by `self.get_state_shape`.
    kv_cache: tuple[torch.Tensor, ...]
    supports_dcp: bool = False

    # What the layer is to speculative decoding beyond a chain: the variant of
    # its verify kernels ("gdn", "mamba2", "short_conv"), the key under which the
    # spec kernel registry (vllm/v1/worker/gpu/spec_decode/kernels.py) says
    # whether they take trees. Not inherited: a subclass that does not set it in
    # its own body runs kernels of its own, so it verifies chains only, whatever
    # its parent declares.
    spec_layer_variant: ClassVar[str | None] = None
    # Whether the layer verifies a chain of drafts at all: False for a layer
    # that decodes one token per request, which refuses speculative decoding.
    spec_chain_supported: ClassVar[bool] = True
    # The provider kernel that verifies the layer's spec-decode rows, bound by
    # the spec kernel registry before capture; None runs the layer's own
    # kernels. Only layers with a ``spec_layer_variant`` read it.
    spec_verify_kernel: "SpecKernel | None" = None

    def __init_subclass__(cls, **kwargs) -> None:
        super().__init_subclass__(**kwargs)
        if "spec_layer_variant" not in cls.__dict__:
            cls.spec_layer_variant = None

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
        )

    def get_attn_backend(self) -> type[AttentionBackend]:
        """Get the attention backend class for this Mamba layer."""
        return get_mamba_attn_backend(self.mamba_type)


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
