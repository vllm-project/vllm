# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Transformers modeling backend support for linear attention layers."""

import weakref
from typing import TYPE_CHECKING

import torch
from torch import nn

from vllm.config.utils import getattr_iter
from vllm.model_executor.layers.mamba.mamba_mixer2 import Mamba2SSM
from vllm.model_executor.layers.mamba.mamba_utils import (
    MambaStateCopyFunc,
    MambaStateCopyFuncCalculator,
    MambaStateDtypeCalculator,
    MambaStateShapeCalculator,
)
from vllm.model_executor.models.interfaces import (
    HasInnerState,
    SupportsMambaPrefixCaching,
)

if TYPE_CHECKING:
    from vllm.config import VllmConfig

VLLM_SSD_ATTR = "ssd"
"""Attribute in which to store the vLLM `Mamba2SSM` that serves a Transformers
mixer, mirroring `VLLM_ATTN_ATTR` for attention."""

LINEAR_ATTENTION_LAYER_TYPES = ("linear_attention", "hybrid")
"""`layer_types` entries whose layer contains a linear attention mixer."""


class TransformersMamba2SSM(Mamba2SSM):
    """`Mamba2SSM` that reads its weights from the Transformers mixer it serves."""

    def __init__(self, mixer: nn.Module, vllm_config: "VllmConfig", prefix: str):
        super().__init__(
            ssm_state_size=mixer.ssm_state_size,
            conv_kernel_size=mixer.conv_kernel_size,
            intermediate_size=mixer.intermediate_size,
            n_groups=mixer.n_groups,
            num_heads=mixer.num_heads,
            head_dim=mixer.head_dim,
            activation=mixer.activation,
            model_config=vllm_config.model_config,
            cache_config=vllm_config.cache_config,
            prefix=prefix,
        )
        # Weak, so the mixer does not become a submodule of its own child
        self._mixer = weakref.ref(mixer)
        A = torch.empty(mixer.num_heads, dtype=torch.float32)
        self.register_buffer("A", A, persistent=False)

    def process_weights_after_loading(self, act_dtype: torch.dtype) -> None:
        mixer = self.mixer
        self.A.copy_(-torch.exp(mixer.A_log.float()))
        # Views of the mixer's loaded weights, so they are not parameters of this
        # module but still see in-place weight updates
        weight = mixer.conv1d.weight.data
        self.conv_weights = weight.view(weight.size(0), weight.size(2))
        self.D = mixer.D.data
        self.dt_bias = mixer.dt_bias.data

    @property
    def mixer(self) -> nn.Module:
        mixer = self._mixer()
        assert mixer is not None
        return mixer

    @property
    def conv_bias(self) -> torch.Tensor | None:
        return self.mixer.conv1d.bias


class LinearAttentionMixin(HasInnerState, SupportsMambaPrefixCaching):
    """State interfaces for models whose linear attention layers are all SSD.

    The geometry is read from the config before the model exists, so the
    attribute names cover the Mamba2 family's configs.
    """

    @classmethod
    def get_mamba_state_dtype_from_config(
        cls, vllm_config: "VllmConfig"
    ) -> tuple[torch.dtype, torch.dtype]:
        return MambaStateDtypeCalculator.mamba2_state_dtype(
            vllm_config.model_config.dtype,
            vllm_config.cache_config.mamba_cache_dtype,
            vllm_config.cache_config.mamba_ssm_cache_dtype,
        )

    @classmethod
    def get_mamba_state_shape_from_config(
        cls, vllm_config: "VllmConfig"
    ) -> tuple[tuple[int, int], tuple[int, int, int]]:
        config = vllm_config.model_config.hf_text_config
        num_heads = getattr_iter(
            config, ("mamba_n_heads", "mamba_num_heads", "n_mamba_heads", "num_heads")
        )
        head_dim = getattr_iter(
            config, ("mamba_d_head", "mamba_head_dim", "mamba_headdim", "head_dim")
        )
        return MambaStateShapeCalculator.mamba2_state_shape(
            tp_world_size=vllm_config.parallel_config.tensor_parallel_size,
            intermediate_size=num_heads * head_dim,
            n_groups=getattr_iter(
                config, ("mamba_n_groups", "mamba_ngroups", "n_groups")
            ),
            num_heads=num_heads,
            head_dim=head_dim,
            state_size=getattr_iter(
                config, ("mamba_d_state", "ssm_state_size", "state_size")
            ),
            conv_kernel=getattr_iter(config, ("mamba_d_conv", "conv_kernel")),
            num_spec=vllm_config.num_speculative_tokens,
        )

    @classmethod
    def get_mamba_state_copy_func(
        cls,
    ) -> tuple[MambaStateCopyFunc, MambaStateCopyFunc]:
        return MambaStateCopyFuncCalculator.mamba2_state_copy_func()
