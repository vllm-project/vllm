# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.config.cache import MambaDType
from vllm.config.model import ModelDType
from vllm.distributed import divide
from vllm.utils.torch_utils import (
    STR_DTYPE_TO_TORCH_DTYPE,
    get_kv_cache_torch_dtype,
)


class MambaStateDtypeCalculator:
    @classmethod
    def linear_attention_state_dtype(
        cls,
        model_dtype: ModelDType | torch.dtype,
        mamba_cache_dtype: MambaDType,
    ) -> tuple[torch.dtype, ...]:
        # TODO (tdoublep) requires testing
        if mamba_cache_dtype == "float32":
            raise ValueError("fp32 state for minimax is not yet supported")
        state_dtype = get_kv_cache_torch_dtype(mamba_cache_dtype, model_dtype)
        return (state_dtype,)

    @classmethod
    def mamba1_state_dtype(
        cls,
        model_dtype: ModelDType | torch.dtype,
        mamba_cache_dtype: MambaDType,
        mamba_ssm_cache_dtype: MambaDType,
    ) -> tuple[torch.dtype, ...]:
        return cls._mamba_state_dtype(
            model_dtype, mamba_cache_dtype, mamba_ssm_cache_dtype
        )

    @classmethod
    def mamba2_state_dtype(
        cls,
        model_dtype: ModelDType | torch.dtype,
        mamba_cache_dtype: MambaDType,
        mamba_ssm_cache_dtype: MambaDType,
    ) -> tuple[torch.dtype, ...]:
        return cls._mamba_state_dtype(
            model_dtype, mamba_cache_dtype, mamba_ssm_cache_dtype
        )

    @classmethod
    def _mamba_state_dtype(
        cls,
        model_dtype: ModelDType | torch.dtype,
        mamba_cache_dtype: MambaDType,
        mamba_ssm_cache_dtype: MambaDType,
    ) -> tuple[torch.dtype, ...]:
        conv_state_dtype = get_kv_cache_torch_dtype(mamba_cache_dtype, model_dtype)
        if mamba_ssm_cache_dtype == "auto":
            temporal_state_dtype = conv_state_dtype
        else:
            temporal_state_dtype = STR_DTYPE_TO_TORCH_DTYPE[mamba_ssm_cache_dtype]

        return (conv_state_dtype, temporal_state_dtype)

    @classmethod
    def short_conv_state_dtype(
        cls,
        model_dtype: ModelDType | torch.dtype,
        mamba_cache_dtype: MambaDType,
    ) -> tuple[torch.dtype, ...]:
        conv_state_dtype = get_kv_cache_torch_dtype(mamba_cache_dtype, model_dtype)
        return (conv_state_dtype,)

    @classmethod
    def gated_delta_net_state_dtype(
        cls,
        model_dtype: ModelDType | torch.dtype,
        mamba_cache_dtype: MambaDType,
    ) -> tuple[torch.dtype, torch.dtype]:
        state_dtype = get_kv_cache_torch_dtype(mamba_cache_dtype, model_dtype)
        return (state_dtype, state_dtype)

    @classmethod
    def kda_state_dtype(
        cls,
        model_dtype: ModelDType | torch.dtype,
        mamba_cache_dtype: MambaDType,
    ):
        state_dtype = get_kv_cache_torch_dtype(mamba_cache_dtype, model_dtype)
        return (state_dtype, state_dtype, state_dtype, torch.float32)


class MambaStateShapeCalculator:
    @classmethod
    def linear_attention_state_shape(
        cls,
        num_heads: int,
        tp_size: int,
        head_dim: int,
    ) -> tuple[tuple[int, int, int], ...]:
        state_shape = (num_heads // tp_size, head_dim, head_dim)
        return (state_shape,)

    @classmethod
    def mamba1_state_shape(
        cls,
        tp_world_size: int,
        intermediate_size: int,
        state_size: int,
        conv_kernel: int,
    ) -> tuple[tuple[int, int], tuple[int, int]]:
        conv_state_shape = (divide(intermediate_size, tp_world_size), conv_kernel - 1)

        temporal_state_shape = (divide(intermediate_size, tp_world_size), state_size)

        conv_state_shape = conv_state_shape[1], conv_state_shape[0]

        return conv_state_shape, temporal_state_shape

    @classmethod
    def mamba2_state_shape(
        cls,
        tp_world_size: int,
        intermediate_size: int,
        n_groups: int,
        num_heads: int,
        head_dim: int,
        state_size: int,
        conv_kernel: int,
    ) -> tuple[tuple[int, int], tuple[int, int, int]]:
        # if n_groups is not divisible by world_size, need to extend the shards
        # to ensure all groups needed by a head is sharded along with it
        n_groups = n_groups + cls.extra_groups_for_head_shards(n_groups, tp_world_size)
        # heads and n_groups are TP-ed
        conv_dim = intermediate_size + 2 * n_groups * state_size

        # contiguous along 'dim' axis
        conv_state_shape = (conv_kernel - 1, divide(conv_dim, tp_world_size))

        # These are not TP-ed as they depend on A, dt_bias, D
        # - they are typically small
        #   e.g., (h_heads, head_dim, state_size) = (128, 64, 128)
        temporal_state_shape = (divide(num_heads, tp_world_size), head_dim, state_size)
        return conv_state_shape, temporal_state_shape

    @classmethod
    def short_conv_state_shape(
        cls,
        tp_world_size: int,
        intermediate_size: int,
        conv_kernel: int,
    ) -> tuple[tuple[int, int]]:
        conv_dim = divide(intermediate_size, tp_world_size)
        conv_state_shape = (conv_kernel - 1, conv_dim)
        return (conv_state_shape,)

    @classmethod
    def extra_groups_for_head_shards(cls, ngroups: int, tp_size: int):
        """Compute the increase in group numbers to account for
        replication in order to accompany the head shards."""

        # in the case ngoups % tp_size == 0, this will be zero
        if ngroups % tp_size == 0:
            return 0

        # for n_groups == 1, this is exactly tp_size - n_groups
        return tp_size - ngroups

    @classmethod
    def gated_delta_net_state_shape(
        cls,
        tp_world_size: int,
        num_k_heads: int,
        num_v_heads: int,
        head_k_dim: int,
        head_v_dim: int,
        conv_kernel_size: int,
        num_spec: int = 0,
    ):
        conv_dim = head_k_dim * num_k_heads * 2 + head_v_dim * num_v_heads
        conv_state_shape = (
            divide(conv_dim, tp_world_size),
            conv_kernel_size - 1 + num_spec,
        )

        conv_state_shape = conv_state_shape[1], conv_state_shape[0]

        temporal_state_shape = (
            divide(num_v_heads, tp_world_size),
            head_k_dim,
            head_v_dim,
        )
        return conv_state_shape, temporal_state_shape

    @classmethod
    def kda_state_shape(
        cls,
        tp_world_size: int,
        num_heads: int,
        head_dim: int,
        num_k_heads: int | None = None,
        head_k_dim: int | None = None,
        conv_kernel_size: int = 4,
        num_spec: int = 0,
    ) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int], tuple[int, int, int]]:
        if num_k_heads is None:
            num_k_heads = num_heads
        if head_k_dim is None:
            head_k_dim = head_dim

        proj_size = num_heads * head_dim
        proj_k_size = num_k_heads * head_k_dim

        conv_state_shape = (divide(proj_size, tp_world_size), conv_kernel_size - 1)
        conv_state_k_shape = (divide(proj_k_size, tp_world_size), conv_kernel_size - 1)
        recurrent_state_shape = (divide(num_heads, tp_world_size), head_dim, head_dim)

        conv_state_shape = conv_state_shape[1], conv_state_shape[0]
        conv_state_k_shape = conv_state_k_shape[1], conv_state_k_shape[0]
        return (
            conv_state_shape,
            conv_state_k_shape,
            conv_state_k_shape,
            recurrent_state_shape,
        )
