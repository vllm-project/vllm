# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import functools
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal, TypeAlias

import torch

import vllm.envs as envs
from vllm.config.cache import MambaDType
from vllm.config.model import ModelDType
from vllm.distributed import divide
from vllm.logger import init_logger
from vllm.utils.torch_utils import (
    STR_DTYPE_TO_TORCH_DTYPE,
    get_kv_cache_torch_dtype,
)

logger = init_logger(__name__)

ConvStateLayoutType = Literal["SD", "DS"]


@functools.lru_cache
def get_conv_state_layout() -> ConvStateLayoutType:
    """Return the SSM conv state layout.

    SD = (state_len, dim) — dim is the innermost contiguous dimension.
    DS = (dim, state_len) — TP-sharded dim is on dim-1 (like HND for KV
         cache), consistent with SSM temporal state layout.
    """
    layout: ConvStateLayoutType | None = envs.VLLM_SSM_CONV_STATE_LAYOUT
    if layout is not None:
        logger.info_once(
            "VLLM_SSM_CONV_STATE_LAYOUT env detected. "
            "Setting SSM conv state layout to %s.",
            layout,
        )
        return layout

    return "SD"


def is_conv_state_dim_first() -> bool:
    """True when the conv state is stored as (dim, state_len) per block."""
    return get_conv_state_layout() == "DS"


class MambaStateDtypeCalculator:
    @classmethod
    def linear_attention_state_dtype(
        cls,
        model_dtype: ModelDType | torch.dtype,
        mamba_cache_dtype: MambaDType,
    ) -> tuple[torch.dtype, ...]:
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
    def mamba2_cached_state_dtype(
        cls,
        model_dtype: ModelDType | torch.dtype,
        mamba_cache_dtype: MambaDType,
        mamba_ssm_cache_dtype: MambaDType,
        use_replayssm: bool,
    ) -> tuple[torch.dtype, ...]:
        """Mamba2 state dtypes, extended for the state-and-output decode kernel.

        Returns the baseline ``(conv, ssm)`` dtypes when
        ``use_replayssm`` is ``False``; otherwise appends the
        state-and-output ring-buffer dtypes ``(x_cache, dt_cache, B_cache)`` =
        ``(activation, fp32, activation)``. Must stay in sync with
        ``MambaMixer2.get_state_dtype``.
        """
        conv_dtype, ssm_dtype = cls.mamba2_state_dtype(
            model_dtype, mamba_cache_dtype, mamba_ssm_cache_dtype
        )
        if not use_replayssm:
            return conv_dtype, ssm_dtype
        activation_dtype = get_kv_cache_torch_dtype("auto", model_dtype)
        return conv_dtype, ssm_dtype, activation_dtype, torch.float32, activation_dtype

    @classmethod
    def mamba2_spec_cached_state_dtype(
        cls,
        model_dtype: ModelDType | torch.dtype,
        mamba_cache_dtype: MambaDType,
        mamba_ssm_cache_dtype: MambaDType,
        use_replayssm_spec: bool,
    ) -> tuple[torch.dtype, ...]:
        """Mamba2 state dtypes for the cached SPECULATIVE-decode (hybrid) kernel.

        Baseline ``(conv, ssm)`` when off; otherwise the hybrid 4-tuple
        ``(conv, ssm_checkpoint, post_conv_cache, dt_cache)``. The checkpoint
        and ``dt_cache`` are forced fp32 (the cached-spec reconstruction was
        validated against an fp32 reference); ``post_conv_cache`` is activation
        dtype. Must stay in sync with ``MambaMixer2.get_state_dtype``.
        """
        conv_dtype, _ssm_dtype = cls.mamba2_state_dtype(
            model_dtype, mamba_cache_dtype, mamba_ssm_cache_dtype
        )
        if not use_replayssm_spec:
            return conv_dtype, _ssm_dtype
        activation_dtype = get_kv_cache_torch_dtype("auto", model_dtype)
        return conv_dtype, torch.float32, activation_dtype, torch.float32

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
        mamba_ssm_cache_dtype: MambaDType = "auto",
    ) -> tuple[torch.dtype, torch.dtype]:
        return cls._mamba_state_dtype(
            model_dtype, mamba_cache_dtype, mamba_ssm_cache_dtype
        )

    @classmethod
    def gated_delta_net_cached_state_dtype(
        cls,
        model_dtype: ModelDType | torch.dtype,
        mamba_cache_dtype: MambaDType,
        mamba_ssm_cache_dtype: MambaDType,
        use_replayssm: bool,
    ) -> tuple[torch.dtype, ...]:
        """GDN state dtypes, extended for the cached decode kernel.

        Returns the baseline ``(conv, ssm)`` dtypes when
        ``use_replayssm`` is ``False``; otherwise appends the ring
        cache dtypes ``(d_cache, k_cache, g_cache)`` =
        ``(activation, activation, float32)``.
        """
        conv_dtype, ssm_dtype = cls._mamba_state_dtype(
            model_dtype, mamba_cache_dtype, mamba_ssm_cache_dtype
        )
        if not use_replayssm:
            return conv_dtype, ssm_dtype
        activation_dtype = get_kv_cache_torch_dtype("auto", model_dtype)
        return conv_dtype, ssm_dtype, activation_dtype, activation_dtype, torch.float32

    @classmethod
    def gated_delta_net_spec_cached_state_dtype(
        cls,
        model_dtype: ModelDType | torch.dtype,
        mamba_cache_dtype: MambaDType,
        mamba_ssm_cache_dtype: MambaDType,
        use_replayssm_spec: bool,
    ) -> tuple[torch.dtype, ...]:
        """GDN state dtypes for the cached SPECULATIVE-decode kernel.

        Same ``d/k/g`` ring page as the non-spec cached path, but the ``ssm``
        checkpoint is forced to ``float32`` Returns the baseline ``(conv, ssm)`` when the flag is off.
        """
        conv_dtype, ssm_dtype = cls._mamba_state_dtype(
            model_dtype, mamba_cache_dtype, mamba_ssm_cache_dtype
        )
        if not use_replayssm_spec:
            return conv_dtype, ssm_dtype
        activation_dtype = get_kv_cache_torch_dtype("auto", model_dtype)
        return (
            conv_dtype,
            torch.float32,  # fp32 checkpoint
            activation_dtype,  # d_cache
            activation_dtype,  # k_cache
            torch.float32,  # g_cache
        )

    @classmethod
    def kda_state_dtype(
        cls,
        model_dtype: ModelDType | torch.dtype,
        mamba_cache_dtype: MambaDType,
    ) -> tuple[torch.dtype, torch.dtype]:
        state_dtype = get_kv_cache_torch_dtype(mamba_cache_dtype, model_dtype)
        return (state_dtype, torch.float32)


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

    @staticmethod
    def _orient_conv_shape(dim: int, state_len: int) -> tuple[int, int]:
        """Return (dim, state_len) for DS layout, (state_len, dim) for SD."""
        if is_conv_state_dim_first():
            return (dim, state_len)
        return (state_len, dim)

    @classmethod
    def mamba1_state_shape(
        cls,
        tp_world_size: int,
        intermediate_size: int,
        state_size: int,
        conv_kernel: int,
    ) -> tuple[tuple[int, int], tuple[int, int]]:
        conv_dim = divide(intermediate_size, tp_world_size)
        conv_state_shape = cls._orient_conv_shape(conv_dim, conv_kernel - 1)

        temporal_state_shape = (divide(intermediate_size, tp_world_size), state_size)

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
        num_spec: int = 0,
    ) -> tuple[tuple[int, int], tuple[int, int, int]]:
        # if n_groups is not divisible by world_size, need to extend the shards
        # to ensure all groups needed by a head is sharded along with it
        n_groups = n_groups + cls.extra_groups_for_head_shards(n_groups, tp_world_size)
        # heads and n_groups are TP-ed
        conv_dim = intermediate_size + 2 * n_groups * state_size

        conv_state_shape = cls._orient_conv_shape(
            divide(conv_dim, tp_world_size), conv_kernel - 1 + num_spec
        )

        # These are not TP-ed as they depend on A, dt_bias, D
        # - they are typically small
        #   e.g., (h_heads, head_dim, state_size) = (128, 64, 128)
        temporal_state_shape = (divide(num_heads, tp_world_size), head_dim, state_size)
        return conv_state_shape, temporal_state_shape

    @classmethod
    def mamba2_cached_state_shape(
        cls,
        tp_world_size: int,
        intermediate_size: int,
        n_groups: int,
        num_heads: int,
        head_dim: int,
        state_size: int,
        conv_kernel: int,
        use_replayssm: bool,
        replayssm_buffer_len: int,
        num_spec: int = 0,
    ) -> tuple[tuple[int, ...], ...]:
        """Mamba2 state shapes, extended for the state-and-output decode kernel.

        Returns the baseline ``(conv, ssm)`` shapes when
        ``use_replayssm`` is ``False``; otherwise appends the
        state-and-output ring-buffer shapes ``x_cache``/``dt_cache``/``B_cache``.
        Group/head counts use the (un-extended) ``n_groups``/``num_heads``
        divided by ``tp_world_size``, matching ``MambaMixer2.get_state_shape``.
        """
        conv_state_shape, temporal_state_shape = cls.mamba2_state_shape(
            tp_world_size=tp_world_size,
            intermediate_size=intermediate_size,
            n_groups=n_groups,
            num_heads=num_heads,
            head_dim=head_dim,
            state_size=state_size,
            conv_kernel=conv_kernel,
            num_spec=num_spec,
        )
        if not use_replayssm:
            return conv_state_shape, temporal_state_shape

        local_nheads = divide(num_heads, tp_world_size)
        local_ngroups = divide(n_groups, tp_world_size)
        x_cache_shape = (local_nheads, replayssm_buffer_len, head_dim)
        dt_cache_shape = (local_nheads, replayssm_buffer_len)
        B_cache_shape = (local_ngroups, replayssm_buffer_len, state_size)
        return (
            conv_state_shape,
            temporal_state_shape,
            x_cache_shape,
            dt_cache_shape,
            B_cache_shape,
        )

    @classmethod
    def mamba2_spec_cached_state_shape(
        cls,
        tp_world_size: int,
        intermediate_size: int,
        n_groups: int,
        num_heads: int,
        head_dim: int,
        state_size: int,
        conv_kernel: int,
        use_replayssm_spec: bool,
        replayssm_buffer_len: int,
        num_spec: int = 0,
    ) -> tuple[tuple[int, ...], ...]:
        """Mamba2 state shapes for the cached SPECULATIVE-decode (hybrid) kernel.

        Baseline ``(conv, ssm)`` when off (conv keeps its spec sliding-window
        size ``conv_kernel-1+num_spec`` -- the hybrid reuses
        ``causal_conv1d_update``); otherwise appends the circular caches
        ``post_conv_cache=(cache_buf_len, conv_dim_local)`` and
        ``dt_cache=(local_nheads, cache_buf_len)``, where the L = B + max_spec_len
        history window sizes ``cache_buf_len = next_pow2(replayssm_buffer_len + 1 +
        num_spec)`` and ``conv_dim_local`` matches the post-conv x|B width (C is
        not cached; read fresh from conv_out). Must stay in sync with
        ``MambaMixer2.get_state_shape``.
        """
        conv_state_shape, temporal_state_shape = cls.mamba2_state_shape(
            tp_world_size=tp_world_size,
            intermediate_size=intermediate_size,
            n_groups=n_groups,
            num_heads=num_heads,
            head_dim=head_dim,
            state_size=state_size,
            conv_kernel=conv_kernel,
            num_spec=num_spec,
        )
        if not use_replayssm_spec:
            return conv_state_shape, temporal_state_shape
        n_groups_ext = n_groups + cls.extra_groups_for_head_shards(
            n_groups, tp_world_size
        )
        conv_dim_local = divide(
            intermediate_size + n_groups_ext * state_size, tp_world_size
        )
        # L = B + max_spec_len history window: physical pow2 buffer next_pow2(L).
        cache_buf_len = 1 << (replayssm_buffer_len + num_spec).bit_length()
        local_nheads = divide(num_heads, tp_world_size)
        post_conv_cache_shape = (cache_buf_len, conv_dim_local)
        dt_cache_shape = (local_nheads, cache_buf_len)
        return (
            conv_state_shape,
            temporal_state_shape,
            post_conv_cache_shape,
            dt_cache_shape,
        )

    @classmethod
    def short_conv_state_shape(
        cls,
        tp_world_size: int,
        intermediate_size: int,
        conv_kernel: int,
    ) -> tuple[tuple[int, int]]:
        conv_dim = divide(intermediate_size, tp_world_size)
        conv_state_shape = cls._orient_conv_shape(conv_dim, conv_kernel - 1)
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
        conv_state_shape = cls._orient_conv_shape(
            divide(conv_dim, tp_world_size),
            conv_kernel_size - 1 + num_spec,
        )

        temporal_state_shape = (
            divide(num_v_heads, tp_world_size),
            head_v_dim,
            head_k_dim,
        )
        return conv_state_shape, temporal_state_shape

    @classmethod
    def gated_delta_net_cached_state_shape(
        cls,
        tp_world_size: int,
        num_k_heads: int,
        num_v_heads: int,
        head_k_dim: int,
        head_v_dim: int,
        conv_kernel_size: int,
        use_replayssm: bool,
        replayssm_buffer_len: int,
        num_spec: int = 0,
    ) -> tuple[tuple[int, ...], ...]:
        """GDN state shapes, extended for the cached decode kernel.

        Returns the baseline ``(conv, ssm)`` shapes when
        ``use_replayssm`` is ``False``; otherwise appends the cached
        ring-buffer shapes ``d_cache``/``k_cache``/``g_cache``. Head counts use
        the (un-extended) ``num_v_heads``/``num_k_heads`` divided by
        ``tp_world_size``, matching ``gated_delta_net_state_shape``.
        """
        conv_state_shape, temporal_state_shape = cls.gated_delta_net_state_shape(
            tp_world_size,
            num_k_heads,
            num_v_heads,
            head_k_dim,
            head_v_dim,
            conv_kernel_size,
            num_spec,
        )
        if not use_replayssm:
            return conv_state_shape, temporal_state_shape

        local_v_heads = divide(num_v_heads, tp_world_size)
        local_k_heads = divide(num_k_heads, tp_world_size)
        d_cache_shape = (local_v_heads, replayssm_buffer_len, head_v_dim)
        k_cache_shape = (local_k_heads, replayssm_buffer_len, head_k_dim)
        g_cache_shape = (local_v_heads, replayssm_buffer_len)
        return (
            conv_state_shape,
            temporal_state_shape,
            d_cache_shape,
            k_cache_shape,
            g_cache_shape,
        )

    @classmethod
    def gated_delta_net_spec_cached_state_shape(
        cls,
        tp_world_size: int,
        num_k_heads: int,
        num_v_heads: int,
        head_k_dim: int,
        head_v_dim: int,
        conv_kernel_size: int,
        use_replayssm_spec: bool,
        replayssm_buffer_len: int,
        num_spec: int = 0,
    ) -> tuple[tuple[int, ...], ...]:
        """GDN state shapes for the cached SPECULATIVE-decode kernel.

        The circular ``d_cache``/``k_cache``/``g_cache`` use the L = B + max_spec_len
        history window: a power-of-two buffer ``next_pow2(replayssm_buffer_len + 1 +
        num_spec)``. Returns the baseline ``(conv, ssm)`` shapes when the flag is
        off. The block-keyed cursors live in the GDN metadata builder, not the page.
        """
        conv_state_shape, temporal_state_shape = cls.gated_delta_net_state_shape(
            tp_world_size,
            num_k_heads,
            num_v_heads,
            head_k_dim,
            head_v_dim,
            conv_kernel_size,
            num_spec,
        )
        if not use_replayssm_spec:
            return conv_state_shape, temporal_state_shape

        cache_buf_len = 1 << (replayssm_buffer_len + num_spec).bit_length()
        local_v_heads = divide(num_v_heads, tp_world_size)
        local_k_heads = divide(num_k_heads, tp_world_size)
        d_cache_shape = (local_v_heads, cache_buf_len, head_v_dim)
        k_cache_shape = (local_k_heads, cache_buf_len, head_k_dim)
        g_cache_shape = (local_v_heads, cache_buf_len)
        return (
            conv_state_shape,
            temporal_state_shape,
            d_cache_shape,
            k_cache_shape,
            g_cache_shape,
        )

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
    ) -> tuple[tuple[int, int], tuple[int, int, int]]:
        if num_k_heads is None:
            num_k_heads = num_heads
        if head_k_dim is None:
            head_k_dim = head_dim

        proj_size = num_heads * head_dim
        proj_k_size = num_k_heads * head_k_dim

        conv_dim = proj_size + 2 * proj_k_size
        conv_state_shape = cls._orient_conv_shape(
            divide(conv_dim, tp_world_size), conv_kernel_size - 1
        )
        recurrent_state_shape = (divide(num_heads, tp_world_size), head_dim, head_dim)
        return (conv_state_shape, recurrent_state_shape)


@dataclass
class MambaCopySpec:
    """
    Data class specifying the memory-copy parameters for Mamba states used for
    prefix caching in align mode.

    Attributes:
        start_addr (int): Starting address for the memory copy operation.
        num_elements (int): Number of elements to copy from the starting address.
    """

    start_addr: int
    num_elements: int


MambaStateCopyFunc: TypeAlias = Callable[
    [torch.Tensor, list[int], int, int], MambaCopySpec
]
"""
Type alias for a function that computes a MambaCopySpec for copying state slices.
Parameters:
  state: torch.Tensor - the Mamba state tensor (e.g., conv or temporal states).
  block_ids: list[int] - the list of block indices for the state to copy.
  cur_block_idx: int - current block index within `block_ids` to copy from.
  num_accepted_tokens: int - number of accepted tokens used to compute the copy offset.
      Range: 1 .. 1 + num_speculative_tokens (inclusive).
"""


def get_conv_copy_spec(
    state: torch.Tensor,
    block_ids: list[int],
    cur_block_idx: int,
    num_accepted_tokens: int,
) -> MambaCopySpec:
    """Return a MambaCopySpec for copying a convolutional state slice.

    Works for both SD layout ``(num_blocks, state_len, dim)`` and
    DS layout ``(num_blocks, dim, state_len)``.
    """
    src_block_id = block_ids[cur_block_idx]
    offset = num_accepted_tokens - 1
    if is_conv_state_dim_first():
        # DS offset > 0 is handled by the fused postprocess kernel.
        assert offset == 0, (
            "DS conv state with num_accepted_tokens > 1 must be handled by "
            "the fused postprocess kernel, not get_conv_copy_spec"
        )
        src_state = state[src_block_id]
    else:
        # SD layout: (num_blocks, state_len, dim), with dim contiguous.
        src_state = state[src_block_id, offset:]
    return MambaCopySpec(
        start_addr=src_state.data_ptr(), num_elements=src_state.numel()
    )


def get_temporal_copy_spec(
    state: torch.Tensor,
    block_ids: list[int],
    cur_block_idx: int,
    num_accepted_tokens: int,
) -> MambaCopySpec:
    """Return a MambaCopySpec for copying a temporal state slice."""
    src_block_id = block_ids[cur_block_idx + num_accepted_tokens - 1]
    src_state = state[src_block_id]
    return MambaCopySpec(
        start_addr=src_state.data_ptr(), num_elements=src_state.numel()
    )


class MambaStateCopyFuncCalculator:
    @classmethod
    def linear_attention_state_copy_func(cls):
        return (get_temporal_copy_spec,)

    @classmethod
    def mamba1_state_copy_func(cls):
        return (get_conv_copy_spec, get_temporal_copy_spec)

    @classmethod
    def mamba2_state_copy_func(cls):
        return get_conv_copy_spec, get_temporal_copy_spec

    @classmethod
    def short_conv_state_copy_func(cls):
        return (get_conv_copy_spec,)

    @classmethod
    def gated_delta_net_state_copy_func(cls):
        return (get_conv_copy_spec, get_temporal_copy_spec)

    @classmethod
    def kda_state_copy_func(cls):
        return (get_conv_copy_spec, get_temporal_copy_spec)
