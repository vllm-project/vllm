# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import functools
import os
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
    def mamba2_replayssm_state_dtype(
        cls,
        model_dtype: ModelDType | torch.dtype,
        mamba_cache_dtype: MambaDType,
        mamba_ssm_cache_dtype: MambaDType,
    ) -> tuple[torch.dtype, ...]:
        """Mamba2 ReplaySSM state dtypes: baseline ``(conv, ssm)`` plus the
        ring-buffer dtypes ``(x_cache, dt_cache, B_cache)`` =
        ``(activation, fp32, activation)``. Call only when use_replayssm is on;
        must stay in sync with ``MambaMixer2.get_state_dtype``.
        """
        conv_dtype, ssm_dtype = cls.mamba2_state_dtype(
            model_dtype, mamba_cache_dtype, mamba_ssm_cache_dtype
        )
        activation_dtype = get_kv_cache_torch_dtype("auto", model_dtype)
        return conv_dtype, ssm_dtype, activation_dtype, torch.float32, activation_dtype

    @classmethod
    def mamba2_replayssm_spec_state_dtype(
        cls,
        model_dtype: ModelDType | torch.dtype,
        mamba_cache_dtype: MambaDType,
        mamba_ssm_cache_dtype: MambaDType,
    ) -> tuple[torch.dtype, ...]:
        """Mamba2 ReplaySSM state dtypes for the SPECULATIVE-decode (hybrid)
        kernel: the hybrid 4-tuple
        ``(conv, ssm_checkpoint, post_conv_cache, dt_cache)``. The checkpoint
        and ``dt_cache`` are forced fp32 (the cached-spec reconstruction was
        validated against an fp32 reference); ``post_conv_cache`` is activation
        dtype. Call only when use_replayssm_spec is on; must stay in sync with
        ``MambaMixer2.get_state_dtype``.
        """
        conv_dtype, _ssm_dtype = cls.mamba2_state_dtype(
            model_dtype, mamba_cache_dtype, mamba_ssm_cache_dtype
        )
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
    def gated_delta_net_replayssm_state_dtype(
        cls,
        model_dtype: ModelDType | torch.dtype,
        mamba_cache_dtype: MambaDType,
        mamba_ssm_cache_dtype: MambaDType,
    ) -> tuple[torch.dtype, ...]:
        """GDN ReplaySSM state dtypes: baseline ``(conv, ssm)`` plus the ring
        cache dtypes ``(d_cache, k_cache, g_cache)``. The ``d``/``k`` input
        caches use fp16 for bf16 activations; ``g_cache`` is float32. Call only
        when use_replayssm is on.
        """
        conv_dtype, ssm_dtype = cls._mamba_state_dtype(
            model_dtype, mamba_cache_dtype, mamba_ssm_cache_dtype
        )
        activation_dtype = get_kv_cache_torch_dtype("auto", model_dtype)
        cache_dtype = (
            torch.float16 if activation_dtype == torch.bfloat16 else activation_dtype
        )
        return conv_dtype, ssm_dtype, cache_dtype, cache_dtype, torch.float32

    @classmethod
    def gated_delta_net_replayssm_spec_state_dtype(
        cls,
        model_dtype: ModelDType | torch.dtype,
        mamba_cache_dtype: MambaDType,
        mamba_ssm_cache_dtype: MambaDType,
        vllm_config=None,
    ) -> tuple[torch.dtype, ...]:
        """GDN ReplaySSM state dtypes for the SPECULATIVE-decode kernel.

        The ``ssm`` checkpoint is forced to ``float32`` unless an explicit
        ``mamba_ssm_cache_dtype`` overrides it (``auto`` keeps the upstream
        fp32 default). The ``d``/``k`` ring caches use fp16 for bf16
        activations (same rule as the non-spec path) — except on the
        flashinfer_ucache spec backend, whose rings must match the CuTeDSL
        kernel's compiled IO dtype (bf16 default / fp16 with
        GDN_UCACHE_IO_DTYPE=fp16); pass ``vllm_config`` so the backend can be
        resolved. Call only when use_replayssm_spec is on.
        """
        conv_dtype, ssm_dtype = cls._mamba_state_dtype(
            model_dtype, mamba_cache_dtype, mamba_ssm_cache_dtype
        )
        activation_dtype = get_kv_cache_torch_dtype("auto", model_dtype)
        cache_dtype = (
            torch.float16 if activation_dtype == torch.bfloat16 else activation_dtype
        )
        # Explicit --mamba-ssm-cache-dtype overrides the ckpt dtype; "auto"
        # keeps the upstream force-fp32 default (_mamba_state_dtype would
        # map "auto" to the model dtype, NOT fp32). On the flashinfer_ucache
        # backend "auto" instead resolves to the kernel's state dtype
        # (default fp16) below.
        ckpt_dtype = torch.float32 if mamba_ssm_cache_dtype == "auto" else ssm_dtype
        # Backend-aware ring default: the ucache kernel reads the u/k rings at
        # its compiled IO dtype, so allocation must match it (the fp16-ring
        # upstream rule only applies to the dtype-agnostic Triton kernel).
        default_ring_dtype = cache_dtype
        if vllm_config is not None:
            try:
                backend = resolve_gdn_spec_backend(vllm_config)
            except Exception:
                backend = "triton"
            if backend == "flashinfer_ucache":
                # ucache defaults: fp16 SSM-state checkpoint + fp16 u/k rings
                # with bf16 input IO. The adapter setdefaults the kernel
                # module's GDN_UCACHE_STATE/RING_DTYPE envs to the same
                # values, so pool allocation and the compiled kernel dtypes
                # agree with no flags. Set the envs to bf16 for bf16 mode.
                _st_env = os.environ.get(
                    "GDN_UCACHE_STATE_DTYPE", "fp16"
                ).lower()
                if mamba_ssm_cache_dtype == "auto":
                    ckpt_dtype = (
                        torch.bfloat16
                        if _st_env in ("bf16", "bfloat16")
                        else torch.float16
                    )
                _ring_env_kernel = os.environ.get(
                    "GDN_UCACHE_RING_DTYPE", "fp16"
                ).lower()
                default_ring_dtype = (
                    torch.bfloat16
                    if _ring_env_kernel in ("bf16", "bfloat16")
                    else torch.float16
                )
        # Ring-dtype override (default: see above):
        #   VLLM_REPLAYSSM_RING_DTYPE=fp16 -> fp16 u/k rings (ucache whole-fp16
        #     mode: pair with GDN_UCACHE_IO_DTYPE=fp16 so the cute kernel
        #     compiles fp16 IO; the adapter casts activations at the call).
        #   VLLM_REPLAYSSM_RING_DTYPE=bf16|fp32 -> bf16 / fp32 u/k rings.
        _ring_env = os.environ.get("VLLM_REPLAYSSM_RING_DTYPE", "").lower()
        if _ring_env in ("fp16", "float16", "half"):
            ring_dtype = torch.float16
        elif _ring_env in ("bf16", "bfloat16"):
            ring_dtype = torch.bfloat16
        elif _ring_env in ("fp32", "float32"):
            ring_dtype = torch.float32
        else:
            ring_dtype = default_ring_dtype
        return (
            conv_dtype,
            ckpt_dtype,
            ring_dtype,  # d_cache
            ring_dtype,  # k_cache
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
    def mamba2_replayssm_state_shape(
        cls,
        tp_world_size: int,
        intermediate_size: int,
        n_groups: int,
        num_heads: int,
        head_dim: int,
        state_size: int,
        conv_kernel: int,
        replayssm_buffer_len: int,
        num_spec: int = 0,
    ) -> tuple[tuple[int, ...], ...]:
        """Mamba2 ReplaySSM state shapes: baseline ``(conv, ssm)`` plus the
        ring-buffer shapes ``x_cache``/``dt_cache``/``B_cache``. Delegates to
        ``mamba2_state_shape`` for ``(conv, ssm)`` so the ring buffers keep the
        un-extended ``n_groups`` (that method extends n_groups only in its own
        scope). Call only when use_replayssm is on; must stay in sync with
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
    def mamba2_replayssm_spec_state_shape(
        cls,
        tp_world_size: int,
        intermediate_size: int,
        n_groups: int,
        num_heads: int,
        head_dim: int,
        state_size: int,
        conv_kernel: int,
        replayssm_buffer_len: int,
        num_spec: int = 0,
    ) -> tuple[tuple[int, ...], ...]:
        """Mamba2 ReplaySSM state shapes for the SPECULATIVE-decode (hybrid)
        kernel: baseline ``(conv, ssm)`` (conv keeps its spec sliding-window
        size ``conv_kernel-1+num_spec`` -- the hybrid reuses
        ``causal_conv1d_update``) plus the circular caches
        ``post_conv_cache=(cache_buf_len, conv_dim_local)`` and
        ``dt_cache=(local_nheads, cache_buf_len)``, where the L = B + max_spec_len
        history window sizes ``cache_buf_len = next_pow2(replayssm_buffer_len + 1 +
        num_spec)`` and ``conv_dim_local`` matches the post-conv x|B width (C is
        not cached; read fresh from conv_out). Call only when use_replayssm_spec
        is on; must stay in sync with ``MambaMixer2.get_state_shape``.
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
    def gated_delta_net_replayssm_state_shape(
        cls,
        tp_world_size: int,
        num_k_heads: int,
        num_v_heads: int,
        head_k_dim: int,
        head_v_dim: int,
        conv_kernel_size: int,
        replayssm_buffer_len: int,
        num_spec: int = 0,
    ) -> tuple[tuple[int, ...], ...]:
        """GDN ReplaySSM state shapes: baseline ``(conv, ssm)`` plus the cached
        ring-buffer shapes ``d_cache``/``k_cache``/``g_cache``. Head counts use
        the (un-extended) ``num_v_heads``/``num_k_heads`` divided by
        ``tp_world_size``, matching ``gated_delta_net_state_shape``. Call only
        when use_replayssm is on.
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
    def gated_delta_net_replayssm_spec_state_shape(
        cls,
        tp_world_size: int,
        num_k_heads: int,
        num_v_heads: int,
        head_k_dim: int,
        head_v_dim: int,
        conv_kernel_size: int,
        replayssm_buffer_len: int,
        num_spec: int = 0,
        ring_slots: int | None = None,
    ) -> tuple[tuple[int, ...], ...]:
        """GDN ReplaySSM state shapes for the SPECULATIVE-decode kernel.

        The circular ``d_cache``/``k_cache``/``g_cache`` use the L = B + max_spec_len
        history window: a power-of-two buffer ``next_pow2(replayssm_buffer_len + 1 +
        num_spec)``. Call only when use_replayssm_spec is on. The block-keyed
        cursors live in the GDN metadata builder, not the page.

        ``ring_slots`` overrides the physical ring depth for backends with a
        fixed linear (non-circular) ring — e.g. the flashinfer_ucache kernel
        uses exactly W_RING=16 slots (page[2] becomes its u_cache, page[3] its
        k_cache, page[4] its g_cache; dtypes unchanged).
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
        cache_buf_len = (
            ring_slots
            if ring_slots is not None
            else 1 << (replayssm_buffer_len + num_spec).bit_length()
        )
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


def _ucache_kernel_available() -> tuple[bool, str]:
    """Init-time check that the ucache CuTeDSL kernel module is loadable
    AND is a ring build (RING_SLOTS == 32).

    Mirrors load_ucache_kernel_module's resolution order without importing
    (and JIT-compiling) the module: an explicit VLLM_GDN_UCACHE_MODULE path
    must exist, else flashinfer.gdn_kernels must provide the module. The
    ring check scans the module SOURCE for the RING_SLOTS constant so that
    a pre-ring kernel fails here, at engine init, rather than at the first
    speculative-decode step mid-serving (load_ucache_kernel_module re-checks
    the imported module authoritatively on first use)."""
    import importlib.util

    path = os.environ.get("VLLM_GDN_UCACHE_MODULE")
    if path:
        if not os.path.isfile(path):
            return (
                False,
                f"VLLM_GDN_UCACHE_MODULE points to a missing file: {path!r}",
            )
        src_path = path
    else:
        try:
            spec = importlib.util.find_spec(
                "flashinfer.gdn_kernels.gdn_decode_bf16_wy_ucache_flush"
            )
        except ModuleNotFoundError:
            spec = None
        if spec is None:
            return (
                False,
                "ucache CuTeDSL kernel module not found: set "
                "VLLM_GDN_UCACHE_MODULE=/abs/path/to/"
                "gdn_decode_bf16_wy_ucache_flush.py or install a FlashInfer "
                "build that provides "
                "flashinfer.gdn_kernels.gdn_decode_bf16_wy_ucache_flush",
            )
        src_path = spec.origin or ""

    if src_path and os.path.isfile(src_path):
        import re

        try:
            with open(src_path, encoding="utf-8", errors="replace") as f:
                src = f.read()
        except OSError:
            src = ""
        if src:
            m = re.search(
                r"^RING_SLOTS(?:\s*:\s*\w+)?\s*=\s*(\d+)", src, re.MULTILINE
            )
            ring_slots = int(m.group(1)) if m else None
            if ring_slots != 32:
                return (
                    False,
                    f"kernel module at {src_path!r} is not a ring build "
                    f"(RING_SLOTS={ring_slots}, need 32): pre-ring ucache "
                    "kernels are incompatible with this backend's "
                    "Triton-ring cursor model — update the FlashInfer "
                    "ucache kernel",
                )
    return (True, "")


def resolve_gdn_spec_backend(vllm_config) -> str:
    """Resolve the GDN cached-SPEC decode backend.

    Returns "triton" (default, PR #47576 gdn_replayssm_spec_decode) or
    "flashinfer_ucache" (CuTeDSL gated_delta_rule_mtp_ucache_flush). Selected
    via additional_config["gdn_spec_backend"]; constraint violations raise at
    init (fail loudly rather than silently falling back to triton).
    """
    additional_config = vllm_config.additional_config
    requested = (
        str(additional_config.get("gdn_spec_backend", "triton")).strip().lower()
        if isinstance(additional_config, dict)
        else "triton"
    )
    if requested in ("triton", "auto", ""):
        return "triton"
    if requested != "flashinfer_ucache":
        raise ValueError(f"unknown gdn_spec_backend={requested!r}")

    from vllm.model_executor.layers.fla.ops.gdn_ucache_spec import UCACHE_W_RING
    from vllm.platforms import current_platform

    cache_config = vllm_config.cache_config
    spec_config = vllm_config.speculative_config
    hf_config = vllm_config.model_config.hf_text_config
    max_spec_len = 1 + (
        spec_config.num_speculative_tokens if spec_config is not None else 0
    )
    ssm_dtype = get_kv_cache_torch_dtype(
        cache_config.mamba_ssm_cache_dtype, vllm_config.model_config.dtype
    )
    checks = [
        (cache_config.use_replayssm_spec, "requires --use-replayssm-spec"),
        (
            not cache_config.use_replayssm,
            "incompatible with non-spec --use-replayssm (its Triton-format "
            "ring shares the same page tuple)",
        ),
        (
            cache_config.replayssm_buffer_len == UCACHE_W_RING,
            f"requires --replayssm-buffer-len {UCACHE_W_RING} (kernel W_RING)",
        ),
        (
            max_spec_len in (4, 8),
            f"verify window T={max_spec_len} unsupported (native T in {{4,8}})",
        ),
        (
            getattr(hf_config, "linear_key_head_dim", None) == 128
            and getattr(hf_config, "linear_value_head_dim", None) == 128,
            "requires linear key/value head dims == 128",
        ),
        (
            cache_config.mamba_ssm_cache_dtype == "auto"
            or ssm_dtype in (torch.bfloat16, torch.float16),
            "requires --mamba-ssm-cache-dtype auto (resolves to the kernel "
            "state dtype, default fp16), bfloat16, or float16; an explicit "
            "dtype must match the kernel module's GDN_UCACHE_STATE_DTYPE "
            "(its wrapper asserts the pool dtype on first call)",
        ),
        _ucache_kernel_available(),
        (
            current_platform.is_cuda()
            and current_platform.get_device_capability().major >= 9,
            "requires SM90+",
        ),
    ]
    for ok, msg in checks:
        if not ok:
            raise ValueError(f"gdn_spec_backend=flashinfer_ucache: {msg}")
    return "flashinfer_ucache"


def gdn_spec_ucache_strided(vllm_config) -> bool:
    """Whether the ucache kernel uses the zero-copy strided q/k/v path.

    Default True. Set additional_config["gdn_spec_ucache_strided"]=false for
    enforce-eager debugging: the strided JIT cache key includes (B, pool), so
    arbitrary eager batch sizes would compile per size; the non-strided path
    is batch-dynamic (one cubin) at the cost of q/k/v .contiguous() copies.
    """
    additional_config = vllm_config.additional_config
    if isinstance(additional_config, dict):
        return bool(additional_config.get("gdn_spec_ucache_strided", True))
    return True
