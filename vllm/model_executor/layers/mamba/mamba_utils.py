# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import functools
import os
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, TypeAlias

import torch

import vllm.envs as envs
from vllm.config.cache import MambaDType
from vllm.config.mamba import MambaBackendEnum
from vllm.config.model import ModelDType
from vllm.distributed import divide
from vllm.logger import init_logger
from vllm.utils.torch_utils import (
    STR_DTYPE_TO_TORCH_DTYPE,
    get_kv_cache_torch_dtype,
)
from vllm.v1.attention.backends.registry import MambaAttentionBackendEnum

if TYPE_CHECKING:
    from vllm.config import VllmConfig

logger = init_logger(__name__)

ConvStateLayoutType = Literal["SD", "DS"]


GDN_REPLAYSSM_LOGICAL_WINDOW = 16
"""Committed tokens per request kept in the GDN ReplaySSM rings. Older history
is folded into the checkpoint state."""
GDN_REPLAYSSM_RING_SLOTS = 32
"""Physical u/k/G ring slots per request. FlashInfer's UMMA kernel appends up
to eight tokens after the 16-token window and takes a 32-slot ring."""


def gdn_replayssm_executed_width(num_speculative_tokens: int) -> int:
    """Return the tokens per request that one GDN ReplaySSM launch computes.

    Shorter rows are padded with no-op tokens up to this width.
    """
    if not 0 <= num_speculative_tokens <= 7:
        raise ValueError(
            "FlashInfer GDN ReplaySSM supports 0 through 7 speculative tokens; got "
            f"{num_speculative_tokens}"
        )
    return num_speculative_tokens + 1


GDN_REPLAYSSM_HMMA_WIDTHS = (4, 8)
"""Tokens per request that FlashInfer's HMMA GDN ReplaySSM kernel supports,
that is, MTP with 3 or 7 draft tokens."""


def gdn_replayssm_backend(state_dtype: torch.dtype) -> Literal["umma", "hmma"]:
    """Return the FlashInfer GDN ReplaySSM decode kernel for this GPU.

    BF16 state runs on the HMMA kernel. FP16 state runs on the UMMA kernel,
    except on NVIDIA GPUs other than SM100-class, which use the HMMA kernel.
    """
    from vllm.platforms import current_platform

    if state_dtype == torch.bfloat16 or (
        current_platform.is_cuda()
        and not current_platform.is_device_capability_family(100)
    ):
        return "hmma"
    return "umma"


_FLASHINFER_GDN_STATE_ENV = {
    torch.float16: ("fp16", ("fp16", "float16", "half")),
    torch.bfloat16: ("bf16", ("bf16", "bfloat16")),
}


def preset_flashinfer_gdn_env(mamba_ssm_cache_dtype: str) -> None:
    """Set default FlashInfer GDN variables before vLLM first imports FlashInfer.

    Building the model config imports FlashInfer, and FlashInfer reads these
    variables only then. A forked engine process keeps that import. This
    function only sets defaults, which affect only FlashInfer's GDN kernels;
    `configure_flashinfer_gdn_env` checks them for GDN models.
    """
    state_dtype = STR_DTYPE_TO_TORCH_DTYPE.get(mamba_ssm_cache_dtype)
    if state_dtype not in _FLASHINFER_GDN_STATE_ENV:
        return
    os.environ.setdefault(
        "GDN_UCACHE_STATE_DTYPE", _FLASHINFER_GDN_STATE_ENV[state_dtype][0]
    )
    if gdn_replayssm_backend(state_dtype) == "hmma":
        os.environ.setdefault("SGLANG_GDN_WY_STRIDED_QKV", "1")


def configure_flashinfer_gdn_env(state_dtype: torch.dtype, hmma: bool) -> None:
    """Set the FlashInfer GDN environment that ReplaySSM needs.

    FlashInfer reads these variables once, when `flashinfer` is first
    imported, so they must be set before vLLM starts its workers.
    `GDN_UCACHE_STATE_DTYPE` selects the state dtype of the HMMA kernel and
    the prefix materializer. The HMMA kernel also needs
    `SGLANG_GDN_WY_STRIDED_QKV=1` to read Q/K/V rows from block-strided pools.

    Raises:
        ValueError: The environment already selects another setting.

    """
    default, accepted = _FLASHINFER_GDN_STATE_ENV[state_dtype]
    state = os.environ.setdefault("GDN_UCACHE_STATE_DTYPE", default)
    if state.strip().lower() not in accepted:
        raise ValueError(
            f"GDN_UCACHE_STATE_DTYPE={state!r} conflicts with the {state_dtype} "
            "recurrent state cache that GDN ReplaySSM uses; unset it"
        )
    if hmma and os.environ.setdefault("SGLANG_GDN_WY_STRIDED_QKV", "1") == "0":
        raise ValueError(
            "SGLANG_GDN_WY_STRIDED_QKV=0 conflicts with FlashInfer's HMMA GDN "
            "ReplaySSM kernel, which reads Q/K/V from block-strided pools; unset it"
        )


def mamba_state_alignment(vllm_config: "VllmConfig") -> int:
    # FlashInfer loads state and ring caches with 16-byte cp.async instructions.
    return 16 if vllm_config.use_flashinfer_replayssm else 1


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
    def append_replayssm_ring(
        cls,
        base_dtypes: tuple[torch.dtype, ...],
        model_dtype: ModelDType | torch.dtype,
    ) -> tuple[torch.dtype, ...]:
        """Append the ReplaySSM ring dtypes to a base `(conv, ssm)` tuple:
        `(x_cache, dt_cache, B_cache)` = `(activation, fp32, activation)`.
        """
        activation_dtype = get_kv_cache_torch_dtype("auto", model_dtype)
        return (*base_dtypes, activation_dtype, torch.float32, activation_dtype)

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
    def gated_delta_net_replayssm_ring_dtypes(
        cls,
        model_dtype: ModelDType | torch.dtype,
    ) -> tuple[torch.dtype, torch.dtype, torch.dtype]:
        """Return FlashInfer GDN `(u, k, G)` ring dtypes."""
        activation_dtype = get_kv_cache_torch_dtype("auto", model_dtype)
        return (activation_dtype, activation_dtype, torch.float32)

    @classmethod
    def kda_state_dtype(
        cls,
        model_dtype: ModelDType | torch.dtype,
        mamba_cache_dtype: MambaDType,
        mamba_ssm_cache_dtype: MambaDType = "auto",
    ) -> tuple[torch.dtype, torch.dtype]:
        conv_state_dtype = get_kv_cache_torch_dtype(mamba_cache_dtype, model_dtype)
        if mamba_ssm_cache_dtype == "auto":
            recurrent_state_dtype = torch.float32
        elif mamba_ssm_cache_dtype in ("float32", "bfloat16"):
            recurrent_state_dtype = STR_DTYPE_TO_TORCH_DTYPE[mamba_ssm_cache_dtype]
        else:
            raise ValueError(
                "KDA recurrent state supports only auto, float32, and bfloat16; "
                f"got {mamba_ssm_cache_dtype}"
            )
        return (conv_state_dtype, recurrent_state_dtype)


class MambaStateShapeCalculator:
    @classmethod
    def linear_attention_state_shape(
        cls,
        num_heads: int,
        tp_size: int,
        head_dim: int,
    ) -> tuple[tuple[int, int, int]]:
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
    def append_replayssm_ring(
        cls,
        base_shapes: tuple[tuple[int, int], tuple[int, int, int]],
        n_groups: int,
        tp_world_size: int,
        logical_window: int,
        backend: MambaBackendEnum,
        num_speculative_tokens: int = 0,
    ) -> tuple[
        tuple[int, int],
        tuple[int, int, int],
        tuple[int, int, int],
        tuple[int, int],
        tuple[int, int, int],
    ]:
        """Append the physical ReplaySSM ring shapes.

        `base_shapes[1]` is `(nheads // tp, head_dim, state_size)`;
        B_cache extends `n_groups` for head shards as in `mamba2_state_shape`.
        """
        ring_buffer_len = logical_window
        if backend == MambaBackendEnum.FLASHINFER or num_speculative_tokens > 0:
            # The tracker contract keeps the live and current verify windows.
            ring_buffer_len += 1 + num_speculative_tokens
        local_nheads, head_dim, state_size = base_shapes[1]
        n_groups = n_groups + cls.extra_groups_for_head_shards(n_groups, tp_world_size)
        local_ngroups = divide(n_groups, tp_world_size)
        return (
            *base_shapes,
            (local_nheads, ring_buffer_len, head_dim),
            (local_nheads, ring_buffer_len),
            (local_ngroups, ring_buffer_len, state_size),
        )

    @classmethod
    def short_conv_state_shape(
        cls,
        tp_world_size: int,
        intermediate_size: int,
        conv_kernel: int,
        num_spec: int = 0,
    ) -> tuple[tuple[int, int]]:
        conv_dim = divide(intermediate_size, tp_world_size)
        conv_state_shape = cls._orient_conv_shape(conv_dim, conv_kernel - 1 + num_spec)
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
    def gated_delta_net_replayssm_ring_shapes(
        cls,
        tp_world_size: int,
        num_k_heads: int,
        num_v_heads: int,
        head_k_dim: int,
        head_v_dim: int,
    ) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
        """Return FlashInfer GDN `(u, k, G)` ring shapes."""
        local_k_heads = divide(num_k_heads, tp_world_size)
        local_v_heads = divide(num_v_heads, tp_world_size)
        slots = GDN_REPLAYSSM_RING_SLOTS
        return (
            (local_v_heads, slots, head_v_dim),
            (local_k_heads, slots, head_k_dim),
            (local_v_heads, slots),
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
            divide(conv_dim, tp_world_size), conv_kernel_size - 1 + num_spec
        )
        recurrent_state_shape = (divide(num_heads, tp_world_size), head_dim, head_dim)
        return (conv_state_shape, recurrent_state_shape)


@dataclass(slots=True)
class MambaCopySpec:
    """Data class specifying the memory-copy parameters for Mamba states used for
    prefix caching.

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
MambaStateCopyFuncs: TypeAlias = tuple[MambaStateCopyFunc, ...]
MambaStateCopyFuncsByType: TypeAlias = dict[
    MambaAttentionBackendEnum, MambaStateCopyFuncs
]


def get_conv_copy_spec(
    state: torch.Tensor,
    block_ids: list[int],
    cur_block_idx: int,
    num_accepted_tokens: int,
) -> MambaCopySpec:
    """Return a MambaCopySpec for copying a convolutional state slice.

    Works for both SD layout `(num_blocks, state_len, dim)` and
    DS layout `(num_blocks, dim, state_len)`.
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
