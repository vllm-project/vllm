# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The ReplaySSM decode planner and FlashInfer ReplaySSM entry. The plain SSU
decode kernels are in `linear_attention/mamba2/decode/`."""

import math
from collections.abc import Callable
from functools import cache

import torch

from vllm.config.cache import CacheConfig
from vllm.config.mamba import MambaBackendEnum, MambaConfig
from vllm.logger import init_logger
from vllm.triton_utils import tl, triton
from vllm.v1.attention.backends.registry import MambaAttentionBackendEnum
from vllm.v1.attention.backends.utils import NULL_BLOCK_ID
from vllm.v1.kv_cache_interface import KVCacheConfig, MambaSpec

logger = init_logger(__name__)


@triton.jit(
    do_not_specialize=["n_rows", "num_states", "state_batch_indices_stride"],
    do_not_specialize_on_alignment=["state_batch_indices", "seq_lens"],
)
def _plan_replayssm_decode_kernel(
    ring_pos,
    state_batch_indices,
    seq_lens,
    write_pos,
    is_flush,
    state_batch_indices_stride,
    n_rows,
    num_states,
    buffer_len: tl.constexpr,
    align_block_size: tl.constexpr,
    pad_slot_id: tl.constexpr,
    BLOCK: tl.constexpr,
) -> None:
    rows = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = rows < n_rows
    slots = tl.load(
        state_batch_indices + rows * state_batch_indices_stride,
        mask=mask,
        other=pad_slot_id,
    )
    valid = mask & (slots != pad_slot_id) & (slots >= 0) & (slots < num_states)
    pos = tl.load(ring_pos + slots, mask=valid, other=0)
    pos = tl.minimum(pos, buffer_len - 1)
    flush = pos == buffer_len - 1
    if align_block_size > 0:
        # Standard decode rows carry one token: num_computed = seq_len - 1.
        seq_len = tl.load(seq_lens + rows, mask=valid, other=1)
        # The first step of a block restarts that block's empty ring, and the
        # step completing a block flushes its exact state for prefix caching.
        pos = tl.where((seq_len - 1) % align_block_size == 0, 0, pos)
        flush = (pos == buffer_len - 1) | (seq_len % align_block_size == 0)
    pos = tl.where(valid, pos, 0)
    flush = valid & flush
    tl.store(write_pos + rows, pos, mask=mask)
    tl.store(is_flush + rows, flush.to(tl.int8), mask=mask)
    tl.store(ring_pos + slots, tl.where(flush, 0, pos + 1), mask=valid)


def plan_replayssm_decode(
    ring_pos: torch.Tensor,
    state_batch_indices: torch.Tensor,
    write_pos: torch.Tensor,
    is_flush: torch.Tensor,
    buffer_len: int,
    seq_lens: torch.Tensor | None = None,
    align_block_size: int = 0,
    pad_slot_id: int = NULL_BLOCK_ID,
) -> None:
    """Plan one Triton ReplaySSM standard-decode step on the GPU.

    `ring_pos[slot]` counts the tokens buffered since the slot's last
    full-state write. For each decode row this writes the step's
    `write_pos`/`is_flush` (read by every layer of the cache group) and
    advances the slot's counter, so the runner supplies no ring metadata.
    Prefill resets the counters in the same layer.

    With `align_block_size` > 0 (prefix caching, see
    `get_replayssm_align_block_size`), the ring restarts at the first step of
    every `align_block_size` tokens and flushes on the step completing them;
    `seq_lens` must then hold each row's sequence length.
    """
    if state_batch_indices.dim() > 1:
        state_batch_indices = state_batch_indices[:, 0]
    n_rows = state_batch_indices.numel()
    if n_rows == 0:
        return
    assert write_pos.numel() >= n_rows and is_flush.numel() >= n_rows
    if align_block_size > 0:
        assert seq_lens is not None and seq_lens.numel() >= n_rows
    block = 128
    _plan_replayssm_decode_kernel[(triton.cdiv(n_rows, block),)](
        ring_pos,
        state_batch_indices,
        seq_lens if align_block_size > 0 else state_batch_indices,
        write_pos,
        is_flush,
        state_batch_indices.stride(0),
        n_rows,
        ring_pos.numel(),
        buffer_len,
        align_block_size,
        pad_slot_id,
        BLOCK=block,
    )


def get_replayssm_align_block_size(cache_config: CacheConfig, block_size: int) -> int:
    """`plan_replayssm_decode` alignment for a Mamba group with prefix caching.

    A prefix-cache hit may resume a request at any prefix match unit inside a
    Mamba block, as a one-token decode in a copied block whose ring counter
    belongs to another request. Restarting the ring on every unit makes that
    step start from the exact cached state; with a unit that is a multiple of
    the ring length this adds no flushes.
    """
    if cache_config.prefix_match_unit is None:
        return block_size
    return math.gcd(block_size, cache_config.prefix_match_unit)


_flashinfer_replayssm_kernel: Callable[..., torch.Tensor] | None = None


@cache
def flashinfer_replayssm_autotune_supported() -> bool:
    """Return True when FlashInfer exposes ReplaySSM autotuning."""
    try:
        from flashinfer.mamba.checkpointing_ssu import (  # noqa: F401
            CheckpointingSSURunner,
        )
    except ImportError:
        return False
    return True


def selective_state_update_replayssm_flashinfer(
    state: torch.Tensor,
    x: torch.Tensor,
    dt: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    out: torch.Tensor,
    x_cache: torch.Tensor,
    B_cache: torch.Tensor,
    dt_cache: torch.Tensor,
    ring_start: torch.Tensor,
    prev_num_accepted_tokens: torch.Tensor,
    D: torch.Tensor | None = None,
    dt_bias: torch.Tensor | None = None,
    dt_softplus: bool = False,
    state_batch_indices: torch.Tensor | None = None,
    null_block_id: int = NULL_BLOCK_ID,
    scratch: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
    enable_stochastic_rounding: bool = False,
    stochastic_rounding_philox_rounds: int = 0,
    rand_seed: torch.Tensor | None = None,
    cu_seqlens: torch.Tensor | None = None,
    max_seqlen: int | None = None,
    enable_pdl: bool = False,
) -> torch.Tensor:
    """Run FlashInfer checkpointing SSU with model-owned tracker metadata.

    With stochastic rounding, `rand_seed` is a one-element int64 CUDA tensor
    refreshed by the caller each step; without one, a seed is drawn here.
    """
    if _flashinfer_replayssm_kernel is None:
        raise RuntimeError(
            "FlashInfer ReplaySSM has not been initialized. "
            "Call initialize_mamba_ssu_backend() with use_replayssm=True."
        )

    if x.dim() == 3:
        dim = 0 if cu_seqlens is not None else 1
        x = x.unsqueeze(dim)
        dt = dt.unsqueeze(dim)
        B = B.unsqueeze(dim)
        C = C.unsqueeze(dim)
        out = out.unsqueeze(dim)

    indices = state_batch_indices
    if indices is not None and indices.dim() > 1:
        indices = indices[:, 0]

    cb_scaled = cumAdt_vec = cb_old = None
    if scratch is not None:
        cb_scaled, cumAdt_vec, cb_old = scratch

    if not enable_stochastic_rounding:
        rand_seed = None
    elif rand_seed is None:
        rand_seed = torch.randint(
            0, 2**32, (1,), device=state.device, dtype=torch.int64
        )
    return _flashinfer_replayssm_kernel(
        state,
        x_cache,
        B_cache,
        dt_cache,
        ring_start,
        prev_num_accepted_tokens,
        x,
        dt,
        A,
        B,
        C,
        out,
        D=D,
        dt_bias=dt_bias,
        dt_softplus=dt_softplus,
        state_batch_indices=indices,
        pad_slot_id=null_block_id,
        rand_seed=rand_seed,
        philox_rounds=stochastic_rounding_philox_rounds or 10,
        cu_seqlens=cu_seqlens,
        max_seqlen=max_seqlen,
        enable_pdl=enable_pdl,
        cb_scaled=cb_scaled,
        cumAdt_vec=cumAdt_vec,
        cb_old=cb_old,
    )


def initialize_mamba_ssu_backend(
    mamba_config: MambaConfig,
    kv_cache_config: KVCacheConfig,
    *,
    use_replayssm: bool = False,
) -> None:
    """Load FlashInfer's ReplaySSM kernel when ReplaySSM decodes on FlashInfer.

    Layers select their SSU kernels themselves (`select_mamba_ssu_kernel`);
    this is removed once ReplaySSM is selected the same way.
    """
    if not any(
        isinstance(spec, MambaSpec)
        and spec.mamba_type == MambaAttentionBackendEnum.MAMBA2
        for g in kv_cache_config.kv_cache_groups
        for spec in g.kv_cache_specs.values()
    ):
        return

    global _flashinfer_replayssm_kernel
    _flashinfer_replayssm_kernel = None
    if not use_replayssm:
        return
    backend = mamba_config.backend or MambaBackendEnum.TRITON
    if backend == MambaBackendEnum.FLASHINFER:
        try:
            from flashinfer.mamba.checkpointing_ssu import checkpointing_ssu
        except ImportError as e:
            raise ImportError(
                "FlashInfer ReplaySSM requires a compatible flashinfer-python package"
            ) from e
        _flashinfer_replayssm_kernel = checkpointing_ssu
    logger.info("Using %s ReplaySSM backend.", backend.value)
