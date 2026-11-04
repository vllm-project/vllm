# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Dispatch module for Mamba selective state update (SSU) backends.

Provides a unified `selective_state_update` function that dispatches to
the Triton, FlashInfer, or CPU backend based on the configured
`MambaBackendEnum`. On CPU-only platforms (PowerPC, x86 without CUDA)
the backend defaults to 'cpu'.
"""

from abc import ABC, abstractmethod
from collections.abc import Callable
from functools import cache

import torch

from vllm.config.mamba import MambaBackendEnum, MambaConfig, MambaSSUAlgorithm
from vllm.logger import init_logger
from vllm.triton_utils import tl, triton
from vllm.v1.attention.backends.registry import MambaAttentionBackendEnum
from vllm.v1.attention.backends.utils import NULL_BLOCK_ID
from vllm.v1.kv_cache_interface import KVCacheConfig, MambaSpec

logger = init_logger(__name__)


@triton.jit(
    do_not_specialize=["n_slots", "state_batch_indices_stride"],
    do_not_specialize_on_alignment=["state_batch_indices"],
)
def _update_replayssm_ring_trackers_kernel(
    ring_start,
    prev_num_accepted,
    prev_query_len,
    state_batch_indices,
    state_batch_indices_stride,
    n_slots,
    num_states,
    logical_window: tl.constexpr,
    ring_buffer_len: tl.constexpr,
    pad_slot_id: tl.constexpr,
    RESET: tl.constexpr,
    BLOCK: tl.constexpr,
) -> None:
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_slots
    slots = tl.load(
        state_batch_indices + offsets * state_batch_indices_stride,
        mask=mask,
        other=pad_slot_id,
    )
    valid = mask & (slots != pad_slot_id) & (slots >= 0) & (slots < num_states)
    if RESET:
        tl.store(ring_start + slots, 0, mask=valid)
        tl.store(prev_num_accepted + slots, 0, mask=valid)
        # MTP alone reads prev_query_len. Standard decode deliberately leaves
        # it unchanged; reset clears the pending MTP window after prefill.
        tl.store(prev_query_len + slots, 0, mask=valid)
    else:
        prev = tl.load(prev_num_accepted + slots, mask=valid, other=0)
        start = tl.load(ring_start + slots, mask=valid, other=0)
        must_checkpoint = prev + 1 > logical_window
        next_start = tl.where(
            must_checkpoint,
            (start + prev) % ring_buffer_len,
            start,
        )
        next_prev = tl.where(must_checkpoint, 1, prev + 1)
        tl.store(ring_start + slots, next_start, mask=valid)
        tl.store(prev_num_accepted + slots, next_prev, mask=valid)


@triton.jit(
    do_not_specialize=["n_slots", "state_batch_indices_stride"],
    do_not_specialize_on_alignment=["state_batch_indices"],
)
def _commit_replayssm_ring_trackers_kernel(
    ring_start,
    prev_num_accepted,
    prev_query_len,
    state_batch_indices,
    num_accepted_tokens,
    query_start_loc,
    state_batch_indices_stride,
    n_slots,
    num_states,
    logical_window: tl.constexpr,
    ring_buffer_len: tl.constexpr,
    pad_slot_id: tl.constexpr,
    BLOCK: tl.constexpr,
) -> None:
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_slots
    slots = tl.load(
        state_batch_indices + offsets * state_batch_indices_stride,
        mask=mask,
        other=pad_slot_id,
    )
    valid = mask & (slots != pad_slot_id) & (slots >= 0) & (slots < num_states)
    accepted_since_checkpoint = tl.load(prev_num_accepted + slots, mask=valid, other=0)
    current_ring_start = tl.load(ring_start + slots, mask=valid, other=0)
    previous_speculative_query_len = tl.load(
        prev_query_len + slots, mask=valid, other=0
    )
    accepted_from_previous_query = tl.load(
        num_accepted_tokens + offsets, mask=mask, other=0
    )
    must_checkpoint = (previous_speculative_query_len > 0) & (
        accepted_since_checkpoint + previous_speculative_query_len > logical_window
    )
    next_start = tl.where(
        must_checkpoint,
        (current_ring_start + accepted_since_checkpoint) % ring_buffer_len,
        current_ring_start,
    )
    next_prev = tl.where(
        previous_speculative_query_len == 0,
        0,
        tl.where(
            must_checkpoint,
            accepted_from_previous_query,
            accepted_since_checkpoint + accepted_from_previous_query,
        ),
    )
    current_speculative_query_len = tl.load(
        query_start_loc + offsets + 1, mask=mask, other=0
    ) - tl.load(query_start_loc + offsets, mask=mask, other=0)
    tl.store(ring_start + slots, next_start, mask=valid)
    tl.store(prev_num_accepted + slots, next_prev, mask=valid)
    tl.store(prev_query_len + slots, current_speculative_query_len, mask=valid)


def update_replayssm_ring_trackers(
    ring_start: torch.Tensor,
    prev_num_accepted: torch.Tensor,
    prev_query_len: torch.Tensor,
    state_batch_indices: torch.Tensor,
    logical_window: int | None = None,
    ring_buffer_len: int | None = None,
    pad_slot_id: int = NULL_BLOCK_ID,
) -> None:
    """Reset selected trackers, or advance them when a window is provided."""
    if state_batch_indices.dim() > 1:
        state_batch_indices = state_batch_indices[:, 0]
    n_slots = state_batch_indices.numel()
    if n_slots == 0:
        return
    reset = logical_window is None
    if reset:
        logical_window = 0
        ring_buffer_len = 1
    else:
        assert ring_buffer_len is not None
    block = 128
    _update_replayssm_ring_trackers_kernel[(triton.cdiv(n_slots, block),)](
        ring_start,
        prev_num_accepted,
        prev_query_len,
        state_batch_indices,
        state_batch_indices.stride(0),
        n_slots,
        min(
            ring_start.numel(),
            prev_num_accepted.numel(),
            prev_query_len.numel(),
        ),
        logical_window,
        ring_buffer_len,
        pad_slot_id,
        RESET=reset,
        BLOCK=block,
    )


def reset_replayssm_ring_trackers(
    ring_start: torch.Tensor,
    prev_num_accepted: torch.Tensor,
    prev_query_len: torch.Tensor,
    state_batch_indices: torch.Tensor,
    pad_slot_id: int = NULL_BLOCK_ID,
) -> None:
    """Reset selected ReplaySSM ring trackers."""
    update_replayssm_ring_trackers(
        ring_start,
        prev_num_accepted,
        prev_query_len,
        state_batch_indices,
        pad_slot_id=pad_slot_id,
    )


def commit_replayssm_ring_trackers(
    ring_start: torch.Tensor,
    prev_num_accepted: torch.Tensor,
    prev_query_len: torch.Tensor,
    state_batch_indices: torch.Tensor,
    num_accepted_tokens: torch.Tensor,
    query_start_loc: torch.Tensor,
    logical_window: int,
    ring_buffer_len: int,
    pad_slot_id: int = NULL_BLOCK_ID,
) -> None:
    """Commit the preceding speculative window and record the current one.

    MTP evaluates a target token and its draft tokens together, but the number
    accepted from that query is available only on the next forward pass. This
    function then advances the per-request ring by the accepted prefix and
    records the current query length for the following pass. A zero
    ``prev_query_len`` means that reset/prefill left no prior MTP query to
    commit; slot validity is handled independently by the kernel mask.

    Standard single-token decode needs no delayed commit: its ReplaySSM kernel
    advances the shared ring trackers directly after every token.
    """
    if state_batch_indices.dim() > 1:
        state_batch_indices = state_batch_indices[:, 0]
    n_slots = state_batch_indices.numel()
    if n_slots == 0:
        return
    block = 128
    _commit_replayssm_ring_trackers_kernel[(triton.cdiv(n_slots, block),)](
        ring_start,
        prev_num_accepted,
        prev_query_len,
        state_batch_indices,
        num_accepted_tokens,
        query_start_loc,
        state_batch_indices.stride(0),
        n_slots,
        min(
            ring_start.numel(),
            prev_num_accepted.numel(),
            prev_query_len.numel(),
        ),
        logical_window,
        ring_buffer_len,
        pad_slot_id,
        BLOCK=block,
    )


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

    ``ring_pos[slot]`` counts the tokens buffered since the slot's last
    full-state write. For each decode row this writes the step's
    ``write_pos``/``is_flush`` (read by every layer of the cache group) and
    advances the slot's counter, so the runner supplies no ring metadata.
    Prefill resets the counters via ``reset_replayssm_ring_trackers``.

    With ``align_block_size`` > 0 (mamba ``align`` prefix caching), the ring
    restarts at each block's first step and flushes on the step completing a
    block; ``seq_lens`` must then hold each row's sequence length.
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


class MambaSSUBackend(ABC):
    """Abstract base class for Mamba SSU backends."""

    def __init__(self, mamba_config: MambaConfig):
        self._mamba_config = mamba_config

    @property
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    def __call__(
        self,
        state: torch.Tensor,
        x: torch.Tensor,
        dt: torch.Tensor,
        A: torch.Tensor,
        B: torch.Tensor,
        C: torch.Tensor,
        D: torch.Tensor,
        dt_bias: torch.Tensor,
        z: torch.Tensor | None = None,
        dt_softplus: bool = False,
        state_batch_indices: torch.Tensor | None = None,
        dst_state_batch_indices: torch.Tensor | None = None,
        null_block_id: int = NULL_BLOCK_ID,
        out: torch.Tensor | None = None,
        num_accepted_tokens: torch.Tensor | None = None,
        cu_seqlens: torch.Tensor | None = None,
        is_blackwell: bool = False,
    ) -> None: ...


class TritonSSUBackend(MambaSSUBackend):
    """Triton-based SSU backend (vLLM's default)."""

    def __init__(self, mamba_config: MambaConfig):
        super().__init__(mamba_config)
        from vllm.model_executor.layers.mamba.ops.mamba_ssm import (
            selective_state_update as _triton_selective_state_update,
        )

        self._kernel = _triton_selective_state_update

    @property
    def name(self) -> str:
        return "triton"

    def __call__(
        self,
        state: torch.Tensor,
        x: torch.Tensor,
        dt: torch.Tensor,
        A: torch.Tensor,
        B: torch.Tensor,
        C: torch.Tensor,
        D: torch.Tensor,
        dt_bias: torch.Tensor,
        z: torch.Tensor | None = None,
        dt_softplus: bool = False,
        state_batch_indices: torch.Tensor | None = None,
        dst_state_batch_indices: torch.Tensor | None = None,
        null_block_id: int = NULL_BLOCK_ID,
        out: torch.Tensor | None = None,
        num_accepted_tokens: torch.Tensor | None = None,
        cu_seqlens: torch.Tensor | None = None,
        is_blackwell: bool = False,
    ) -> None:
        self._kernel(
            state,
            x,
            dt,
            A,
            B,
            C,
            D=D,
            z=z,
            dt_bias=dt_bias,
            dt_softplus=dt_softplus,
            state_batch_indices=state_batch_indices,
            dst_state_batch_indices=dst_state_batch_indices,
            null_block_id=null_block_id,
            out=out,
            num_accepted_tokens=num_accepted_tokens,
            cu_seqlens=cu_seqlens,
            is_blackwell=is_blackwell,
            enable_stochastic_rounding=self._mamba_config.enable_stochastic_rounding,
            cache_philox_rounds=self._mamba_config.stochastic_rounding_philox_rounds,
        )


class FlashInferSSUBackend(MambaSSUBackend):
    """FlashInfer-based SSU backend."""

    def __init__(self, mamba_config: MambaConfig):
        super().__init__(mamba_config)
        try:
            from flashinfer.mamba import selective_state_update as _fi_ssu
        except ImportError as e:
            raise ImportError(
                "FlashInfer is required for the flashinfer Mamba SSU backend. "
                "Please install flashinfer (>= 0.6.4): "
                "pip install flashinfer-python"
            ) from e
        logger.info_once("Using FlashInfer Mamba SSU algorithm: %s", self._algorithm)
        self._kernel = _fi_ssu

    @property
    def _algorithm(self) -> MambaSSUAlgorithm:
        return self._mamba_config.ssu_algorithm or "auto"

    @property
    def name(self) -> str:
        return "flashinfer"

    def __call__(
        self,
        state: torch.Tensor,
        x: torch.Tensor,
        dt: torch.Tensor,
        A: torch.Tensor,
        B: torch.Tensor,
        C: torch.Tensor,
        D: torch.Tensor,
        dt_bias: torch.Tensor,
        z: torch.Tensor | None = None,
        dt_softplus: bool = False,
        state_batch_indices: torch.Tensor | None = None,
        dst_state_batch_indices: torch.Tensor | None = None,
        null_block_id: int = NULL_BLOCK_ID,
        out: torch.Tensor | None = None,
        num_accepted_tokens: torch.Tensor | None = None,
        cu_seqlens: torch.Tensor | None = None,
        is_blackwell: bool = False,
    ) -> None:
        rand_seed = (
            torch.randint(0, 2**32, (1,), device=state.device)
            if self._mamba_config.enable_stochastic_rounding
            else None
        )
        self._kernel(
            state,
            x,
            dt,
            A,
            B,
            C,
            D=D,
            z=z,
            dt_bias=dt_bias,
            dt_softplus=dt_softplus,
            state_batch_indices=state_batch_indices,
            dst_state_batch_indices=dst_state_batch_indices,
            cu_seqlens=cu_seqlens,
            num_accepted_tokens=num_accepted_tokens,
            cache_steps=state_batch_indices.size(-1)
            if cu_seqlens is not None and state_batch_indices is not None
            else 0,
            pad_slot_id=null_block_id,
            out=out,
            rand_seed=rand_seed,
            philox_rounds=self._mamba_config.stochastic_rounding_philox_rounds or 10,
            algorithm=self._algorithm,
        )


class CPUSSUBackend(MambaSSUBackend):
    """CPU SSU backend using the compiled C++ VSX/scalar kernel.

    On CPU-only platforms (PowerPC, x86 without CUDA) this dispatches to
    the vectorized C++ kernel registered as ``torch.ops._C.selective_state_update_cpu``.
    That kernel uses vec_op SIMD intrinsics (VSX on ppc64le, AVX2 on x86,
    scalar fallback elsewhere) and is parallelised with OpenMP across heads.

    Falls back to the pure-PyTorch implementation only if the C++ op is
    unavailable (e.g. a CPU-less build).
    """

    def __init__(self, mamba_config: MambaConfig):
        super().__init__(mamba_config)
        from vllm import _custom_ops as ops

        self._cpp_kernel = ops.selective_state_update_cpu
        logger.info("CPUSSUBackend: using compiled C++ selective_state_update kernel.")

    @property
    def name(self) -> str:
        return "cpu"

    def __call__(
        self,
        state: torch.Tensor,
        x: torch.Tensor,
        dt: torch.Tensor,
        A: torch.Tensor,
        B: torch.Tensor,
        C: torch.Tensor,
        D: torch.Tensor,
        dt_bias: torch.Tensor,
        z: torch.Tensor | None = None,
        dt_softplus: bool = False,
        state_batch_indices: torch.Tensor | None = None,
        dst_state_batch_indices: torch.Tensor | None = None,
        null_block_id: int = NULL_BLOCK_ID,
        out: torch.Tensor | None = None,
        num_accepted_tokens: torch.Tensor | None = None,
        cu_seqlens: torch.Tensor | None = None,
        is_blackwell: bool = False,
    ) -> None:
        # C++ kernel: state shape expected as (nstates, nheads, dim, dstate)
        # The kernel writes in-place into `out` and updates `state`.
        self._cpp_kernel(
            state,
            x,
            dt,
            A,
            B,
            C,
            D,
            z,
            dt_bias,
            dt_softplus,
            state_batch_indices,
            dst_state_batch_indices,
            null_block_id,
            out,
            num_accepted_tokens,
            cu_seqlens,
        )


_BACKEND_REGISTRY: dict[MambaBackendEnum, type[MambaSSUBackend]] = {
    MambaBackendEnum.TRITON: TritonSSUBackend,
    MambaBackendEnum.FLASHINFER: FlashInferSSUBackend,
    MambaBackendEnum.CPU: CPUSSUBackend,
}

_mamba_ssu_backend: MambaSSUBackend | None = None


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
    prev_query_len: torch.Tensor,
    logical_window: int,
    D: torch.Tensor | None = None,
    dt_bias: torch.Tensor | None = None,
    dt_softplus: bool = False,
    state_batch_indices: torch.Tensor | None = None,
    null_block_id: int = NULL_BLOCK_ID,
    scratch: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
    update_trackers: bool = True,
    enable_stochastic_rounding: bool = False,
    stochastic_rounding_philox_rounds: int = 0,
    cu_seqlens: torch.Tensor | None = None,
    max_seqlen: int | None = None,
    enable_pdl: bool = False,
) -> torch.Tensor:
    """Run FlashInfer checkpointing SSU and optionally advance shared trackers."""
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

    rand_seed = (
        torch.randint(0, 2**32, (1,), device=state.device, dtype=torch.int64)
        if enable_stochastic_rounding
        else None
    )
    result = _flashinfer_replayssm_kernel(
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
    if update_trackers and indices is not None:
        update_replayssm_ring_trackers(
            ring_start,
            prev_num_accepted_tokens,
            prev_query_len,
            indices,
            logical_window=logical_window,
            ring_buffer_len=x_cache.size(2),
            pad_slot_id=null_block_id,
        )
    return result


def initialize_mamba_ssu_backend(
    mamba_config: MambaConfig,
    kv_cache_config: KVCacheConfig,
    *,
    use_replayssm: bool = False,
) -> None:
    """Initialize the Mamba SSU backend and optional FlashInfer ReplaySSM."""
    if not any(
        isinstance(g.kv_cache_spec, MambaSpec)
        and g.kv_cache_spec.mamba_type
        in (MambaAttentionBackendEnum.MAMBA1, MambaAttentionBackendEnum.MAMBA2)
        for g in kv_cache_config.kv_cache_groups
    ):
        return

    global _flashinfer_replayssm_kernel, _mamba_ssu_backend
    backend = mamba_config.backend

    if backend == MambaBackendEnum.TRITON:
        from vllm.platforms import current_platform

        if current_platform.is_cpu():
            logger.info(
                "CPU platform detected: overriding Mamba SSU backend "
                "from 'triton' to 'cpu'."
            )
            backend = MambaBackendEnum.CPU

    if backend not in _BACKEND_REGISTRY:
        raise ValueError(
            f"Unknown Mamba SSU backend: {backend}. "
            f"Valid options: {list(_BACKEND_REGISTRY.keys())}"
        )
    if use_replayssm and backend not in (
        MambaBackendEnum.TRITON,
        MambaBackendEnum.FLASHINFER,
    ):
        raise ValueError(f"ReplaySSM does not support mamba backend {backend.value!r}")

    backend_cls = _BACKEND_REGISTRY[backend]
    if not isinstance(_mamba_ssu_backend, backend_cls):
        _mamba_ssu_backend = backend_cls(mamba_config)
        logger.info("Using %s Mamba SSU backend.", _mamba_ssu_backend.name)

    _flashinfer_replayssm_kernel = None
    if use_replayssm and backend == MambaBackendEnum.FLASHINFER:
        try:
            from flashinfer.mamba.checkpointing_ssu import checkpointing_ssu
        except ImportError as e:
            raise ImportError(
                "FlashInfer ReplaySSM requires a compatible flashinfer-python package"
            ) from e
        _flashinfer_replayssm_kernel = checkpointing_ssu
    if use_replayssm:
        logger.info("Using %s ReplaySSM backend.", backend.value)


def get_mamba_ssu_backend() -> MambaSSUBackend:
    """Get the current Mamba SSU backend. Raises if not initialized."""
    if _mamba_ssu_backend is None:
        raise RuntimeError(
            "Mamba SSU backend has not been initialized. "
            "Call initialize_mamba_ssu_backend() first."
        )
    return _mamba_ssu_backend


def selective_state_update(
    state: torch.Tensor,
    x: torch.Tensor,
    dt: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    D: torch.Tensor,
    dt_bias: torch.Tensor,
    z: torch.Tensor | None = None,
    dt_softplus: bool = False,
    state_batch_indices: torch.Tensor | None = None,
    dst_state_batch_indices: torch.Tensor | None = None,
    null_block_id: int = NULL_BLOCK_ID,
    out: torch.Tensor | None = None,
    num_accepted_tokens: torch.Tensor | None = None,
    cu_seqlens: torch.Tensor | None = None,
    is_blackwell: bool = False,
) -> None:
    """Unified dispatch for Mamba selective state update.

    Delegates to the initialized backend (Triton or FlashInfer).
    """
    get_mamba_ssu_backend()(
        state,
        x,
        dt,
        A,
        B,
        C,
        D,
        dt_bias,
        z=z,
        dt_softplus=dt_softplus,
        state_batch_indices=state_batch_indices,
        dst_state_batch_indices=dst_state_batch_indices,
        null_block_id=null_block_id,
        out=out,
        num_accepted_tokens=num_accepted_tokens,
        cu_seqlens=cu_seqlens,
        is_blackwell=is_blackwell,
    )
