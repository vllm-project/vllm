# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Dispatch module for Mamba selective state update (SSU) backends.

Provides a unified `selective_state_update` function that dispatches to
the Triton, FlashInfer, or CPU backend based on the configured
`MambaBackendEnum`. On CPU-only platforms (PowerPC, x86 without CUDA)
the backend defaults to 'cpu'.
"""

from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping, Sequence
from functools import cache
from inspect import signature
from typing import Any

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
    prev = tl.load(prev_num_accepted + slots, mask=valid, other=0)
    start = tl.load(ring_start + slots, mask=valid, other=0)
    previous_query_len = tl.load(prev_query_len + slots, mask=valid, other=0)
    accepted = tl.load(num_accepted_tokens + offsets, mask=mask, other=0)
    must_checkpoint = (previous_query_len > 0) & (
        prev + previous_query_len > logical_window
    )
    next_start = tl.where(
        must_checkpoint,
        (start + prev) % ring_buffer_len,
        start,
    )
    next_prev = tl.where(
        previous_query_len == 0,
        0,
        tl.where(must_checkpoint, accepted, prev + accepted),
    )
    current_query_len = tl.load(
        query_start_loc + offsets + 1, mask=mask, other=0
    ) - tl.load(query_start_loc + offsets, mask=mask, other=0)
    tl.store(ring_start + slots, next_start, mask=valid)
    tl.store(prev_num_accepted + slots, next_prev, mask=valid)
    tl.store(prev_query_len + slots, current_query_len, mask=valid)


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
    """Commit the preceding speculative window and record the current one."""
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
        from flashinfer.mamba.checkpointing_ssu import CheckpointingSSURunner
    except ImportError:
        return False
    return callable(CheckpointingSSURunner)


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
    algorithm: str = "auto",
    d_split: int | None = None,
    precompute_heads_per_cta: int = 0,
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
        d_split=d_split,
        precompute_heads_per_cta=precompute_heads_per_cta,
        algorithm=algorithm,
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


def _reinterpret_u64_as_i64(value: int) -> int:
    """Preserve a uint64 pointer bit pattern in a torch.int64 tensor."""
    return value if value < (1 << 63) else value - (1 << 64)


def _cuda_i64_ptrs(tensors: list[torch.Tensor]) -> torch.Tensor:
    return torch.tensor(
        [_reinterpret_u64_as_i64(t.data_ptr()) for t in tensors],
        dtype=torch.int64,
        device=tensors[0].device,
    )


def _cuda_i64_slot_strides(tensors: list[torch.Tensor]) -> torch.Tensor:
    return torch.tensor(
        [t.stride(0) for t in tensors],
        dtype=torch.int64,
        device=tensors[0].device,
    )


def _flashinfer_replayssm_mixers_by_group(
    kv_cache_config: KVCacheConfig,
    mamba_group_ids: Sequence[int],
    forward_context: Mapping[str, Any],
) -> list[tuple[int, list[Any]]]:
    grouped: list[tuple[int, list[Any]]] = []
    for gid in mamba_group_ids:
        mixers: list[Any] = []
        for layer_name in kv_cache_config.kv_cache_groups[gid].layer_names:
            layer = forward_context.get(layer_name)
            if layer is None:
                continue
            kv_cache = getattr(layer, "kv_cache", ())
            mamba_config = getattr(layer, "mamba_config", None)
            backend = getattr(mamba_config, "backend", None)
            if (
                getattr(layer, "use_replayssm", False)
                and backend == MambaBackendEnum.FLASHINFER
                and len(kv_cache) >= 5
            ):
                mixers.append(layer)
        if mixers:
            grouped.append((gid, mixers))
    return grouped


@cache
def _load_replayssm_materialize() -> Callable[..., None]:
    try:
        from flashinfer.mamba.replayssm_materialize import (
            replayssm_materialize,
        )
    except ImportError as e:
        raise ImportError(
            "FlashInfer ReplaySSM prefix caching requires "
            "flashinfer.mamba.replayssm_materialize"
        ) from e
    return replayssm_materialize


def materialize_replayssm_prefix(
    kv_cache_config: KVCacheConfig,
    mamba_group_ids: Sequence[int],
    forward_context: Mapping[str, Any],
    req_ids: Sequence[str],
    requests: Mapping[str, Any],
    src_cols: Sequence[int],
    dst_cols: Sequence[int],
    num_reqs: int,
) -> None:
    """Materialize FlashInfer ReplaySSM SSM state for align column copies.

    Writes destination SSM from the source checkpoint plus ring, then resets
    destination trackers. Convolution state is left to the caller.

    The caller must invoke this only for FlashInfer ReplaySSM when STP
    copies across a block column.
    """
    grouped = _flashinfer_replayssm_mixers_by_group(
        kv_cache_config, mamba_group_ids, forward_context
    )
    if not grouped:
        raise RuntimeError(
            "materialize_replayssm_prefix requires FlashInfer ReplaySSM mixers"
        )

    replayssm_materialize = _load_replayssm_materialize()
    batch_req_ids = req_ids[:num_reqs]
    for gid, mixers in grouped:
        _materialize_replayssm_prefix_group(
            mixers,
            gid,
            batch_req_ids,
            requests,
            src_cols,
            dst_cols,
            num_reqs,
            replayssm_materialize,
        )


def _materialize_replayssm_prefix_group(
    mixers: list[Any],
    gid: int,
    req_ids: Sequence[str],
    requests: Mapping[str, Any],
    src_cols: Sequence[int],
    dst_cols: Sequence[int],
    num_reqs: int,
    replayssm_materialize: Callable[..., None],
) -> None:
    ssm = mixers[0].kv_cache[1]
    x_cache = mixers[0].kv_cache[2]
    ring_start_full = mixers[0]._replayssm_ring_start
    if (
        ssm.numel() == 0
        or not ssm.is_cuda
        or ring_start_full.numel() == 0
        or x_cache.numel() == 0
        or not mixers[0].replayssm_buffer_len
    ):
        return

    src_phys = [0] * num_reqs
    dst_phys = [0] * num_reqs
    for i, req_id in enumerate(req_ids):
        src_col = src_cols[i]
        if src_col < 0:
            continue
        block_ids = requests[req_id].block_ids[gid]
        src_phys[i] = int(block_ids[src_col])
        dst_phys[i] = int(block_ids[dst_cols[i]])

    device = ssm.device
    src_row = torch.tensor(src_phys, dtype=torch.int32, device=device)
    dst_row = torch.tensor(dst_phys, dtype=torch.int32, device=device)
    copied = torch.tensor(
        [col >= 0 for col in src_cols[:num_reqs]],
        dtype=torch.bool,
        device=device,
    )
    slot_hi = ring_start_full.numel() - 1
    src_clamped = src_row.clamp(min=0, max=slot_hi)
    flush_count = torch.where(
        copied,
        mixers[0]._replayssm_prev_num_accepted[src_clamped],
        torch.tensor(-1, dtype=torch.int32, device=device),
    )
    num_layers = len(mixers)
    src_slots = src_row.unsqueeze(0).expand(num_layers, -1).contiguous()
    dst_slots = dst_row.unsqueeze(0).expand(num_layers, -1).contiguous()

    states = [m.kv_cache[1] for m in mixers]
    x_caches = [m.kv_cache[2] for m in mixers]
    dt_caches = [m.kv_cache[3] for m in mixers]
    b_caches = [m.kv_cache[4] for m in mixers]
    a_weights = [m.A for m in mixers]
    zero_table = torch.zeros(num_layers, dtype=torch.int64, device=device)
    mamba_config = mixers[0].mamba_config
    rand_seed = None
    philox_rounds = 0
    if mamba_config.enable_stochastic_rounding:
        rand_seed = torch.randint(0, 2**32, (1,), device=device, dtype=torch.int64)
        philox_rounds = mamba_config.stochastic_rounding_philox_rounds or 10

    replayssm_materialize(
        _cuda_i64_ptrs(states),
        _cuda_i64_slot_strides(states),
        _cuda_i64_ptrs(x_caches),
        _cuda_i64_slot_strides(x_caches),
        _cuda_i64_ptrs(b_caches),
        _cuda_i64_slot_strides(b_caches),
        _cuda_i64_ptrs(dt_caches),
        _cuda_i64_slot_strides(dt_caches),
        _cuda_i64_ptrs(a_weights),
        zero_table,
        zero_table,
        src_slots,
        dst_slots,
        ring_start_full[src_clamped],
        flush_count,
        state_dtype=ssm.dtype,
        input_dtype=x_cache.dtype,
        matrixA_dtype=mixers[0].A.dtype,
        dim=ssm.size(2),
        dstate=ssm.size(3),
        num_heads=ssm.size(1),
        heads_per_group=ssm.size(1) // b_caches[0].size(1),
        max_window=int(mixers[0].replayssm_buffer_len),
        ring_buffer_len=x_cache.size(2),
        rand_seed=rand_seed,
        philox_rounds=philox_rounds,
    )
    reset_replayssm_ring_trackers(
        mixers[0]._replayssm_ring_start,
        mixers[0]._replayssm_prev_num_accepted,
        mixers[0]._replayssm_prev_query_len,
        dst_row[copied],
    )


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
            from flashinfer.mamba.checkpointing_ssu import (
                CheckpointingSSURunner,
                checkpointing_ssu,
            )
        except ImportError as e:
            raise ImportError(
                "FlashInfer ReplaySSM requires a compatible flashinfer-python package"
            ) from e
        if not callable(CheckpointingSSURunner):
            raise ImportError("FlashInfer ReplaySSM requires native autotuning support")
        required_parameters = {"cu_seqlens", "max_seqlen", "enable_pdl"}
        missing_parameters = (
            required_parameters - signature(checkpointing_ssu).parameters.keys()
        )
        if missing_parameters:
            raise ImportError(
                "FlashInfer ReplaySSM requires native MTP and PDL support; missing "
                + ", ".join(sorted(missing_parameters))
            )
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
