# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Dispatch module for Mamba selective state update (SSU) backends.

Provides unified ``selective_state_update`` (baseline decode) and ReplaySSM
decode entry points that dispatch to Triton / FlashInfer / CPU based on
``MambaBackendEnum``. On CPU-only platforms (PowerPC, x86 without CUDA) the
baseline SSU backend defaults to ``cpu``.

ReplaySSM backends:
  - Triton: ``write_pos`` / ``is_flush`` / ``bc_pre``
    (``selective_state_update_replayssm_triton``)
  - FlashInfer: ``ring_start`` / ``prev_num_accepted_tokens`` (+ optional
    two-kernel scratch), matching ``flashinfer.mamba.checkpointing_ssu``
    (``selective_state_update_replayssm_flashinfer``)
"""

import os
from abc import ABC, abstractmethod
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass

import torch

from vllm.config.mamba import MambaBackendEnum, MambaConfig, MambaSSUAlgorithm
from vllm.logger import init_logger
from vllm.triton_utils import tl, triton
from vllm.v1.attention.backends.registry import MambaAttentionBackendEnum
from vllm.v1.attention.backends.utils import NULL_BLOCK_ID
from vllm.v1.kv_cache_interface import KVCacheConfig, MambaSpec

logger = init_logger(__name__)

_FLASHINFER_SSU_PIPELINE_STAGES_ENV = "FLASHINFER_SSU_MAIN_PIPELINE_STAGES"
_FLASHINFER_SSU_CTA_PER_SM_ENV = "FLASHINFER_SSU_MAIN_CTA_PER_SM"


@dataclass(frozen=True)
class FlashInferReplaySSMTactic:
    algorithm: str
    pipeline_stages: int | None = None
    ctas_per_sm: int | None = None
    precompute_heads_per_cta: int = 0

    def __post_init__(self) -> None:
        if self.algorithm not in {"auto", "monolith", "two-kernel"}:
            raise ValueError(f"Unsupported ReplaySSM algorithm: {self.algorithm}")
        has_launch_config = (
            self.pipeline_stages is not None or self.ctas_per_sm is not None
        )
        if self.algorithm == "two-kernel":
            if self.pipeline_stages not in {1, 2}:
                raise ValueError("two-kernel requires pipeline_stages in {1, 2}")
            if self.ctas_per_sm is None or self.ctas_per_sm <= 0:
                raise ValueError("two-kernel requires a positive ctas_per_sm")
            if self.precompute_heads_per_cta < 0:
                raise ValueError(
                    "two-kernel requires non-negative precompute_heads_per_cta"
                )
        elif has_launch_config:
            raise ValueError(
                f"{self.algorithm} does not accept pipeline or CTA settings"
            )
        elif self.precompute_heads_per_cta != 0:
            raise ValueError(
                f"{self.algorithm} does not accept precompute_heads_per_cta"
            )

    @property
    def name(self) -> str:
        if self.algorithm != "two-kernel":
            return self.algorithm
        name = f"two_kernel_s{self.pipeline_stages}_c{self.ctas_per_sm}"
        if self.precompute_heads_per_cta:
            name += f"_h{self.precompute_heads_per_cta}"
        return name


FLASHINFER_REPLAYSSM_AUTO_TACTIC = FlashInferReplaySSMTactic("auto")


@triton.jit
def _update_replayssm_ring_trackers_kernel(
    ring_start,
    prev_num_accepted,
    state_batch_indices,
    n_slots,
    logical_window: tl.constexpr,
    ring_buffer_len: tl.constexpr,
    pad_slot_id: tl.constexpr,
    BLOCK: tl.constexpr,
) -> None:
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_slots
    slots = tl.load(state_batch_indices + offsets, mask=mask, other=pad_slot_id)
    valid = mask & (slots != pad_slot_id)
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


@triton.jit
def _reset_replayssm_ring_trackers_kernel(
    ring_start,
    prev_num_accepted,
    state_batch_indices,
    n_slots,
    pad_slot_id: tl.constexpr,
    BLOCK: tl.constexpr,
) -> None:
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_slots
    slots = tl.load(state_batch_indices + offsets, mask=mask, other=pad_slot_id)
    valid = mask & (slots != pad_slot_id)
    tl.store(ring_start + slots, 0, mask=valid)
    tl.store(prev_num_accepted + slots, 0, mask=valid)


def update_replayssm_ring_trackers(
    ring_start: torch.Tensor,
    prev_num_accepted: torch.Tensor,
    state_batch_indices: torch.Tensor,
    logical_window: int,
    pad_slot_id: int = NULL_BLOCK_ID,
) -> None:
    if ring_start.shape != prev_num_accepted.shape:
        raise ValueError("ReplaySSM tracker tensors must have matching shapes")
    if ring_start.dim() != 1:
        raise ValueError("ReplaySSM tracker tensors must be one-dimensional")
    if not ring_start.is_contiguous() or not prev_num_accepted.is_contiguous():
        raise ValueError("ReplaySSM tracker tensors must be contiguous")
    state_batch_indices = state_batch_indices.reshape(-1)
    n_slots = state_batch_indices.numel()
    if n_slots == 0:
        return
    block = 128
    _update_replayssm_ring_trackers_kernel[(triton.cdiv(n_slots, block),)](
        ring_start,
        prev_num_accepted,
        state_batch_indices,
        n_slots,
        logical_window,
        logical_window + 1,
        pad_slot_id,
        BLOCK=block,
    )


def reset_replayssm_ring_trackers(
    ring_start: torch.Tensor,
    prev_num_accepted: torch.Tensor,
    state_batch_indices: torch.Tensor,
    pad_slot_id: int = NULL_BLOCK_ID,
) -> None:
    state_batch_indices = state_batch_indices.reshape(-1)
    n_slots = state_batch_indices.numel()
    if n_slots == 0:
        return
    block = 128
    _reset_replayssm_ring_trackers_kernel[(triton.cdiv(n_slots, block),)](
        ring_start,
        prev_num_accepted,
        state_batch_indices,
        n_slots,
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


class ReplaySSMBackend(ABC):
    """Marker base for ReplaySSM decode backends."""

    def __init__(self, mamba_config: MambaConfig):
        self._mamba_config = mamba_config

    @property
    @abstractmethod
    def name(self) -> str: ...


class TritonReplaySSMBackend(ReplaySSMBackend):
    """vLLM Triton ReplaySSM (``write_pos`` / ``is_flush`` / ``bc_pre``)."""

    def __init__(self, mamba_config: MambaConfig):
        super().__init__(mamba_config)
        from vllm.model_executor.layers.mamba.ops.selective_state_update_replayssm_output_only import (  # noqa: E501
            selective_state_update_replayssm_output_only as _triton_replayssm,
        )

        self._kernel = _triton_replayssm

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
        D: torch.Tensor | None = None,
        dt_bias: torch.Tensor | None = None,
        z: torch.Tensor | None = None,
        dt_softplus: bool = False,
        x_cache: torch.Tensor | None = None,
        dt_cache: torch.Tensor | None = None,
        B_cache: torch.Tensor | None = None,
        bc_pre: torch.Tensor | None = None,
        write_pos: torch.Tensor | None = None,
        is_flush: torch.Tensor | None = None,
        max_cache_len: int = 16,
        state_batch_indices: torch.Tensor | None = None,
        null_block_id: int = NULL_BLOCK_ID,
        out: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self._kernel(
            state,
            x,
            dt,
            A,
            B,
            C,
            D=D,
            dt_bias=dt_bias,
            z=z,
            dt_softplus=dt_softplus,
            x_cache=x_cache,
            dt_cache=dt_cache,
            B_cache=B_cache,
            bc_pre=bc_pre,
            write_pos=write_pos,
            is_flush=is_flush,
            max_cache_len=max_cache_len,
            state_batch_indices=state_batch_indices,
            null_block_id=null_block_id,
            out=out,
            enable_stochastic_rounding=self._mamba_config.enable_stochastic_rounding,
            cache_philox_rounds=self._mamba_config.stochastic_rounding_philox_rounds,
        )


class FlashInferReplaySSMBackend(ReplaySSMBackend):
    """FlashInfer ``checkpointing_ssu`` ReplaySSM backend."""

    def __init__(self, mamba_config: MambaConfig):
        super().__init__(mamba_config)
        try:
            from flashinfer.mamba import checkpointing_ssu as _fi_checkpointing_ssu
        except ImportError as e:
            raise ImportError(
                "FlashInfer is required for the flashinfer ReplaySSM backend. "
                "Please install flashinfer with mamba.checkpointing_ssu support: "
                "pip install flashinfer-python"
            ) from e
        self._kernel = _fi_checkpointing_ssu
        self._algorithm = FLASHINFER_REPLAYSSM_AUTO_TACTIC.algorithm
        self._precompute_heads_per_cta = 0

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
        out: torch.Tensor,
        x_cache: torch.Tensor,
        B_cache: torch.Tensor,
        dt_cache: torch.Tensor,
        ring_start: torch.Tensor,
        prev_num_accepted_tokens: torch.Tensor,
        D: torch.Tensor | None = None,
        dt_bias: torch.Tensor | None = None,
        z: torch.Tensor | None = None,
        dt_softplus: bool = False,
        state_batch_indices: torch.Tensor | None = None,
        null_block_id: int = NULL_BLOCK_ID,
        cb_scaled: torch.Tensor | None = None,
        cumAdt_vec: torch.Tensor | None = None,
        cb_old: torch.Tensor | None = None,
        algorithm: str | None = None,
        update_trackers: bool = True,
    ) -> torch.Tensor:
        # AR decode currently passes (batch, nheads, dim); checkpointing_ssu
        # expects a predicted-token axis T. Unsqueeze T=1 here.
        if x.dim() == 3:
            x = x.unsqueeze(1)
            dt = dt.unsqueeze(1)
            B = B.unsqueeze(1)
            C = C.unsqueeze(1)
            out = out.unsqueeze(1)
            z = z.unsqueeze(1) if z is not None else None

        rand_seed = (
            torch.randint(0, 2**32, (1,), device=state.device, dtype=torch.int64)
            if self._mamba_config.enable_stochastic_rounding
            else None
        )
        indices = state_batch_indices
        if indices is not None and indices.dim() > 1:
            indices = indices[:, 0]

        result = self._kernel(
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
            z=z,
            dt_bias=dt_bias,
            dt_softplus=dt_softplus,
            state_batch_indices=indices,
            pad_slot_id=null_block_id,
            rand_seed=rand_seed,
            philox_rounds=self._mamba_config.stochastic_rounding_philox_rounds or 10,
            cb_scaled=cb_scaled,
            cumAdt_vec=cumAdt_vec,
            cb_old=cb_old,
            precompute_heads_per_cta=self._precompute_heads_per_cta,
            algorithm=self._algorithm if algorithm is None else algorithm,
        )
        if update_trackers and indices is not None:
            update_replayssm_ring_trackers(
                ring_start,
                prev_num_accepted_tokens,
                indices,
                logical_window=x_cache.size(2) - 1,
                pad_slot_id=null_block_id,
            )
        return result


@contextmanager
def use_flashinfer_replayssm_tactic(
    tactic: FlashInferReplaySSMTactic,
) -> Iterator[None]:
    """Apply a ReplaySSM launch tactic during serial warmup or graph capture."""
    backend = get_replayssm_backend()
    if not isinstance(backend, FlashInferReplaySSMBackend):
        yield
        return

    old_algorithm = backend._algorithm
    old_precompute_heads_per_cta = backend._precompute_heads_per_cta
    old_stages = os.environ.get(_FLASHINFER_SSU_PIPELINE_STAGES_ENV)
    old_ctas = os.environ.get(_FLASHINFER_SSU_CTA_PER_SM_ENV)
    backend._algorithm = tactic.algorithm
    backend._precompute_heads_per_cta = tactic.precompute_heads_per_cta
    try:
        if tactic.algorithm == "two-kernel":
            assert tactic.pipeline_stages is not None
            assert tactic.ctas_per_sm is not None
            os.environ[_FLASHINFER_SSU_PIPELINE_STAGES_ENV] = str(
                tactic.pipeline_stages
            )
            os.environ[_FLASHINFER_SSU_CTA_PER_SM_ENV] = str(tactic.ctas_per_sm)
        else:
            os.environ.pop(_FLASHINFER_SSU_PIPELINE_STAGES_ENV, None)
            os.environ.pop(_FLASHINFER_SSU_CTA_PER_SM_ENV, None)
        yield
    finally:
        backend._algorithm = old_algorithm
        backend._precompute_heads_per_cta = old_precompute_heads_per_cta
        if old_stages is None:
            os.environ.pop(_FLASHINFER_SSU_PIPELINE_STAGES_ENV, None)
        else:
            os.environ[_FLASHINFER_SSU_PIPELINE_STAGES_ENV] = old_stages
        if old_ctas is None:
            os.environ.pop(_FLASHINFER_SSU_CTA_PER_SM_ENV, None)
        else:
            os.environ[_FLASHINFER_SSU_CTA_PER_SM_ENV] = old_ctas


_REPLAYSSM_BACKEND_REGISTRY: dict[MambaBackendEnum, type[ReplaySSMBackend]] = {
    MambaBackendEnum.TRITON: TritonReplaySSMBackend,
    MambaBackendEnum.FLASHINFER: FlashInferReplaySSMBackend,
}

_replayssm_backend: ReplaySSMBackend | None = None


def initialize_replayssm_backend(
    mamba_config: MambaConfig,
    *,
    use_replayssm: bool,
) -> None:
    """Initialize the global ReplaySSM backend when ``--use-replayssm`` is set."""
    global _replayssm_backend
    if not use_replayssm:
        _replayssm_backend = None
        return

    backend = mamba_config.backend
    if backend not in _REPLAYSSM_BACKEND_REGISTRY:
        raise ValueError(
            f"--use-replayssm does not support mamba backend {backend.value!r}. "
            f"Valid options: {[b.value for b in _REPLAYSSM_BACKEND_REGISTRY]}"
        )

    backend_cls = _REPLAYSSM_BACKEND_REGISTRY[backend]
    if isinstance(_replayssm_backend, backend_cls):
        return

    _replayssm_backend = backend_cls(mamba_config)
    logger.info("Using %s ReplaySSM backend.", _replayssm_backend.name)


def get_replayssm_backend() -> ReplaySSMBackend:
    """Get the current ReplaySSM backend. Raises if not initialized."""
    if _replayssm_backend is None:
        raise RuntimeError(
            "ReplaySSM backend has not been initialized. "
            "Call initialize_mamba_ssu_backend() with use_replayssm=True first."
        )
    return _replayssm_backend


def selective_state_update_replayssm_triton(
    state: torch.Tensor,
    x: torch.Tensor,
    dt: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    D: torch.Tensor | None = None,
    dt_bias: torch.Tensor | None = None,
    z: torch.Tensor | None = None,
    dt_softplus: bool = False,
    x_cache: torch.Tensor | None = None,
    dt_cache: torch.Tensor | None = None,
    B_cache: torch.Tensor | None = None,
    bc_pre: torch.Tensor | None = None,
    write_pos: torch.Tensor | None = None,
    is_flush: torch.Tensor | None = None,
    max_cache_len: int = 16,
    state_batch_indices: torch.Tensor | None = None,
    null_block_id: int = NULL_BLOCK_ID,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Triton ReplaySSM decode (``write_pos`` / ``is_flush`` / ``bc_pre``)."""
    backend = get_replayssm_backend()
    if not isinstance(backend, TritonReplaySSMBackend):
        raise RuntimeError(
            "selective_state_update_replayssm_triton is the Triton ReplaySSM "
            f"entry point; current backend is {backend.name!r}. Use "
            "selective_state_update_replayssm_flashinfer for FlashInfer."
        )
    return backend(
        state,
        x,
        dt,
        A,
        B,
        C,
        D=D,
        dt_bias=dt_bias,
        z=z,
        dt_softplus=dt_softplus,
        x_cache=x_cache,
        dt_cache=dt_cache,
        B_cache=B_cache,
        bc_pre=bc_pre,
        write_pos=write_pos,
        is_flush=is_flush,
        max_cache_len=max_cache_len,
        state_batch_indices=state_batch_indices,
        null_block_id=null_block_id,
        out=out,
    )


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
    z: torch.Tensor | None = None,
    dt_softplus: bool = False,
    state_batch_indices: torch.Tensor | None = None,
    null_block_id: int = NULL_BLOCK_ID,
    cb_scaled: torch.Tensor | None = None,
    cumAdt_vec: torch.Tensor | None = None,
    cb_old: torch.Tensor | None = None,
    algorithm: str | None = None,
    update_trackers: bool = True,
) -> torch.Tensor:
    """FlashInfer ReplaySSM decode (``checkpointing_ssu``)."""
    backend = get_replayssm_backend()
    if not isinstance(backend, FlashInferReplaySSMBackend):
        raise RuntimeError(
            "selective_state_update_replayssm_flashinfer requires the "
            f"flashinfer ReplaySSM backend; current backend is {backend.name!r}."
        )
    return backend(
        state,
        x,
        dt,
        A,
        B,
        C,
        out,
        x_cache,
        B_cache,
        dt_cache,
        ring_start,
        prev_num_accepted_tokens,
        D=D,
        dt_bias=dt_bias,
        z=z,
        dt_softplus=dt_softplus,
        state_batch_indices=state_batch_indices,
        null_block_id=null_block_id,
        cb_scaled=cb_scaled,
        cumAdt_vec=cumAdt_vec,
        cb_old=cb_old,
        algorithm=algorithm,
        update_trackers=update_trackers,
    )


def initialize_mamba_ssu_backend(
    mamba_config: MambaConfig,
    kv_cache_config: KVCacheConfig,
    *,
    use_replayssm: bool = False,
) -> None:
    """Initialize the global Mamba SSU backend (and ReplaySSM when enabled).

    No-op for baseline SSU if `kv_cache_config` contains no specs that call
    selective_state_update. Always (re)considers ReplaySSM when
    ``use_replayssm`` is set.
    """
    if any(
        isinstance(g.kv_cache_spec, MambaSpec)
        and g.kv_cache_spec.mamba_type
        in (MambaAttentionBackendEnum.MAMBA1, MambaAttentionBackendEnum.MAMBA2)
        for g in kv_cache_config.kv_cache_groups
    ):
        global _mamba_ssu_backend

        backend = mamba_config.backend

        # On CPU-only platforms (PowerPC, x86 without CUDA) Triton JIT is
        # unstable or unavailable.  Silently fall back to the CPU
        # backend unless the user explicitly chose something other than "triton".
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

        backend_cls = _BACKEND_REGISTRY[backend]
        if not isinstance(_mamba_ssu_backend, backend_cls):
            _mamba_ssu_backend = backend_cls(mamba_config)
            logger.info("Using %s Mamba SSU backend.", _mamba_ssu_backend.name)

    initialize_replayssm_backend(mamba_config, use_replayssm=use_replayssm)


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
