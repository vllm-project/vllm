# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Dispatch module for Mamba selective state update (SSU) backends.

Provides unified ``selective_state_update`` (baseline decode) and
``selective_state_update_replayssm`` (cached-input ReplaySSM decode) that
dispatch to Triton / FlashInfer / CPU based on ``MambaBackendEnum``. On
CPU-only platforms (PowerPC, x86 without CUDA) the baseline SSU backend
defaults to ``cpu``.

The FlashInfer ReplaySSM path imports ``flashinfer.mamba.checkpointing_ssu``
and reshapes T=1 AR tensors, but the vLLM ``write_pos`` / ``is_flush`` /
``bc_pre`` → FlashInfer ``ring_start`` / ``prev_num_accepted_tokens`` /
scratch contract is intentionally unfinished (see
``translate_vllm_replayssm_bookkeeping``).
"""

from abc import ABC, abstractmethod

import torch

from vllm.config.mamba import MambaBackendEnum, MambaConfig, MambaSSUAlgorithm
from vllm.logger import init_logger
from vllm.v1.attention.backends.registry import MambaAttentionBackendEnum
from vllm.v1.attention.backends.utils import NULL_BLOCK_ID
from vllm.v1.kv_cache_interface import KVCacheConfig, MambaSpec

logger = init_logger(__name__)


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
    """Abstract base class for ReplaySSM decode backends."""

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
    ) -> torch.Tensor: ...


class TritonReplaySSMBackend(ReplaySSMBackend):
    """vLLM's in-tree Triton ReplaySSM output_only kernel."""

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


def translate_vllm_replayssm_bookkeeping(
    *,
    write_pos: torch.Tensor,
    is_flush: torch.Tensor,
    state_batch_indices: torch.Tensor | None,
    max_cache_len: int,
    batch: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Map vLLM ReplaySSM host metadata to FlashInfer checkpointing_ssu args.

    vLLM (Triton ReplaySSM) provides per-decode-row:
      - ``write_pos``: ring append cursor in ``[0, max_cache_len)``
      - ``is_flush``: materialize full SSM state this step

    FlashInfer ``checkpointing_ssu`` expects per-cache-slot:
      - ``ring_start``: oldest live ring index
      - ``prev_num_accepted_tokens``: live history length to replay

    Returns:
        ``(ring_start, prev_num_accepted_tokens)`` shaped for the FlashInfer
        call (typically indexed by cache slot, not batch row).

    Raises:
        NotImplementedError: contract adapter not wired yet.
    """
    raise NotImplementedError(
        "FlashInfer ReplaySSM bookkeeping adapter is not implemented yet. "
        "Map vLLM write_pos/is_flush (and state_batch_indices) to FlashInfer "
        "ring_start/prev_num_accepted_tokens for max_cache_len="
        f"{max_cache_len}, batch={batch}."
    )


class FlashInferReplaySSMBackend(ReplaySSMBackend):
    """FlashInfer ``checkpointing_ssu`` ReplaySSM backend (contract pending)."""

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
        del bc_pre  # Triton-only scratch; FI uses its own precompute buffers.
        if write_pos is None or is_flush is None:
            raise ValueError(
                "FlashInfer ReplaySSM requires write_pos and is_flush metadata"
            )
        if out is None:
            raise ValueError("FlashInfer ReplaySSM requires a preallocated out tensor")
        if x_cache is None or dt_cache is None or B_cache is None:
            raise ValueError(
                "FlashInfer ReplaySSM requires x_cache, dt_cache, and B_cache"
            )

        # Mechanical T=1 reshape for AR decode. MTP (T>1) is out of scope until
        # ReplaySSM speculative decode is enabled.
        if x.dim() == 3:
            x_t = x.unsqueeze(1)
            dt_t = dt.unsqueeze(1)
            B_t = B.unsqueeze(1)
            C_t = C.unsqueeze(1)
            out_t = out.unsqueeze(1)
            z_t = z.unsqueeze(1) if z is not None else None
        else:
            x_t, dt_t, B_t, C_t, out_t, z_t = x, dt, B, C, out, z

        batch = x_t.shape[0]
        ring_start, prev_num_accepted = translate_vllm_replayssm_bookkeeping(
            write_pos=write_pos,
            is_flush=is_flush,
            state_batch_indices=state_batch_indices,
            max_cache_len=max_cache_len,
            batch=batch,
        )

        rand_seed = (
            torch.randint(0, 2**32, (1,), device=state.device, dtype=torch.int64)
            if self._mamba_config.enable_stochastic_rounding
            else None
        )
        indices = state_batch_indices
        if indices is not None and indices.dim() > 1:
            indices = indices[:, 0]

        return self._kernel(
            state,
            x_cache,
            B_cache,
            dt_cache,
            ring_start,
            prev_num_accepted,
            x_t,
            dt_t,
            A,
            B_t,
            C_t,
            out_t,
            D=D,
            z=z_t,
            dt_bias=dt_bias,
            dt_softplus=dt_softplus,
            state_batch_indices=indices,
            pad_slot_id=null_block_id,
            rand_seed=rand_seed,
            philox_rounds=self._mamba_config.stochastic_rounding_philox_rounds or 10,
        )


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


def selective_state_update_replayssm(
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
    """Unified dispatch for ReplaySSM selective state update."""
    return get_replayssm_backend()(
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
