# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Startup selection and dispatch for Mamba2 SSD prefill scans."""

from abc import ABC, abstractmethod

import torch

from vllm.config.mamba import MambaBackendEnum, MambaConfig
from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.utils.flashinfer import is_flashinfer_ssd_supported
from vllm.v1.attention.backends.registry import MambaAttentionBackendEnum
from vllm.v1.kv_cache_interface import KVCacheConfig, MambaSpec

logger = init_logger(__name__)


class MambaSSDBackend(ABC):
    @property
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    def __call__(
        self,
        x: torch.Tensor,
        dt: torch.Tensor,
        A: torch.Tensor,
        B: torch.Tensor,
        C: torch.Tensor,
        cu_seqlens: torch.Tensor,
        out: torch.Tensor,
        state_dtype: torch.dtype,
        D: torch.Tensor | None = None,
        dt_bias: torch.Tensor | None = None,
        initial_states: torch.Tensor | None = None,
        dt_softplus: bool = False,
        seq_idx: torch.Tensor | None = None,
        checkpoint_chunk_indices: torch.Tensor | None = None,
        chunk_size: int | None = None,
        cu_chunk_seqlens: torch.Tensor | None = None,
        last_chunk_indices: torch.Tensor | None = None,
        checkpoint_token_indices: torch.Tensor | None = None,
        checkpoint_states: torch.Tensor | None = None,
        final_states_out: torch.Tensor | None = None,
        final_state_indices: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """Return the final states, or None if already written into
        `final_states_out`, and the checkpoint states."""


class TritonSSDBackend(MambaSSDBackend):
    def __init__(self):
        from .ssd_combined import mamba_chunk_scan_combined_varlen

        self._kernel = mamba_chunk_scan_combined_varlen

    @property
    def name(self) -> str:
        return "triton"

    def __call__(
        self,
        x: torch.Tensor,
        dt: torch.Tensor,
        A: torch.Tensor,
        B: torch.Tensor,
        C: torch.Tensor,
        cu_seqlens: torch.Tensor,
        out: torch.Tensor,
        state_dtype: torch.dtype,
        D: torch.Tensor | None = None,
        dt_bias: torch.Tensor | None = None,
        initial_states: torch.Tensor | None = None,
        dt_softplus: bool = False,
        seq_idx: torch.Tensor | None = None,
        checkpoint_chunk_indices: torch.Tensor | None = None,
        chunk_size: int | None = None,
        cu_chunk_seqlens: torch.Tensor | None = None,
        last_chunk_indices: torch.Tensor | None = None,
        checkpoint_token_indices: torch.Tensor | None = None,
        checkpoint_states: torch.Tensor | None = None,
        final_states_out: torch.Tensor | None = None,
        final_state_indices: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        has_checkpoints = checkpoint_chunk_indices is not None
        # The kernel writes final states directly only when it returns no
        # chunk states.
        states = self._kernel(
            x,
            dt,
            A,
            B,
            C,
            chunk_size,
            cu_seqlens=cu_seqlens,
            cu_chunk_seqlens=cu_chunk_seqlens,
            last_chunk_indices=last_chunk_indices,
            seq_idx=seq_idx,
            out=out,
            D=D,
            dt_bias=dt_bias,
            initial_states=initial_states,
            dt_softplus=dt_softplus,
            return_intermediate_states=has_checkpoints,
            state_dtype=state_dtype,
            final_states_out=None if has_checkpoints else final_states_out,
            final_state_indices=None if has_checkpoints else final_state_indices,
        )
        if not has_checkpoints:
            return states, None
        return states[last_chunk_indices], states


class FlashInferSSDBackend(MambaSSDBackend):
    def __init__(self):
        try:
            from flashinfer.mamba import ssd_combined_fwd
        except ImportError as error:
            raise ImportError(
                "FlashInfer SSD requires flashinfer-python with ssd_combined_fwd."
            ) from error
        self._kernel = ssd_combined_fwd

    @property
    def name(self) -> str:
        return "flashinfer"

    def __call__(
        self,
        x: torch.Tensor,
        dt: torch.Tensor,
        A: torch.Tensor,
        B: torch.Tensor,
        C: torch.Tensor,
        cu_seqlens: torch.Tensor,
        out: torch.Tensor,
        state_dtype: torch.dtype,
        D: torch.Tensor | None = None,
        dt_bias: torch.Tensor | None = None,
        initial_states: torch.Tensor | None = None,
        dt_softplus: bool = False,
        seq_idx: torch.Tensor | None = None,
        checkpoint_chunk_indices: torch.Tensor | None = None,
        chunk_size: int | None = None,
        cu_chunk_seqlens: torch.Tensor | None = None,
        last_chunk_indices: torch.Tensor | None = None,
        checkpoint_token_indices: torch.Tensor | None = None,
        checkpoint_states: torch.Tensor | None = None,
        final_states_out: torch.Tensor | None = None,
        final_state_indices: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        # FlashInfer takes the state dtype from the initial states.
        assert initial_states is not None
        _, final_states = self._kernel(
            x.unsqueeze(0),
            dt.unsqueeze(0),
            A,
            B.unsqueeze(0),
            C.unsqueeze(0),
            D=D,
            dt_bias=dt_bias,
            dt_softplus=dt_softplus,
            initial_states=initial_states,
            cu_seqlens=cu_seqlens,
            checkpoint_token_indices=checkpoint_token_indices,
            checkpoint_state_slots=checkpoint_chunk_indices,
            checkpoint_states=checkpoint_states,
            out=out.unsqueeze(0),
        )
        return final_states, checkpoint_states


class CPUSSDBackend(MambaSSDBackend):
    def __init__(self):
        from .cpu.mamba_ssm import _mamba_chunk_scan_combined_fwd_cpu

        self._kernel = _mamba_chunk_scan_combined_fwd_cpu

    @property
    def name(self) -> str:
        return "cpu"

    def __call__(
        self,
        x: torch.Tensor,
        dt: torch.Tensor,
        A: torch.Tensor,
        B: torch.Tensor,
        C: torch.Tensor,
        cu_seqlens: torch.Tensor,
        out: torch.Tensor,
        state_dtype: torch.dtype,
        D: torch.Tensor | None = None,
        dt_bias: torch.Tensor | None = None,
        initial_states: torch.Tensor | None = None,
        dt_softplus: bool = False,
        seq_idx: torch.Tensor | None = None,
        checkpoint_chunk_indices: torch.Tensor | None = None,
        chunk_size: int | None = None,
        cu_chunk_seqlens: torch.Tensor | None = None,
        last_chunk_indices: torch.Tensor | None = None,
        checkpoint_token_indices: torch.Tensor | None = None,
        checkpoint_states: torch.Tensor | None = None,
        final_states_out: torch.Tensor | None = None,
        final_state_indices: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        assert checkpoint_chunk_indices is None, (
            "The CPU SSD kernel cannot export prefill checkpoints."
        )
        final_states = self._kernel(
            x,
            dt,
            A,
            B,
            C,
            chunk_size,
            out,
            D=D,
            dt_bias=dt_bias,
            initial_states=initial_states,
            cu_seqlens=cu_seqlens,
            dt_softplus=dt_softplus,
            state_dtype=state_dtype,
        )
        return final_states, None


_BACKEND_REGISTRY: dict[MambaBackendEnum, type[MambaSSDBackend]] = {
    MambaBackendEnum.TRITON: TritonSSDBackend,
    MambaBackendEnum.FLASHINFER: FlashInferSSDBackend,
    MambaBackendEnum.CPU: CPUSSDBackend,
}


_mamba_ssd_backend: MambaSSDBackend | None = None


def initialize_mamba_ssd_backend(
    mamba_config: MambaConfig, kv_cache_config: KVCacheConfig
) -> None:
    """Initialize the Mamba2 SSD backend."""
    if not any(
        isinstance(g.kv_cache_spec, MambaSpec)
        and g.kv_cache_spec.mamba_type == MambaAttentionBackendEnum.MAMBA2
        for g in kv_cache_config.kv_cache_groups
    ):
        return

    global _mamba_ssd_backend
    backend = mamba_config.effective_prefill_backend
    if backend not in _BACKEND_REGISTRY:
        raise ValueError(
            f"Unknown Mamba SSD backend: {backend}. "
            f"Valid options: {list(_BACKEND_REGISTRY.keys())}"
        )
    if (
        backend == MambaBackendEnum.FLASHINFER
        and mamba_config.prefill_backend is None
        and not is_flashinfer_ssd_supported()
    ):
        logger.warning_once(
            "Mamba2 prefill inherited FlashInfer from --mamba-backend, but "
            "FlashInfer SSD is unavailable on this platform. Using Triton SSD. "
            "Set --mamba-prefill-backend to require a particular prefill backend."
        )
        backend = MambaBackendEnum.TRITON
    if backend == MambaBackendEnum.TRITON and current_platform.is_cpu():
        backend = MambaBackendEnum.CPU
    backend_cls = _BACKEND_REGISTRY[backend]
    if not isinstance(_mamba_ssd_backend, backend_cls):
        _mamba_ssd_backend = backend_cls()
        logger.info("Using %s Mamba SSD backend.", _mamba_ssd_backend.name)


def get_mamba_ssd_backend() -> MambaSSDBackend:
    """Get the current Mamba SSD backend. Raises if not initialized."""
    if _mamba_ssd_backend is None:
        raise RuntimeError(
            "Mamba SSD backend has not been initialized. "
            "Call initialize_mamba_ssd_backend() first."
        )
    return _mamba_ssd_backend


def mamba_chunk_scan_combined_varlen(
    x: torch.Tensor,
    dt: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    cu_seqlens: torch.Tensor,
    out: torch.Tensor,
    state_dtype: torch.dtype,
    D: torch.Tensor | None = None,
    dt_bias: torch.Tensor | None = None,
    initial_states: torch.Tensor | None = None,
    dt_softplus: bool = False,
    seq_idx: torch.Tensor | None = None,
    checkpoint_chunk_indices: torch.Tensor | None = None,
    chunk_size: int | None = None,
    cu_chunk_seqlens: torch.Tensor | None = None,
    last_chunk_indices: torch.Tensor | None = None,
    checkpoint_token_indices: torch.Tensor | None = None,
    checkpoint_states: torch.Tensor | None = None,
    final_states_out: torch.Tensor | None = None,
    final_state_indices: torch.Tensor | None = None,
) -> torch.Tensor | None:
    """Unified dispatch for the Mamba2 SSD prefill scan, writing `out` in place.

    `checkpoint_chunk_indices` follows the selected backend's layout, as built
    by `build_ssd_prefill_metadata`, and selects each sequence's row of the
    returned checkpoint states. Triton also reads `chunk_size`, `seq_idx`,
    `cu_chunk_seqlens` and `last_chunk_indices`; FlashInfer reads
    `checkpoint_token_indices`, writes `checkpoint_states`, and requires
    `initial_states`, zeros for sequences without cached state.

    Each sequence's final state is written into `final_states_out` at
    `final_state_indices`, or discarded when `final_states_out` is None.

    Returns:
        The checkpoint states, or None when no checkpoints are exported.

    """
    final_states, checkpoint_states = get_mamba_ssd_backend()(
        x,
        dt,
        A,
        B,
        C,
        cu_seqlens,
        out,
        state_dtype,
        D=D,
        dt_bias=dt_bias,
        initial_states=initial_states,
        dt_softplus=dt_softplus,
        seq_idx=seq_idx,
        checkpoint_chunk_indices=checkpoint_chunk_indices,
        chunk_size=chunk_size,
        cu_chunk_seqlens=cu_chunk_seqlens,
        last_chunk_indices=last_chunk_indices,
        checkpoint_token_indices=checkpoint_token_indices,
        checkpoint_states=checkpoint_states,
        final_states_out=final_states_out,
        final_state_indices=final_state_indices,
    )
    if final_states is not None and final_states_out is not None:
        final_states_out[final_state_indices] = final_states
    return checkpoint_states
