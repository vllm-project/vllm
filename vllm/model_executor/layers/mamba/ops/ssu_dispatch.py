# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Dispatch module for Mamba selective state update (SSU) backends.

Provides a unified `selective_state_update` function that dispatches to
either the Triton or FlashInfer backend based on the configured
`MambaBackendEnum`. Follows SGLang's dispatch pattern adapted for vLLM.
"""

from abc import ABC, abstractmethod

import torch

from vllm.config.mamba import MambaBackendEnum, MambaConfig
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
        self._kernel = _fi_ssu

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
        )


_BACKEND_REGISTRY: dict[MambaBackendEnum, type[MambaSSUBackend]] = {
    MambaBackendEnum.TRITON: TritonSSUBackend,
    MambaBackendEnum.FLASHINFER: FlashInferSSUBackend,
}

_mamba_ssu_backend: MambaSSUBackend | None = None


def initialize_mamba_ssu_backend(
    mamba_config: MambaConfig,
    kv_cache_config: KVCacheConfig,
) -> None:
    """Initialize the global Mamba SSU backend.

    No-op if `kv_cache_config` contains no specs that call
    selective_state_update.
    """
    if not any(
        isinstance(g.kv_cache_spec, MambaSpec)
        and g.kv_cache_spec.mamba_type
        in (MambaAttentionBackendEnum.MAMBA1, MambaAttentionBackendEnum.MAMBA2)
        for g in kv_cache_config.kv_cache_groups
    ):
        return

    global _mamba_ssu_backend

    backend = mamba_config.backend
    if backend not in _BACKEND_REGISTRY:
        raise ValueError(
            f"Unknown Mamba SSU backend: {backend}. "
            f"Valid options: {list(_BACKEND_REGISTRY.keys())}"
        )

    backend_cls = _BACKEND_REGISTRY[backend]
    if isinstance(_mamba_ssu_backend, backend_cls):
        return

    _mamba_ssu_backend = backend_cls(mamba_config)
    logger.info("Using %s Mamba SSU backend.", _mamba_ssu_backend.name)


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
    *,
    x_cache: torch.Tensor,
    dt_cache: torch.Tensor,
    B_cache: torch.Tensor,
    write_pos: torch.Tensor,
    is_flush: torch.Tensor,
    bc_pre: torch.Tensor | None = None,
    route: str = "output_only",
    max_cache_len: int = 16,
    state_batch_indices: torch.Tensor | None = None,
    null_block_id: int = NULL_BLOCK_ID,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Unified ReplaySSM decode dispatch (opt-in via ``--use-replayssm``).

    Instead of writing the recurrent SSM state back to HBM each step, the
    ReplaySSM kernels keep a per-layer ring buffer of recent inputs
    (``x_cache``/``dt_cache``/``B_cache``) plus a cursor (``write_pos``) and
    replay them to recompute the output, flushing the rebuilt state to the
    checkpoint only every ``max_cache_len`` steps (``is_flush``). This trades a
    little recompute for much less HBM traffic at batch-1 decode.

    All replay decode paths (Mamba2 today; GDN as a follow-up) funnel through
    this single dispatch rather than duplicating a decode kernel per
    architecture.

    Args:
        route: ``"output_only"`` (inner-product, default; needs ``bc_pre``
            scratch) or ``"state_and_output"`` (outer-product, rebuilds the
            full state every step).
    """
    if route == "output_only":
        from vllm.model_executor.layers.mamba.ops.selective_state_update_replayssm_output_only import (  # noqa: E501
            selective_state_update_replayssm_output_only,
        )

        if bc_pre is None:
            raise ValueError(
                "ReplaySSM 'output_only' route requires the bc_pre scratch tensor"
            )
        return selective_state_update_replayssm_output_only(
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
    if route == "state_and_output":
        from vllm.model_executor.layers.mamba.ops.selective_state_update_replayssm_state_and_output import (  # noqa: E501
            selective_state_update_replayssm_state_and_output,
        )

        return selective_state_update_replayssm_state_and_output(
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
            write_pos=write_pos,
            is_flush=is_flush,
            max_cache_len=max_cache_len,
            state_batch_indices=state_batch_indices,
            null_block_id=null_block_id,
            out=out,
        )
    raise ValueError(
        f"Unknown ReplaySSM route: {route!r}. "
        "Valid options: 'output_only', 'state_and_output'"
    )
